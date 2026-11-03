"""Shared helpers for the static baseline and Dynamic-ARC. Place next to ARC's model.py/utils.py
and run everything from the ARC repo root."""
import json
import random
import sys
from pathlib import Path

import numpy as np
# Archives may be written under NumPy 2.x and read under ARC's NumPy 1.x.
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from model import ARC
from utils import Dataset, normalize_adj, sparse_mx_to_torch_sparse_tensor

TRAIN_DATASETS = ["pubmed", "Flickr", "questions", "YelpChi"]   # ARC's own training domains
ARC_CONFIG = "params/ARC_10.json"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arc_config():
    cfg = json.loads(Path(ARC_CONFIG).read_text(encoding="utf-8")); cfg["in_feats"] = 64
    return cfg


class Archive:
    """A snapshot archive written by build_snapshots.py."""
    def __init__(self, path):
        self.path = str(path); self.stem = Path(path).stem
        with np.load(path, allow_pickle=True) as z:
            self.snapshots = list(z["snapshots"]); self.y = z["y"]; self.active = z["active_mask"]
            self.anomalous = np.flatnonzero(z["user_ever_anomalous"]); self.pool = z["context_pool"]
            self.banned_by_cutoff = z["banned_by_cutoff"] if "banned_by_cutoff" in z else np.zeros_like(self.y[:int((z["node_type"] == 0).sum())])
            self.forecast_complete = z["forecast_complete"] if "forecast_complete" in z else np.ones(self.y.shape[1], dtype=bool)
            self.samples = z["context_samples"].item(); self.split = json.loads(str(z["split"].item()))
            self.node_type = z["node_type"]
            self.config = json.loads(str(z["config"].item())) if "config" in z else {}
        self.n_users = int((self.node_type == 0).sum()); self.T = len(self.snapshots)


class TrainOnlyAligner:
    """PCA -> 64-d, [0,1] range for smoothness, smoothness ordering; fitted on TRAIN windows
    only (unsupervised, per dataset, as ARC's feat_alignment is)."""
    def __init__(self, dims=64, seed=0): self.dims, self.seed = dims, seed

    def fit(self, snapshots, active, train_windows, sort_smooth=True):
        rng = np.random.default_rng(self.seed)
        per_window = max(1, 100000 // len(train_windows)); chunks = []
        for t in train_windows:
            ids = np.flatnonzero(active[:, t])
            if len(ids) > per_window: ids = rng.choice(ids, per_window, replace=False)
            chunks.append(snapshots[t]["x"][ids])
        train_x = np.concatenate(chunks, axis=0)
        self.pca = PCA(n_components=self.dims, svd_solver="randomized", random_state=self.seed).fit(train_x)
        proj = self.pca.transform(train_x)
        self.minimum, self.maximum = proj.min(axis=0), proj.max(axis=0)
        span = np.where(self.maximum > self.minimum, self.maximum - self.minimum, 1)
        smooth = np.zeros(self.dims); denom = 0
        for t in train_windows:
            src, dst = snapshots[t]["edge_index"]
            if not len(src): continue
            z = (self.pca.transform(snapshots[t]["x"]) - self.minimum) / span
            smooth += np.square(z[src] - z[dst]).sum(axis=0); denom += len(src)
        self.order = np.argsort(smooth / max(denom, 1)) if sort_smooth else np.arange(self.dims)
        return self

    def transform(self, x):
        return self.pca.transform(x)[:, self.order].astype(np.float32)


def prepare_graph(snap, aligner, device, hops=2, binary=True):
    """ARC graph interface: aligned x, sym-normalised (A+I), x_list=[X, AX, A^2 X].
    ARC's datasets use binary adjacency, so edge multiplicities are dropped by default."""
    n = snap["x"].shape[0]; src, dst = snap["edge_index"]
    w = np.ones(len(src), np.float32) if binary else snap["edge_weight"]
    adj = sp.coo_matrix((w, (src, dst)), shape=(n, n))
    adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj + sp.eye(n)))
    x = torch.as_tensor(aligner.transform(snap["x"]), dtype=torch.float32)
    data = Data(x=x, adj=adj)
    hs = [x.to(device)]
    for _ in range(hops): hs.append(torch.spmm(adj.to(device), hs[-1]))
    data.x_list = hs
    return data


def train_arc(config, device, seed, epochs=40, ckpt=None):
    """Train ARC exactly as ARC's main.py/train_test.py (4 static domains); cache the weights."""
    set_seed(seed)
    net = ARC(**config).to(device)
    if ckpt and Path(ckpt).exists():
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return net.eval()
    train = [Dataset(64, n) for n in TRAIN_DATASETS]
    for ds in train: ds.propagated(config["num_hops"])
    opt = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    for epoch in range(epochs):
        net.train()
        for ds in train:
            g = ds.graph.to(device); emb = net(g)
            loss = net.cross_attn.get_train_loss(emb, g.ano_labels, config["num_prompt"])
            opt.zero_grad(); loss.backward(); opt.step()
    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), ckpt)
    return net.eval()


def resolve_context(ids, t, active, pool, size, seed):
    """Context nodes active in window t; refill from the (never-anomalous) pool if some are inactive.
    Used by BOTH static and dynamic evaluation so the protocols are identical."""
    ids = np.asarray(ids, dtype=np.int64); ids = ids[active[ids, t]]
    if len(ids) < size:
        avail = pool[active[pool, t] & ~np.isin(pool, ids)]
        take = min(size - len(ids), len(avail))
        if take: ids = np.r_[ids, np.random.default_rng(seed).choice(avail, take, replace=False)]
    return ids


class Eval:
    """Pooled AUROC/AUPRC over all (node, window) pairs, plus macro mean over windows."""
    def __init__(self): self.y, self.s, self.w = [], [], []

    def add(self, y, s, t):
        self.y.append(np.asarray(y)); self.s.append(np.asarray(s)); self.w.append(np.full(len(y), t))

    def result(self):
        y, s, w = (np.concatenate(v) for v in (self.y, self.s, self.w))
        if len(np.unique(y)) < 2: return {k: float("nan") for k in ("AUROC", "AUPRC", "AUROC_macro", "AUPRC_macro")}
        out = {"AUROC": float(roc_auc_score(y, s)), "AUPRC": float(average_precision_score(y, s))}
        a, p = [], []
        for t in np.unique(w):
            m = w == t
            if 0 < y[m].sum() < m.sum(): a.append(roc_auc_score(y[m], s[m])); p.append(average_precision_score(y[m], s[m]))
        out["AUROC_macro"] = float(np.mean(a)) if a else float("nan"); out["AUPRC_macro"] = float(np.mean(p)) if p else float("nan")
        return out


def summarize(results):
    """list of Eval.result() dicts -> mean/std per metric."""
    out = {}
    for k in results[0]:
        v = np.array([r[k] for r in results], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(v)); out[f"{k}_std"] = float(np.nanstd(v))
    return out
