"""Static ARC applied independently to every snapshot (RQ1 reference).

  arc-multidomain (default): ARC trained exactly as in the paper (pubmed/Flickr/questions/YelpChi),
                             zero-shot on the temporal graph. 5 seeds x 5 context samples.
  reddit-union:              ARC trained in-domain on the union of training windows (supervised
                             reference; NOT zero-shot).
Run from the ARC repo root."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from arc_common import (Archive, Eval, TrainOnlyAligner, arc_config, prepare_graph,
                        set_seed, summarize, train_arc)
from model import ARC


def union_graph(ds, aligner, device):
    tw = ds.split["train"]; n = ds.snapshots[0]["x"].shape[0]
    x = np.zeros((n, ds.snapshots[0]["x"].shape[1]), np.float32); cnt = np.zeros(n, np.float32)
    for t in tw:
        m = ds.active[:, t]; x[m] += ds.snapshots[t]["x"][m]; cnt[m] += 1
    x /= np.maximum(cnt[:, None], 1)
    keys = np.unique(np.concatenate([s["edge_index"][0] * n + s["edge_index"][1] for s in (ds.snapshots[t] for t in tw)]))
    snap = {"x": x, "edge_index": np.vstack([keys // n, keys % n]), "edge_weight": np.ones(len(keys), np.float32)}
    g = prepare_graph(snap, aligner, device)
    labels = np.zeros(n, np.float32); labels[:ds.n_users] = ds.y[:ds.n_users, tw].max(axis=1)
    g.ano_labels = torch.as_tensor(labels); return g.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", nargs="?", default="reddit_dyn/reddit_delta86400_windowed.npz")
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--trials", type=int, default=5)
    p.add_argument("--training-setup", choices=("arc-multidomain", "reddit-union"), default="arc-multidomain")
    p.add_argument("--device", default="cuda:0"); a = p.parse_args()
    ds = Archive(a.archive); out_dir = Path(a.archive).parent; cfg = arc_config()
    aligner = TrainOnlyAligner(64).fit(ds.snapshots, ds.active, ds.split["train"])
    test_graphs = {t: prepare_graph(ds.snapshots[t], aligner, a.device, cfg["num_hops"]) for t in ds.split["test"]}
    runs = []
    for seed in range(a.trials):
        if a.training_setup == "arc-multidomain":
            net = train_arc(cfg, a.device, seed, a.epochs, ckpt=out_dir / f"arc_encoder_seed{seed}.pt")
        else:
            set_seed(seed); net = ARC(**cfg).to(a.device); g = union_graph(ds, aligner, a.device)
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            for _ in range(a.epochs):
                net.train(); loss = net.cross_attn.get_train_loss(net(g), g.ano_labels, cfg["num_prompt"])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
        with torch.no_grad():
            for sid, sample_ids in ds.samples.items():
                ev = Eval()
                for t in ds.split["test"]:
                    emb = net(test_graphs[t].to(a.device))
                    risk = ds.active[:ds.n_users, t]
                    ids = np.asarray(sample_ids, dtype=np.int64)
                    ctx = ids[risk[ids]]
                    if len(ctx) < len(ids):
                        avail = ds.pool[risk[ds.pool] & ~np.isin(ds.pool, ctx)]
                        take = min(len(ids) - len(ctx), len(avail))
                        if take:
                            ctx = np.r_[ctx, np.random.default_rng(2026 + int(sid) * 100 + t).choice(avail, take, replace=False)]
                    q = np.setdiff1d(np.flatnonzero(risk), ctx)
                    if not len(ctx) or not len(q): continue
                    qt = torch.as_tensor(q, device=a.device); ct = torch.as_tensor(ctx, device=a.device)
                    rec = net.cross_attn.cross_attention(emb[qt], emb[ct])       # ARC's original in-context scorer
                    ev.add(ds.y[q, t], torch.linalg.vector_norm(emb[qt] - rec, dim=1).cpu().numpy(), t)
                runs.append(ev.result())
        print(f"model_seed={seed} completed {len(ds.samples)} context evaluations", flush=True)
    res = {"method": a.training_setup, "archive": a.archive,
           "task": ds.config.get("evaluation_task", "snapshot anomaly detection"),
           "epochs": a.epochs, "trials": a.trials,
           "contexts_per_trial": len(ds.samples),
           "evaluation": "same-window anomaly detection on test snapshots and active users",
           **summarize(runs), "runs": runs}
    split_tag = "window" if ds.config.get("split_strategy", "chronological") == "chronological" else f"window_random_seed{ds.config.get('split_seed', 42)}"
    out = out_dir / f"static_{a.training_setup.replace('-', '_')}_results_{split_tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8"); print("saved", out)


if __name__ == "__main__":
    main()
