"""Dynamic-ARC experiments (RQ1-RQ4) on top of a frozen ARC encoder. Run from the ARC repo root.

  # in-domain (train + test windows of one dataset, time split)
  python dynamic_experiments.py --target reddit_dyn/reddit_delta86400_windowed.npz
  # RQ3 transfer: temporal scorer trained on Wikipedia, evaluated on Reddit test windows
  python dynamic_experiments.py --source reddit_dyn/wikipedia_delta86400_windowed.npz \
                                --target reddit_dyn/reddit_delta86400_windowed.npz

ARC alignment is fitted per dataset on its own train windows (unsupervised, as in ARC). The
2048-d ARC embedding -> 128-d reducer is fitted ONCE on the SOURCE train windows and re-used
on the target, so the temporal scorer sees the same coordinates in both domains."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from arc_common import (Archive, Eval, TrainOnlyAligner, arc_config, prepare_graph,
                        set_seed, summarize, train_arc)
from dynamic_arc import MODES, DynamicARC, trajectory_batch


class Reducer:
    """PCA(128) + z-score + clip, as plain arrays so it can be saved and applied to another dataset."""
    def fit(self, rows, dims=128):
        p = PCA(n_components=dims, svd_solver="randomized", random_state=0).fit(rows)
        z = p.transform(rows); self.mu, self.comp = p.mean_, p.components_
        self.zm, self.zs = z.mean(0), np.where(z.std(0) < 1e-6, 1, z.std(0)); return self

    def transform(self, x):
        return np.clip(((x - self.mu) @ self.comp.T - self.zm) / self.zs, -20, 20).astype(np.float16)

    def save(self, path): np.savez(path, mu=self.mu, comp=self.comp, zm=self.zm, zs=self.zs)

    @classmethod
    def load(cls, path):
        r = cls(); z = np.load(path); r.mu, r.comp, r.zm, r.zs = z["mu"], z["comp"], z["zm"], z["zs"]; return r


@torch.no_grad()
def window_embeddings(net, aligner, ds, device, hops, windows=None):
    for t in (range(ds.T) if windows is None else windows):
        yield t, net(prepare_graph(ds.snapshots[t], aligner, device, hops).to(device)).float().cpu().numpy()


def fit_reducer(net, aligner, ds, device, hops, dims=128, seed=0):
    rng = np.random.default_rng(seed); per = max(1, 100000 // len(ds.split["train"])); rows = []
    for t, e in window_embeddings(net, aligner, ds, device, hops, ds.split["train"]):
        ids = np.flatnonzero(ds.active[:, t]); ids = rng.choice(ids, per, replace=False) if len(ids) > per else ids
        rows.append(e[ids])
    return Reducer().fit(np.concatenate(rows), dims)


class Bank:
    """[T, N, D] float16 window embeddings, resident on the GPU; trajectories built on demand."""
    def __init__(self, arr, device): self.x = torch.from_numpy(np.asarray(arr)).to(device); self.cache = {}

    def traj(self, t, history):
        if (t, history) not in self.cache:
            if len(self.cache) > 40: self.cache.clear()
            s = max(0, t - history + 1)
            self.cache[(t, history)] = trajectory_batch([self.x[i].float() for i in range(s, t + 1)], t - s, history)
        return self.cache[(t, history)]


def resolve_forecast_context(ids, t, ds, size, seed):
    """Use only users active and not banned as of the context cutoff."""
    eligible = ds.active[:ds.n_users, t] & ~ds.banned_by_cutoff[:, t].astype(bool)
    ids = np.asarray(ids, dtype=np.int64)
    ids = ids[eligible[ids]]
    if len(ids) < size:
        avail = ds.pool[eligible[ds.pool] & ~np.isin(ds.pool, ids)]
        take = min(size - len(ids), len(avail))
        if take:
            ids = np.r_[ids, np.random.default_rng(seed).choice(avail, take, replace=False)]
    return ids


@torch.no_grad()
def run_protocol(model, bank, ds, windows, samples, history, lag=0, fixed_ctx=None, contam=0, hold=0):
    """Score active users in `windows` against few-shot context nodes.
      lag/fixed_ctx : context taken from window t-lag (or a fixed window) -> stale context (RQ4)
      contam        : replace `contam` context nodes by anomalous users (contaminated context, RQ4)
      hold          : anomalous users removed from the query set for EVERY contamination level, so the
                      evaluated population is identical across levels."""
    dev = bank.x.device; res = []
    for sid, ids in samples.items():
        ev = Eval()
        for t in windows:
            ct = fixed_ctx if fixed_ctx is not None else max(0, t - lag)
            ctx = resolve_forecast_context(ids, ct, ds, len(ids), 2026 + int(sid) * 100 + ct)
            held = np.array([], np.int64)
            if hold:
                risk = ds.active[:ds.n_users, t] & ~ds.banned_by_cutoff[:, t].astype(bool)
                anom = np.setdiff1d(np.flatnonzero(risk & (ds.y[:ds.n_users, t] == 1)), ctx)
                held = np.random.default_rng(994 + int(sid) * 100 + t).permutation(anom)[:hold]
            if contam and len(held):
                k = min(contam, len(held)); ctx = np.r_[ctx[:len(ctx) - k], held[:k]]
            risk = ds.active[:ds.n_users, t] & ~ds.banned_by_cutoff[:, t].astype(bool)
            q = np.setdiff1d(np.flatnonzero(risk), np.r_[ctx, held])
            if not len(ctx) or not len(q): continue
            tq = bank.traj(t, history); tc = tq if ct == t else bank.traj(ct, history)
            qi = torch.as_tensor(q, dtype=torch.long, device=dev); ci = torch.as_tensor(ctx, dtype=torch.long, device=dev)
            ev.add(ds.y[q, t], model.score_with_context(tq[qi], tc[ci]).cpu().numpy(), t)
        res.append(ev.result())
    return res


def train_temporal(src, sbank, mode, seed, args):
    """Train on source train windows; keep the epoch with best source-validation AUPRC."""
    dev = args.device
    smooth_tag = "u" if args.no_smooth else "s"
    horizon_tag = f"fh{src.config.get('forecast_horizon_hours', 'legacy'):g}" if isinstance(src.config.get("forecast_horizon_hours"), (int, float)) else "fhlegacy"
    ck = Path(args.work_dir) / f"dyn_{src.stem}__{horizon_tag}_{mode}_s{seed}_ep{args.epochs}_h{args.history}_{smooth_tag}.pt"
    set_seed(seed); model = DynamicARC(sbank.x.shape[-1], mode).to(dev)
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=dev, weights_only=True)); return model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    val_samples = {k: src.samples[k] for k in list(src.samples)[:2]}
    best, best_state = -1, None
    for epoch in range(args.epochs):
        model.train(); order = list(src.split["train"]); np.random.shuffle(order); losses = []
        for t in order:
            risk = src.active[:src.n_users, t] & ~src.banned_by_cutoff[:, t].astype(bool)
            cand = src.pool[risk[src.pool]]
            if len(cand) < 10: continue
            elig = np.zeros(src.active.shape[0], dtype=bool); elig[:src.n_users] = risk
            loss = model.training_loss(sbank.traj(t, args.history), torch.as_tensor(src.y[:, t].astype(np.int64), device=dev),
                                       torch.as_tensor(elig, device=dev), torch.as_tensor(cand, device=dev))
            if loss is None: continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss))
        model.eval()
        v = np.nanmean([r["AUPRC"] for r in run_protocol(model, sbank, src, src.split["validation"], val_samples, args.history)])
        print(f"{mode} s{seed} ep{epoch+1}/{args.epochs} loss={np.mean(losses) if losses else float('nan'):.4f} val_AUPRC={v:.4f}", flush=True)
        if v > best: best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    torch.save(model.state_dict(), ck); return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="reddit_dyn/reddit_delta86400_windowed.npz")
    ap.add_argument("--source", default=None, help="training dataset (default: same as target)")
    ap.add_argument("--work-dir", default="reddit_dyn"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--history", type=int, default=5); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--robust-mode", default="both", choices=MODES)
    ap.add_argument("--no-smooth", action="store_true"); args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required (ARC's Dataset.propagated uses .cuda())")
    Path(args.work_dir).mkdir(exist_ok=True, parents=True)
    tgt = Archive(args.target); src = Archive(args.source) if args.source else tgt
    cfg = arc_config(); sm = "unsorted" if args.no_smooth else "sorted"
    net = train_arc(cfg, args.device, 0, 40, ckpt=Path(args.work_dir) / "arc_encoder_seed0.pt")

    rpath = Path(args.work_dir) / f"reducer_{src.stem}_{sm}.npz"
    if rpath.exists(): reducer = Reducer.load(rpath)
    else:
        al = TrainOnlyAligner(64).fit(src.snapshots, src.active, src.split["train"], sort_smooth=not args.no_smooth)
        reducer = fit_reducer(net, al, src, args.device, cfg["num_hops"]); reducer.save(rpath)
    banks = {}
    for ds in {src.stem: src, tgt.stem: tgt}.values():
        f = Path(args.work_dir) / f"bank_{ds.stem}__red-{src.stem}_{sm}.npy"
        if not f.exists():
            al = TrainOnlyAligner(64).fit(ds.snapshots, ds.active, ds.split["train"], sort_smooth=not args.no_smooth)
            np.save(f, np.stack([reducer.transform(e) for _, e in window_embeddings(net, al, ds, args.device, cfg["num_hops"])]))
        banks[ds.stem] = Bank(np.load(f), args.device)
    sbank, tbank = banks[src.stem], banks[tgt.stem]

    if src.config.get("forecast_horizon_hours") != tgt.config.get("forecast_horizon_hours"):
        raise ValueError("Source and target archives must use the same forecast horizon")
    results = {"source": src.path, "target": tgt.path, "history": args.history, "epochs": args.epochs,
               "forecast_horizon_hours": tgt.config.get("forecast_horizon_hours"),
               "seeds": args.seeds, "sort_smooth": not args.no_smooth, "variants": {}}
    models = {}
    for mode in args.modes:
        runs = []
        for seed in range(args.seeds):
            m = train_temporal(src, sbank, mode, seed, args); models[(mode, seed)] = m
            runs += run_protocol(m, tbank, tgt, tgt.split["test"], tgt.samples, args.history)   # seeds x 5 contexts
        results["variants"][mode] = {**summarize(runs), "n_runs": len(runs)}
        print(mode, {k: round(v, 4) for k, v in results["variants"][mode].items() if k.startswith(("AUROC_mean", "AUPRC_mean"))}, flush=True)

    rm = args.robust_mode if args.robust_mode in args.modes else args.modes[-1]; results["robust_mode"] = rm
    def sweep(samples=None, **kw):
        samples = samples or tgt.samples
        return summarize([r for sd in range(args.seeds)
                          for r in run_protocol(models[(rm, sd)], tbank, tgt, tgt.split["test"], samples, args.history, **kw)])
    # context size
    results["context_sizes"] = {}
    for k in (2, 5, 10, 20, 50):
        rng = np.random.default_rng(2026 + k); loc = {str(i): rng.choice(tgt.pool, size=min(k, len(tgt.pool)), replace=False) for i in range(5)}
        results["context_sizes"][str(k)] = sweep(samples=loc)
    # stale context: context window = t - lag, or a fixed early window
    results["context_lag"] = {str(l): sweep(lag=l) for l in (0, 1, 2, 5, 10)}
    results["context_lag"]["fixed_window_1"] = sweep(fixed_ctx=1)
    # contaminated context (query population held fixed across levels)
    results["context_contamination"] = {str(k): sweep(contam=k, hold=5) for k in (0, 1, 2, 5)}
    tag = f"{src.stem}_to_{tgt.stem}" if args.source else tgt.stem
    horizon_tag = f"fh{tgt.config.get('forecast_horizon_hours', 'legacy')}h"
    out = Path(args.work_dir) / f"dynamic_results_{tag}_{horizon_tag}_{sm}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8"); print("saved", out)


if __name__ == "__main__":
    main()
