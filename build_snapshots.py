"""Build time-windowed snapshots from a JODIE-format CSV (Reddit, Wikipedia, ...).

CSV format: user_id, item_id, timestamp, state_label, <d feature columns>
(the header's fifth cell is a placeholder; rows have 4 + d cells).

Each window t gives G_t = (V, E_t, X_t) over ONE fixed node set V (users, then items).
X_t is purely *within-window* (static) so the static-ARC baseline sees no temporal
signal; all temporal information is left to Dynamic-ARC.

    python build_snapshots.py --input reddit_dataset/reddit.csv --name reddit
    python build_snapshots.py --input wikipedia_dataset/wikipedia.csv --name wikipedia
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

N_CONTEXT = 10
SEED = 42
CHUNK_ROWS = 50000
N_EXTRA = 3  # log post count, log distinct neighbours, log mean inter-event gap (users)


def read_layout(path: Path):
    with path.open("r", encoding="utf-8") as f:
        f.readline()
        first = f.readline().rstrip("\n").split(",")
    d = len(first) - 4
    if d <= 0:
        raise ValueError(f"Expected 4 metadata fields + features, found {len(first)} cells")
    return ["user_id", "item_id", "timestamp", "state_label"] + [f"f_{i+1}" for i in range(d)], d


def accumulate(flat, idx, vals):
    """flat[idx] += vals, grouped so memory stays O(chunk)."""
    uniq, inv = np.unique(idx, return_inverse=True)
    S = sp.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    flat[uniq] += S @ vals


def build(input_path, name, delta, out_dir, train_frac=0.60, val_frac=0.15,
          split_strategy="chronological", split_seed=42):
    path = Path(input_path)
    names, d = read_layout(path)

    # ---- pass 1: metadata only (fast) -------------------------------------------------
    meta = pd.read_csv(path, header=None, skiprows=1, usecols=[0, 1, 2, 3], names=names[:4])
    assert meta.timestamp.is_monotonic_increasing, "CSV must be time-sorted"
    n_users, n_items = int(meta.user_id.max()) + 1, int(meta.item_id.max()) + 1
    n_total = n_users + n_items
    tmin, tmax = float(meta.timestamp.min()), float(meta.timestamp.max())
    T = int((tmax - tmin) // delta) + 1
    print(f"rows={len(meta)} users={n_users} items={n_items} d={d} windows={T} (delta={delta}s) "
          f"positives={int((meta.state_label == 1).sum())}")
    u = meta.user_id.to_numpy(np.int64)
    it = meta.item_id.to_numpy(np.int64) + n_users
    ts = meta.timestamp.to_numpy(np.float64)
    lab = meta.state_label.to_numpy(np.int8)
    w = ((ts - tmin) // delta).astype(np.int64)
    del meta

    # ---- pass 2: feature sums per (node, window) --------------------------------------
    feat = np.zeros((n_total * T, d), dtype=np.float32)
    start = 0
    for chunk in pd.read_csv(path, header=None, skiprows=1, names=names, chunksize=CHUNK_ROWS):
        fs = chunk.iloc[:, 4:].to_numpy(np.float32)
        if np.isnan(fs).any():
            raise ValueError("NaNs in features; refusing to impute silently")
        sl = slice(start, start + len(chunk)); start += len(chunk)
        accumulate(feat, u[sl] * T + w[sl], fs)
        accumulate(feat, it[sl] * T + w[sl], fs)
    feat = feat.reshape(n_total, T, d)

    post = np.bincount(u * T + w, minlength=n_total * T) + np.bincount(it * T + w, minlength=n_total * T)
    post = post.reshape(n_total, T).astype(np.float32)
    active = post > 0

    # distinct (user,item,window) pairs -> edges, and distinct-neighbour counts
    key = (u * n_items + (it - n_users)) * T + w
    ukey, cnt = np.unique(key, return_counts=True)
    pw = ukey % T; pair = ukey // T
    pu = pair // n_items; pi = pair % n_items + n_users
    distinct = (np.bincount(pu * T + pw, minlength=n_total * T) +
                np.bincount(pi * T + pw, minlength=n_total * T)).reshape(n_total, T).astype(np.float32)

    # mean inter-event gap per (user, window), computed within window only
    order = np.lexsort((ts, w, u))
    us, ws, tss = u[order], w[order], ts[order]
    same = (us[1:] == us[:-1]) & (ws[1:] == ws[:-1])
    gap_sum = np.bincount((us[1:] * T + ws[1:])[same], weights=np.diff(tss)[same], minlength=n_users * T)
    gap_n = np.bincount((us[1:] * T + ws[1:])[same], minlength=n_users * T)
    gap = np.zeros((n_total, T), dtype=np.float32)
    gap[:n_users] = (gap_sum / np.maximum(gap_n, 1)).reshape(n_users, T)

    # Snapshot anomaly labels: a user is positive in the window containing a
    # state_label==1 post. Features and labels describe the same completed window.
    y = np.zeros((n_total, T), dtype=np.uint8)
    y[u[lab == 1], w[lab == 1]] = 1
    user_ever = (np.bincount(u[lab == 1], minlength=n_users) > 0).astype(np.uint8)

    # ---- splits (time-ordered) ---------------------------------------------------------
    n_train = max(1, int(train_frac * T)); n_val = max(1, int(val_frac * T))
    if split_strategy == "chronological":
        split = {"train": list(range(n_train)),
                 "validation": list(range(n_train, min(T, n_train + n_val))),
                 "test": list(range(min(T, n_train + n_val), T))}
    elif split_strategy == "random":
        order = np.random.default_rng(split_seed).permutation(T).tolist()
        split = {"train": order[:n_train],
                 "validation": order[n_train:n_train + n_val],
                 "test": order[n_train + n_val:]}
    else:
        raise ValueError("split_strategy must be 'chronological' or 'random'")

    # ---- node features: raw d + 3 static per-window statistics, z-scored on TRAIN rows -
    X = np.zeros((T, n_total, d + N_EXTRA), dtype=np.float32)
    for s in range(T):
        on = active[:, s]
        X[s, on, :d] = feat[on, s] / post[on, s, None]
        X[s, :, d] = np.log1p(post[:, s]); X[s, :, d + 1] = np.log1p(distinct[:, s]); X[s, :, d + 2] = np.log1p(gap[:, s])
        X[s, ~on] = 0
    rows = np.concatenate([X[s][active[:, s]] for s in split["train"]], axis=0)
    mean, std = rows.mean(axis=0), rows.std(axis=0); std[std < 1e-6] = 1
    snapshots = []
    for s in range(T):
        on = active[:, s]
        X[s, on] = np.clip((X[s, on] - mean) / std, -10, 10)   # inactive rows stay exactly 0
        m = pw == s
        src = np.r_[pu[m], pi[m]]; dst = np.r_[pi[m], pu[m]]
        wt = np.r_[cnt[m], cnt[m]].astype(np.float32)
        snapshots.append({"edge_index": np.vstack([src, dst]).astype(np.int64), "edge_weight": wt, "x": X[s].copy()})
        print(f"window {s}: active={int(on.sum())} directed edges={len(src)} positives={int(y[:, s].sum())}")

    # Select context candidates using training labels only. They may later become
    # anomalous, so test-label information is not needed for context selection.
    train_anomalous = y[:n_users, split["train"]].max(axis=1).astype(bool)
    pool = np.flatnonzero((~train_anomalous) & (active[:n_users, split["train"]].sum(axis=1) >= 3))
    contexts = {}
    for seed in range(5):
        rng = np.random.default_rng(SEED + seed)
        contexts[str(seed)] = rng.choice(pool, size=min(N_CONTEXT, len(pool)), replace=False).astype(np.int64)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    split_tag = "" if split_strategy == "chronological" else f"_random_split_seed{split_seed}"
    file = out / f"{name}_delta{delta}_windowed{split_tag}.npz"
    np.savez_compressed(file,
        node_type=np.r_[np.zeros(n_users, np.uint8), np.ones(n_items, np.uint8)],
        snapshots=np.asarray(snapshots, dtype=object), y=y, active_mask=active,
        user_ever_anomalous=user_ever,
        split=json.dumps(split), context_pool=pool,
        context_samples=np.asarray(contexts, dtype=object),
        config=json.dumps({"name": name, "delta": delta, "seed": SEED, "n_context": N_CONTEXT,
                           "n_extra": N_EXTRA, "input": str(path),
                           "label_definition": "state_label==1 event in the same snapshot window",
                           "split_strategy": split_strategy, "split_seed": split_seed,
                           "evaluation_task": "window-level anomaly detection"}),
        feature_scaler_mean=mean, feature_scaler_std=std)
    print(f"saved {file}: nodes={n_total} windows={T} context pool={len(pool)}")
    return file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reddit_dataset/reddit.csv")
    ap.add_argument("--name", default="reddit")
    ap.add_argument("--delta", type=int, default=86400)
    ap.add_argument("--split-strategy", choices=("chronological", "random"), default="chronological")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out-dir", default="reddit_dyn")
    a = ap.parse_args()
    build(a.input, a.name, a.delta, a.out_dir,
          split_strategy=a.split_strategy, split_seed=a.split_seed)
