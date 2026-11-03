"""Verify and plot a snapshot archive (no torch needed)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(path):
    z = np.load(path, allow_pickle=True); snaps = z["snapshots"]; y, act = z["y"], z["active_mask"]
    ever, pool = z["user_ever_anomalous"], z["context_pool"]; split = json.loads(str(z["split"].item()))
    n_users = int((z["node_type"] == 0).sum()); T = len(snaps)
    banned = z["banned_by_cutoff"] if "banned_by_cutoff" in z else np.zeros((n_users, T), dtype=bool)
    complete = z["forecast_complete"] if "forecast_complete" in z else np.ones(T, dtype=bool)
    assert not y[n_users:].any() and np.all(y[~act] == 0), "labels only on active users"
    assert not y[:, ~complete].any(), "incomplete forecast origins must not carry labels"
    assert not np.any((banned.astype(bool)) & (y[:n_users] > 0)), "already-banned users must be outside the risk set"
    for t, s in enumerate(snaps):
        ei = s["edge_index"]
        assert np.isfinite(s["x"]).all() and np.all(s["x"][~act[:, t]] == 0), f"window {t}: bad/inactive features"
        assert act[ei[0], t].all() and act[ei[1], t].all(), f"window {t}: edge touches inactive node"
    edges = np.array([s["edge_index"].shape[1] for s in snaps])
    print("windows:", T, "| feature dim:", snaps[0]["x"].shape[1], "| windows with <1000 edges:", np.flatnonzero(edges < 1000).tolist())
    for name, ws in split.items():
        print(f"{name}: windows={ws[0] if ws else 'none'}..{ws[-1] if ws else 'none'} positives={int(y[:n_users, ws].sum())} "
              f"active user-windows={int(act[:n_users, ws].sum())}")
    pos_per_window = y[:n_users].sum(axis=0); print("positives per window:", pos_per_window.tolist())
    need = 5
    print("test windows with < 5 positives (too few for per-window metrics / contamination k=5):",
          [t for t in split["test"] if pos_per_window[t] < need])
    ban = np.argmax(y[:n_users], axis=1); users = np.flatnonzero(ever)
    prior = np.array([act[u, :ban[u]].sum() for u in users])
    print(f"ever-banned users: {len(users)}; context candidate pool: {len(pool)}")
    if len(users):
        print(f"ever-banned users with >=3 active windows before first positive forecast: {100*np.mean(prior >= 3):.1f}%")
    print(f"forecast-complete origins: {int(complete.sum())}/{T}; positives per origin: {pos_per_window.tolist()}")
    out = Path(path).parent
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(T), edges); ax[0].set(xlabel="window", ylabel="directed edges", title="Edges per snapshot")
    ax[1].bar(range(T), pos_per_window); ax[1].set(xlabel="window", ylabel="positives", title="Anomalous users per window")
    fig.tight_layout(); fig.savefig(out / f"{Path(path).stem}_overview.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("archive", nargs="?", default="reddit_dyn/reddit_delta86400_windowed.npz")
    run(p.parse_args().archive)
