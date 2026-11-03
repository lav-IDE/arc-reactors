"""Evaluate simple activity and neighbor-change baselines on snapshot labels."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from arc_common import Archive


def user_item_sets(snapshot, n_users):
    src, dst = snapshot["edge_index"]
    keep = (src < n_users) & (dst >= n_users)
    result = [set() for _ in range(n_users)]
    for u, item in zip(src[keep], dst[keep]):
        result[int(u)].add(int(item))
    return result


def evaluate(y_parts, score_parts, window_parts):
    y = np.concatenate(y_parts); score = np.concatenate(score_parts); win = np.concatenate(window_parts)
    result = {"AUROC": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else float("nan"),
              "AUPRC": float(average_precision_score(y, score)) if y.sum() else float("nan"),
              "prevalence": float(y.mean()), "n": int(len(y)), "positives": int(y.sum())}
    per_window = {}
    for t in np.unique(win):
        m = win == t
        per_window[str(int(t))] = {
            "AUROC": float(roc_auc_score(y[m], score[m])) if np.unique(y[m]).size == 2 else None,
            "AUPRC": float(average_precision_score(y[m], score[m])) if y[m].sum() else None,
            "prevalence": float(y[m].mean()), "n": int(m.sum()), "positives": int(y[m].sum())}
    result["per_window"] = per_window
    return result


def run(path):
    ds = Archive(path)
    # The builder stores log activity and distinct-neighbor counts as the last
    # three feature columns. They are train-standardized, which preserves order.
    activity = np.stack([s["x"][:ds.n_users, -3] for s in ds.snapshots], axis=1)
    neighbors = [user_item_sets(s, ds.n_users) for s in ds.snapshots]
    names = ("recent_activity", "activity_increase", "new_neighbor_count", "neighbor_churn")
    scores = {name: np.zeros((ds.n_users, ds.T), dtype=np.float32) for name in names}
    for t in range(ds.T):
        prev = max(0, t - 1)
        scores["recent_activity"][:, t] = activity[:, t]
        scores["activity_increase"][:, t] = activity[:, t] - activity[:, prev]
        for u in range(ds.n_users):
            now, old = neighbors[t][u], neighbors[prev][u]
            union = len(now | old)
            scores["new_neighbor_count"][u, t] = len(now - old)
            scores["neighbor_churn"][u, t] = len(now ^ old) / union if union else 0.0

    windows = ds.split["test"]
    outputs = {}
    for name, values in scores.items():
        yp, sp, wp = [], [], []
        for t in windows:
            risk = ds.active[:ds.n_users, t]
            ids = np.flatnonzero(risk)
            if not len(ids):
                continue
            yp.append(ds.y[ids, t]); sp.append(values[ids, t]); wp.append(np.full(len(ids), t))
        outputs[name] = evaluate(yp, sp, wp) if yp else {"AUROC": None, "AUPRC": None, "prevalence": None}

    result = {"archive": ds.path, "task": ds.config.get("evaluation_task", "snapshot anomaly detection"),
              "split": "test", "baselines": outputs}
    out = Path(path).parent / f"snapshot_baselines_{Path(path).stem}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print("saved", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", default="reddit_dyn/reddit_delta86400_windowed.npz")
    run(parser.parse_args().archive)
