"""Consolidate static-baseline and Dynamic-ARC JSON outputs into one CSV (+ optional README table)."""
import argparse
import csv
import json
from pathlib import Path

FIELDS = ["experiment", "AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std", "AUROC_macro_mean", "AUPRC_macro_mean", "note"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="reddit_dyn"); ap.add_argument("--readme", default=None)
    a = ap.parse_args(); d = Path(a.dir); rows = []

    def add(name, m, note=""):
        rows.append({"experiment": name, "note": note, **{k: round(m[k], 4) for k in FIELDS[1:-1] if k in m}})

    for f, name, note in (("static_arc_results.json", "static_arc (zero-shot, ARC 4 domains)", "per-snapshot ARC; seeds x contexts"),
                          ("static_union_results.json", "static_arc (trained in-domain on union graph)", "supervised reference")):
        if (d / f).exists(): add(name, json.loads((d / f).read_text()), note)
    for f in sorted(d.glob("dynamic_results_*.json")):
        r = json.loads(f.read_text()); tag = f.stem.replace("dynamic_results_", "")
        # Older result files predate explicit seed/run metadata. Keep them usable
        # alongside current runs, which record the number of seeds and contexts.
        run_note = (f"{r['seeds']} seeds x 5 contexts" if "seeds" in r else "legacy result; run count not recorded")
        robust_mode = r.get("robust_mode", "not recorded")
        for mode, m in r["variants"].items(): add(f"[{tag}] dynamic_{mode}", m, f"{run_note}; 'none' = same training, no temporal module")
        for k, m in r.get("context_sizes", {}).items(): add(f"[{tag}] context_size={k}", m, f"mode={robust_mode}")
        drift = r.get("context_lag", r.get("context_drift", {}))
        for k, m in drift.items(): add(f"[{tag}] context_lag={k}", m, f"stale context; mode={robust_mode}")
        for k, m in r.get("context_contamination", {}).items(): add(f"[{tag}] contaminated={k}/10", m, f"fixed query set; mode={robust_mode}")
    with (d / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {d / 'results.csv'} ({len(rows)} rows)")
    if a.readme and Path(a.readme).exists():
        t = ["| Experiment | AUROC | AUPRC | Note |", "|---|---:|---:|---|"] + [
            f"| {r['experiment']} | {r.get('AUROC_mean','')} ± {r.get('AUROC_std','')} | {r.get('AUPRC_mean','')} ± {r.get('AUPRC_std','')} | {r['note']} |" for r in rows]
        readme = Path(a.readme)
        original = readme.read_text(encoding="utf-8")
        marker = "## Experiment results"
        prefix = original.split(marker, 1)[0].rstrip()
        section = marker + "\n\n" + "\n".join(t) + "\n"
        readme.write_text((prefix + "\n\n" if prefix else "") + section, encoding="utf-8")


if __name__ == "__main__":
    main()
