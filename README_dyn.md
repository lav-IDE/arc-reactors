# Reddit Dynamic-ARC experiments

## Prospective ban forecasting (current protocol)

The current task predicts whether an active, not-yet-banned user will have a
`state_label=1` post during a future horizon. At origin window `t`, features and
the graph use events before that window's end; the target interval is
`[window_end, window_end + horizon)`. Origins without full outcome coverage are
excluded. Train and validation origins are purged when their target interval
would cross the next split boundary. Users already banned by an origin are
removed from its risk set and cannot be normal context users.

The default horizon is 24 hours and can be varied independently of snapshot
width with `--forecast-horizon-hours`:

```powershell
python build_snapshots.py --delta 86400 --forecast-horizon-hours 24
python build_snapshots.py --delta 86400 --forecast-horizon-hours 6
python dynamic_experiments.py --target reddit_dyn/reddit_delta86400_windowed.npz --epochs 30 --seeds 3
python forecast_baselines.py reddit_dyn/reddit_delta86400_windowed.npz
```

The builder stores forecast labels, cutoff timestamps, completeness masks, and
the known-banned-by-cutoff risk mask. Rebuild the archive after changing the
horizon. The dynamic scorer checkpoints and result files include the horizon in
their names. `forecast_baselines.py` reports test AUROC/AUPRC and per-window
metrics for recent activity, activity increase, new-neighbor count, and neighbor
churn. AUPRC should be read beside test prevalence.

These changes correct the target timing but do not guarantee strong prediction.
The feature encoder still represents each full snapshot; it does not inspect
within-window events before the cutoff. A 24-hour forecast can only use activity
in completed windows, as specified above.

### First prospective 24-hour result

The rebuilt 24-hour archive has 91 future bans in 41,373 eligible test
user-windows (prevalence 0.00220). The initial GRU run used 3 training seeds and
5 context samples per seed. Simple baselines score every eligible user directly.

| Method | AUROC | AUPRC | Note |
|---|---:|---:|---|
| Recent activity | 0.6379 | 0.00443 | Direct post-activity level |
| Activity increase | 0.5089 | 0.00282 | Change from previous window |
| New-neighbor count | 0.5902 | 0.00462 | New user-item neighbors |
| Neighbor churn | 0.5415 | 0.00225 | Symmetric-difference / union |
| Dynamic-ARC GRU | 0.5420 ± 0.0215 | 0.00273 ± 0.00022 | 3 seeds × 5 contexts |

The activity and neighbor baselines have more useful ranking signal than the
current GRU; none approaches AUROC 0.80. The per-window baseline metrics are in
`reddit_dyn/forecast_baselines_reddit_delta86400_windowed_fh24.0h.json`, and the
GRU run is in `reddit_dyn/dynamic_results_reddit_delta86400_windowed_fh24.0h_sorted.json`.
These are exploratory results on one small test period with 91 positives.

## Prior retrospective results

The result table below predates the prospective forecasting protocol. It scores
the window containing the ban event using activity from that same window, so it
is retrospective and must not be compared directly with future-horizon results.
New prospective runs have not yet been added to the table.

---

## Reproduce the snapshot archives

Requirements: use the Python 3.8.20 ARCGAD environment for model runs; dataset
construction needs pandas, NumPy, and SciPy. Plotting in `sanity_checks.py` also
uses matplotlib.
From the repository root, run:

```powershell
python build_snapshots.py --delta 43200 --forecast-horizon-hours 24
python build_snapshots.py --delta 86400 --forecast-horizon-hours 24
python build_snapshots.py --delta 172800 --forecast-horizon-hours 24
python sanity_checks.py reddit_dyn/reddit_delta86400_windowed.npz
```

The builder streams the source CSV in chunks. Each NPZ contains global node type,
snapshot dictionaries (`edge_index`, duplicate-event `edge_weight`, and `x`),
future-horizon node labels and activity mask, user-ever-banned labels,
chronological 60/15/25 split, context candidates and five deterministic samples,
cutoff risk masks, forecast completeness, and config.
Use `allow_pickle=True` when loading because `snapshots` contains dictionaries.

## Input findings and assumptions

- The source has 672,447 rows, 10,000 users, 984 items, timestamps 0–2,678,390.016,
  366 positive events across 366 users, no NaNs, and is already timestamp-sorted.
- Its header has five cells, but each data row has 176 fields. The last header name
  is a placeholder; the builder interprets the remaining 172 row values as features.
- Item IDs are offset by `n_users`; node IDs remain fixed across snapshots.
- Snapshot features are built from events in each completed window; they are
  available at that window's end, the forecast origin.
- Context candidates are drawn from users active in training. At each forecast
  cutoff, already-banned users are filtered using only bans before that cutoff.
- The 172 original post features are averaged per node and window. Three added
  statistics (log event count, log distinct neighbors, and log mean inter-event
  gap) are standardized using active training-window rows only.

## What was added to ARC

- `build_snapshots.py`: builds the windowed Reddit archives at 12h, 24h, or 48h.
- `sanity_checks.py`: checks labels, context eligibility, graph sparsity, and
  prior activity; saves the dataset plots.
- `baseline_static_arc.py`: runs ARC with its original multi-domain training data
  or on the Reddit training-window union, then evaluates test snapshots.
- `dynamic_arc.py`: trajectory residual, GRU, temporal attention, and
  cross-attentive reconstruction scorer.
- `dynamic_experiments.py`: fits the encoder and runs temporal ablations and
  context robustness experiments.
- `forecast_baselines.py`: evaluates simple causal activity and neighborhood
  change baselines with pooled and per-window metrics.
- `run_experiments.py`: consolidates recorded metrics into `reddit_dyn/results.csv`.
- `reddit_dyn/`: generated snapshot archives, metrics, plots, trained weights,
  and finite cached embeddings for reproducibility.

The original ARC model, training loop, utility module, and `README.md` were left
in place. This extension uses separate files around those modules.

## ARC data interface and model runs

The local ARC implementation uses `torch_geometric.data.Data` with `x`, normalized
`adj`, `ano_labels`, and `shot_mask`; `Dataset.propagated(k)` supplies `x_list`.
`ARC.forward` maps each propagated feature matrix through its MLP and concatenates
layer residuals. `CrossAttn` reconstructs queries from normal context embeddings.
The dynamic module in `dynamic_arc.py` accepts those per-window ARC embeddings and
adds trajectory residual, GRU, or self-attention components. The embedding
reduction is fit on training-window embeddings only and standardized before the
temporal module to keep the GPU inputs finite.

Use the ARC environment from the parent `.vscode` workspace settings:

```powershell
& 'D:\DevTools\envs\ARCGAD\python.exe' baseline_static_arc.py --training-setup arc-multidomain --epochs 40 --trials 5
& 'D:\DevTools\envs\ARCGAD\python.exe' baseline_static_arc.py --training-setup reddit-union --epochs 40 --trials 5
& 'D:\DevTools\envs\ARCGAD\python.exe' dynamic_experiments.py --target reddit_dyn/reddit_delta86400_windowed.npz --epochs 30 --seeds 3
& 'D:\DevTools\envs\ARCGAD\python.exe' dynamic_experiments.py --target reddit_dyn/reddit_delta43200_windowed.npz --epochs 30 --seeds 3 --modes both
& 'D:\DevTools\envs\ARCGAD\python.exe' dynamic_experiments.py --target reddit_dyn/reddit_delta172800_windowed.npz --epochs 30 --seeds 3 --modes both
& 'D:\DevTools\envs\ARCGAD\python.exe' dynamic_experiments.py --target reddit_dyn/reddit_delta86400_windowed.npz --no-smooth --modes both --epochs 30 --seeds 3
& 'D:\DevTools\envs\ARCGAD\python.exe' run_experiments.py
```

The multidomain static ARC run trains on pubmed, Flickr, questions, and YelpChi;
the separate Reddit-union run collapses training-window edges/features and trains
ARC on that one static graph. The dynamic runner trains one 40-epoch ARC encoder
and temporal scorers, then evaluates five saved context samples. Detailed JSON
outputs and checkpoints are kept alongside. The prospective 24-hour results near
the top of this file are the current forecast results; the legacy table below
records the earlier same-window task only.

The 12h/24h/48h comparisons, temporal component ablations, context sizes,
window-1/previous-window drift, contamination, and smoothness-sort ablation were
run. These are exploratory, in-domain results: the temporal scorer trains on
Reddit training windows because this repository contains no collection of other
temporal training graphs. They therefore do not establish generalist transfer.
Dynamic results use one training seed and five sampled contexts; their uncertainty
does not include model-training variation. Cross-dataset Wikipedia-to-Reddit
transfer is unavailable because no Wikipedia temporal dataset is present.

## Legacy same-window experiment results

| Experiment | AUROC | AUPRC | Note |
|---|---:|---:|---|
| static_arc (zero-shot, ARC 4 domains) | 0.4698 ± 0.0308 | 0.0031 ± 0.0004 | per-snapshot ARC; seeds x contexts |
| static_arc (trained in-domain on union graph) | 0.4702 ± 0.0283 | 0.0031 ± 0.0003 | supervised reference |
| [reddit_delta172800_windowed_scaled_sorted] dynamic_both | 0.4869 ± 0.0034 | 0.005 ± 0.0 | legacy result; run count not recorded; 'none' = same training, no temporal module |
| [reddit_delta172800_windowed_scaled_sorted] context_size=2 | 0.4873 ± 0.0074 | 0.005 ± 0.0001 | mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] context_size=5 | 0.4921 ± 0.0028 | 0.005 ± 0.0 | mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] context_size=10 | 0.4869 ± 0.0015 | 0.005 ± 0.0 | mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] context_size=20 | 0.4862 ± 0.0013 | 0.005 ± 0.0 | mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] context_size=50 | 0.4853 ± 0.0024 | 0.005 ± 0.0 | mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] context_lag=window_1 | 0.4869 ± 0.0037 | 0.005 ± 0.0 | stale context; mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] context_lag=previous_window | 0.4874 ± 0.0029 | 0.005 ± 0.0 | stale context; mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] contaminated=0/10 | 0.4897 ± 0.004 | 0.005 ± 0.0 | fixed query set; mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] contaminated=1/10 | 0.4878 ± 0.0027 | 0.0049 ± 0.0 | fixed query set; mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] contaminated=2/10 | 0.4897 ± 0.0052 | 0.0049 ± 0.0001 | fixed query set; mode=not recorded |
| [reddit_delta172800_windowed_scaled_sorted] contaminated=5/10 | 0.4848 ± 0.0047 | 0.0046 ± 0.0001 | fixed query set; mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] dynamic_both | 0.495 ± 0.0024 | 0.0027 ± 0.0 | legacy result; run count not recorded; 'none' = same training, no temporal module |
| [reddit_delta43200_windowed_scaled_sorted] context_size=2 | 0.5047 ± 0.011 | 0.0027 ± 0.0001 | mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] context_size=5 | 0.4937 ± 0.0085 | 0.0027 ± 0.0 | mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] context_size=10 | 0.4954 ± 0.0016 | 0.0027 ± 0.0 | mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] context_size=20 | 0.4957 ± 0.0029 | 0.0027 ± 0.0 | mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] context_size=50 | 0.4926 ± 0.0017 | 0.0027 ± 0.0 | mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] context_lag=window_1 | 0.492 ± 0.0024 | 0.0027 ± 0.0 | stale context; mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] context_lag=previous_window | 0.4925 ± 0.0009 | 0.0026 ± 0.0 | stale context; mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] contaminated=0/10 | 0.494 ± 0.0037 | 0.0027 ± 0.0 | fixed query set; mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] contaminated=1/10 | 0.4919 ± 0.0039 | 0.0026 ± 0.0001 | fixed query set; mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] contaminated=2/10 | 0.488 ± 0.0059 | 0.0024 ± 0.0001 | fixed query set; mode=not recorded |
| [reddit_delta43200_windowed_scaled_sorted] contaminated=5/10 | 0.503 ± 0.0054 | 0.0023 ± 0.0001 | fixed query set; mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] dynamic_both | 0.4688 ± 0.0031 | 0.0031 ± 0.0 | legacy result; run count not recorded; 'none' = same training, no temporal module |
| [reddit_delta86400_windowed_scaled_unsorted] context_size=2 | 0.4852 ± 0.0138 | 0.0032 ± 0.0001 | mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] context_size=5 | 0.4686 ± 0.0043 | 0.0031 ± 0.0 | mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] context_size=10 | 0.4659 ± 0.0017 | 0.003 ± 0.0 | mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] context_size=20 | 0.4672 ± 0.002 | 0.0031 ± 0.0 | mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] context_size=50 | 0.4665 ± 0.002 | 0.0031 ± 0.0 | mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] context_lag=window_1 | 0.4678 ± 0.0022 | 0.0031 ± 0.0 | stale context; mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] context_lag=previous_window | 0.4699 ± 0.0033 | 0.0031 ± 0.0 | stale context; mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] contaminated=0/10 | 0.4667 ± 0.002 | 0.003 ± 0.0 | fixed query set; mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] contaminated=1/10 | 0.467 ± 0.0056 | 0.003 ± 0.0 | fixed query set; mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] contaminated=2/10 | 0.4668 ± 0.0038 | 0.003 ± 0.0 | fixed query set; mode=not recorded |
| [reddit_delta86400_windowed_scaled_unsorted] contaminated=5/10 | 0.4724 ± 0.0065 | 0.0028 ± 0.0001 | fixed query set; mode=not recorded |
| [reddit_delta86400_windowed_sorted] dynamic_none | 0.4827 ± 0.0028 | 0.0032 ± 0.0 | 3 seeds x 5 contexts; 'none' = same training, no temporal module |
| [reddit_delta86400_windowed_sorted] dynamic_residual | 0.48 ± 0.0034 | 0.0032 ± 0.0 | 3 seeds x 5 contexts; 'none' = same training, no temporal module |
| [reddit_delta86400_windowed_sorted] dynamic_gru | 0.5282 ± 0.009 | 0.004 ± 0.0002 | 3 seeds x 5 contexts; 'none' = same training, no temporal module |
| [reddit_delta86400_windowed_sorted] dynamic_attention | 0.5011 ± 0.015 | 0.0037 ± 0.0004 | 3 seeds x 5 contexts; 'none' = same training, no temporal module |
| [reddit_delta86400_windowed_sorted] dynamic_both | 0.5158 ± 0.0068 | 0.0035 ± 0.0001 | 3 seeds x 5 contexts; 'none' = same training, no temporal module |
| [reddit_delta86400_windowed_sorted] context_size=2 | 0.526 ± 0.0105 | 0.0037 ± 0.0002 | mode=both |
| [reddit_delta86400_windowed_sorted] context_size=5 | 0.5126 ± 0.0082 | 0.0035 ± 0.0001 | mode=both |
| [reddit_delta86400_windowed_sorted] context_size=10 | 0.515 ± 0.0067 | 0.0035 ± 0.0001 | mode=both |
| [reddit_delta86400_windowed_sorted] context_size=20 | 0.5147 ± 0.0048 | 0.0035 ± 0.0001 | mode=both |
| [reddit_delta86400_windowed_sorted] context_size=50 | 0.5184 ± 0.0044 | 0.0036 ± 0.0001 | mode=both |
| [reddit_delta86400_windowed_sorted] context_lag=0 | 0.5158 ± 0.0068 | 0.0035 ± 0.0001 | stale context; mode=both |
| [reddit_delta86400_windowed_sorted] context_lag=1 | 0.5161 ± 0.0043 | 0.0035 ± 0.0 | stale context; mode=both |
| [reddit_delta86400_windowed_sorted] context_lag=2 | 0.5179 ± 0.0076 | 0.0035 ± 0.0001 | stale context; mode=both |
| [reddit_delta86400_windowed_sorted] context_lag=5 | 0.5163 ± 0.0061 | 0.0035 ± 0.0001 | stale context; mode=both |
| [reddit_delta86400_windowed_sorted] context_lag=10 | 0.5124 ± 0.007 | 0.0035 ± 0.0001 | stale context; mode=both |
| [reddit_delta86400_windowed_sorted] context_lag=fixed_window_1 | 0.5104 ± 0.021 | 0.0035 ± 0.0002 | stale context; mode=both |
| [reddit_delta86400_windowed_sorted] contaminated=0/10 | 0.5141 ± 0.0098 | 0.0031 ± 0.0001 | fixed query set; mode=both |
| [reddit_delta86400_windowed_sorted] contaminated=1/10 | 0.5135 ± 0.0086 | 0.0031 ± 0.0001 | fixed query set; mode=both |
| [reddit_delta86400_windowed_sorted] contaminated=2/10 | 0.5121 ± 0.0088 | 0.0031 ± 0.0001 | fixed query set; mode=both |
| [reddit_delta86400_windowed_sorted] contaminated=5/10 | 0.5146 ± 0.0104 | 0.0031 ± 0.0002 | fixed query set; mode=both |
