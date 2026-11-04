# ARC
 This repository an extended fork of the official implementation of "[ARC: A Generalist Graph Anomaly Detector with In-Context Learning](https://arxiv.org/pdf/2405.16771)", accepted by NeurIPS 2024.

 ![pipeline_00](https://github.com/user-attachments/assets/bcb97083-c250-4557-945b-850d306020f6)


# Setup
```js/java/c#/text
conda create -n ARCGAD python=3.8
conda activate ARCGAD
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install scipy
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.3.1

```

# Usage
Due to file size limitations, the tFinance dataset can be downloaded via [Google Drive](https://drive.google.com/file/d/1mvqGlqS1hvAGj9TkA0vIHjNDX3291Ugx/view?usp=sharing). 
Just run the script corresponding to the dataset and method you want. For instance:

```js/java/c#/text
python main.py --trial 5 --shot 10
```

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```js/java/c#/text
@inproceedings{liu2024arc,
  title={ARC: A Generalist Graph Anomaly Detector with In-Context Learning},
  author={Liu, Yixin and Li, Shiyuan and Zheng, Yu and Chen, Qingfeng and Zhang, Chengqi and Pan, Shirui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

# Dynamic-ARC on Reddit

This extension evaluates ARC on a sequence of Reddit graph snapshots and tests
whether temporal modeling improves window-level anomaly detection.

## Task and evaluation

Each snapshot is one 24-hour window. A user is labeled anomalous when one of
their posts in that window has `state_label=1`. Original ARC scores each completed
snapshot from its graph and features. Dynamic-ARC summarizes the five most recent
ARC embeddings (with a GRU, residual, or attention module) and scores the final
window. This is same-window anomaly detection, not future-ban forecasting; the
features may include the post associated with the label.

Snapshots are split chronologically 60/15/25 into train, validation, and test.
Context candidates have at least three active training windows and no anomaly
labels in training. Selection uses no validation or test labels, so a context
user can become anomalous later. Test metrics score active users except the
sampled context nodes.

## Current 24-hour results

The test split contains 174 positive user-windows among 53,119 active
user-windows (prevalence 0.00328).

| Method | AUROC | AUPRC |
|---|---:|---:|
| Recent activity | 0.5357 | 0.00434 |
| Activity increase | 0.4556 | 0.00385 |
| New-neighbor count | 0.5836 | 0.00489 |
| Neighbor churn | 0.5970 | 0.00413 |
| Original ARC, per snapshot | 0.5014 ± 0.0042 | 0.00342 ± 0.00004 |
| Dynamic-ARC GRU | 0.5368 ± 0.0190 | 0.00411 ± 0.00024 |

Original ARC uses its pretrained seed-0 encoder and original cross-attentive
scorer. Dynamic-ARC uses the same frozen encoder, a GRU over five snapshots, and
a trajectory scorer trained on Reddit training windows. Original ARC is near
chance, while GRU is modestly above chance. Simple graph-change baselines score
better in AUROC; all AUPRC values are close to the prevalence baseline. These
results indicate weak performance on this Reddit setup and do not establish how
ARC performs on all dynamic graphs.

### Random-window split sensitivity check

A seeded random split (`seed=42`) assigns complete snapshots to 60% train, 15%
validation, and 25% test. Its test subset has 75 positive user-windows among
51,957 active user-windows (prevalence 0.00144).

| Method | AUROC | AUPRC |
|---|---:|---:|
| Recent activity | 0.5748 | 0.00218 |
| Activity increase | 0.4255 | 0.00204 |
| New-neighbor count | 0.5524 | 0.00185 |
| Neighbor churn | 0.5559 | 0.00159 |
| Original ARC, per snapshot | 0.5104 ± 0.0120 | 0.00150 ± 0.00003 |
| Dynamic-ARC GRU | 0.6406 ± 0.0241 | 0.00246 ± 0.00026 |

This is a diagnostic comparison, not the primary result: random assignment puts
snapshots from later in the timeline into training and can separate adjacent
windows across splits. It therefore does not measure performance on a strictly
future period. The test prevalence also differs from the chronological split.

```powershell
python build_snapshots.py --delta 86400 --split-strategy random --split-seed 42
python baseline_static_arc.py reddit_dyn/reddit_delta86400_windowed_random_split_seed42.npz --training-setup arc-multidomain --epochs 40 --trials 1
python dynamic_experiments.py --target reddit_dyn/reddit_delta86400_windowed_random_split_seed42.npz --epochs 30 --seeds 3 --modes gru --robust-mode gru
python snapshot_baselines.py reddit_dyn/reddit_delta86400_windowed_random_split_seed42.npz
```

Results are saved under `reddit_dyn/` with `random_split_seed42` in the filename.

## Reproduce

Run these commands from the repository root in the ARCGAD Python 3.8.20
environment. Snapshot building also requires pandas, NumPy, and SciPy.

```powershell
python build_snapshots.py --delta 86400
python baseline_static_arc.py reddit_dyn/reddit_delta86400_windowed.npz --training-setup arc-multidomain --epochs 40 --trials 1
python dynamic_experiments.py --target reddit_dyn/reddit_delta86400_windowed.npz --epochs 30 --seeds 3 --modes gru --robust-mode gru
python snapshot_baselines.py reddit_dyn/reddit_delta86400_windowed.npz
```

The source has 672,447 events, 10,000 users, 984 items, and 172 post-feature
values. Snapshot archives, per-window baseline metrics, ARC results, and Dynamic-
ARC results are saved under `reddit_dyn/`. The earlier `*_fh24*` files record a
separate forecasting experiment and are not part of the current evaluation.

Implementation details and the broader historical results are retained in the
project files: `build_snapshots.py`, `baseline_static_arc.py`,
`dynamic_experiments.py`, `dynamic_arc.py`, and `snapshot_baselines.py`.
