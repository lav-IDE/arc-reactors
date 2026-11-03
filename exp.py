import numpy as np, torch
from arc_common import Archive
from dynamic_arc import DynamicARC
from dynamic_experiments import Bank

ds = Archive("reddit_dyn/reddit_delta86400_windowed.npz")
bank = np.load("reddit_dyn/bank_reddit_delta86400_windowed__red-reddit_delta86400_windowed_sorted.npy")
t = ds.split["test"][0]; dev = "cuda:0"
act = np.flatnonzero(ds.active[:ds.n_users, t])

x = bank[t][act].astype(np.float32)
print("bank: finite", np.isfinite(x).all(), "| std across nodes", x.std(0).mean(), "| frac clipped", (np.abs(x) >= 20).mean())

ckpt = "reddit_dyn/dyn_reddit_delta86400_windowed__both_s0_ep30_h5_s.pt"   # adjust
m = DynamicARC(128, "both").to(dev); m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval()
tr = Bank(bank, dev).traj(t, 5)
q = torch.as_tensor(act, device=dev); c = torch.as_tensor(ds.samples["0"], device=dev)
with torch.no_grad():
    s = m.score_with_context(tr[q], tr[c]).cpu().numpy(); z = m.encode(tr[q])
print("scores: min/max/std", s.min(), s.max(), s.std(), "| unique", len(np.unique(s.round(6))))
print("encoded std across nodes", z.std(0).mean().item())