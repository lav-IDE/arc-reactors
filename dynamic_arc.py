"""Trajectory-level temporal components on top of ARC's residual encoder.

Input: per-node trajectories [nodes, history, dim] built from ARC window embeddings
(ARC alignment + residual encoder stay in ARC's own code).
  none      last-window embedding only                 (static ablation, same training as the rest)
  residual  [h_t ; h_t - mean(h_{t-K..t-1})]          (temporal residual: drift from own history)
  gru       GRU final state
  attention self-attention over the K windows, last token
  both      [GRU state ; temporal residual]
Score = L2 distance between a node's trajectory code and its cross-attention reconstruction from
the context nodes' trajectory codes (ARC's scorer, applied to trajectories)."""
import torch
import torch.nn.functional as F
from torch import nn

MODES = ("none", "residual", "gru", "attention", "both")


class DynamicARC(nn.Module):
    def __init__(self, hidden_dim, module="gru", heads=4):
        super().__init__()
        if module not in MODES: raise ValueError(f"module must be one of {MODES}")
        self.module = module
        if module in ("gru", "both"): self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        if module == "attention":
            self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True); self.norm = nn.LayerNorm(hidden_dim)
        out_dim = hidden_dim * (2 if module in ("residual", "both") else 1)
        self.query = nn.Linear(out_dim, out_dim); self.key = nn.Linear(out_dim, out_dim)

    def encode(self, h):
        if h.ndim != 3: raise ValueError("expected [nodes, history, hidden_dim]")
        last = h[:, -1]
        if self.module in ("gru", "both"):
            temporal = self.gru(h)[1][-1]
        elif self.module == "attention":
            seq, _ = self.attn(h, h, h, need_weights=False); temporal = self.norm(seq + h)[:, -1]
        else:
            temporal = last
        if self.module in ("residual", "both"):
            past = h[:, :-1].mean(dim=1) if h.shape[1] > 1 else torch.zeros_like(last)
            return torch.cat((temporal, last - past), dim=-1)
        return temporal

    def reconstruct(self, q, support):
        w = torch.softmax(self.query(q) @ self.key(support).T / (q.shape[-1] ** 0.5), dim=-1)
        return w @ support

    def score_with_context(self, query_traj, context_traj):
        q = self.encode(query_traj); return torch.linalg.vector_norm(q - self.reconstruct(q, self.encode(context_traj)), dim=-1)

    def training_loss(self, traj, labels, eligible, candidates, support_size=10):
        """ARC's balanced in-context loss for one window. Queries are drawn ONLY from `eligible`
        nodes (active users) so training matches what is scored at test time."""
        dev = traj.device
        pos = torch.nonzero((labels == 1) & eligible).flatten(); normal = torch.nonzero((labels == 0) & eligible).flatten()
        cand = candidates[(labels[candidates] == 0)]
        if len(pos) == 0 or len(cand) < support_size: return None
        sup = cand[torch.randperm(len(cand), device=dev)[:support_size]]
        pool = normal[~torch.isin(normal, sup)]; n = min(len(pos), len(pool))
        if n == 0: return None
        qn = pool[torch.randperm(len(pool), device=dev)[:n]]
        z = self.encode(traj); rec = self.reconstruct(torch.cat((z[pos[:n]], z[qn])), z[sup])
        lp = F.cosine_embedding_loss(z[pos[:n]], rec[:n], -torch.ones(n, device=dev))
        ln = F.cosine_embedding_loss(z[qn], rec[n:], torch.ones(n, device=dev))
        return lp + ln


def trajectory_batch(snapshot_embeddings, end_window, history=5):
    """Causal history [N, history, D]; early windows are left-padded with their first state."""
    start = max(0, end_window - history + 1)
    seq = torch.stack(snapshot_embeddings[start:end_window + 1], dim=1)
    if seq.shape[1] < history:
        seq = torch.cat([seq[:, :1].expand(-1, history - seq.shape[1], -1), seq], dim=1)
    return seq