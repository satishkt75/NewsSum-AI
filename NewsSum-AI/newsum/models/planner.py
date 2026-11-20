
import torch
from torch import nn


class SaliencePlanner(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor, attention_mask=None):
        """Returns (plan[B,D], score[B,S], valid[B,S]). Padding segments = -1."""
        b, n, d = hidden.shape
        if attention_mask is None:
            attention_mask = torch.ones((b, n), device=hidden.device, dtype=torch.long)
        valid_tokens = (segment_ids >= 0) & attention_mask.bool()
        indices = segment_ids.clamp(min=0).long()
        count = int(indices.max().item()) + 1 if valid_tokens.any() else 1
        summed = hidden.new_zeros(b, count, d)
        summed.scatter_add_(1, indices.unsqueeze(-1).expand(-1, -1, d),
                            hidden * valid_tokens.unsqueeze(-1))
        totals = hidden.new_zeros(b, count)
        totals.scatter_add_(1, indices, valid_tokens.to(hidden.dtype))
        pooled = summed / totals.clamp_min(1).unsqueeze(-1)
        valid = totals > 0
        raw = torch.sigmoid(self.scorer(pooled).squeeze(-1)).masked_fill(~valid, 0)
        scores = raw / raw.sum(-1, keepdim=True).clamp_min(1e-8)
        plan = (scores.unsqueeze(-1) * pooled).sum(1)
        return plan, scores, valid


def fallback_segments(attention_mask: torch.Tensor, max_tokens: int = 64):
    """Contiguous token blocks are used only when sentence boundaries aren't provided."""
    b, t = attention_mask.shape
    ids = torch.arange(t, device=attention_mask.device)[None, :].expand(b, -1) // max_tokens
    return ids.masked_fill(~attention_mask.bool(), -1)
