"""Research losses and differentiable coverage objective."""
import torch
import torch.nn.functional as F


def sequence_loss(logits, labels, smoothing=0.1):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, label_smoothing=smoothing)


def entity_mass_loss(logits, labels, entity_vocabulary_ids, supported_entity_ids):
    """Probability assigned to corpus entity subwords not supported by source entities.

    This is the token-level surrogate specified in the paper, not semantic entailment.
    Empty entity vocabulary means loss=0 (explicitly visible in logs).
    """
    if not entity_vocabulary_ids:
        return logits.sum() * 0.0
    vocab_size = logits.shape[-1]
    ids = sorted({int(x) for x in entity_vocabulary_ids if 0 <= int(x) < vocab_size})
    if not ids:
        return logits.sum() * 0.0
    ids_tensor = torch.tensor(ids, device=logits.device, dtype=torch.long)
    selected = logits.index_select(-1, ids_tensor)
    denominator = torch.logsumexp(logits, dim=-1, keepdim=True)
    mass = (selected - denominator).exp()
    supported = []
    for row in supported_entity_ids:
        present = set(int(x) for x in row)
        supported.append([x not in present for x in ids])
    unsupported = torch.tensor(supported, dtype=mass.dtype, device=mass.device)
    loss_per_token = (mass * unsupported[:, None, :]).sum(-1)
    valid = (labels != -100).to(loss_per_token.dtype)
    return (loss_per_token * valid).sum() / valid.sum().clamp_min(1)


def coverage_loss(cross_attention, labels, source_mask):
    """Last-layer, head-averaged cross-attention: sum_t,i min(a_ti, coverage_ti)."""
    if cross_attention is None:
        raise ValueError('Coverage requires output_attentions=True and eager attention.')
    weights = cross_attention.float().mean(dim=1)
    weights = weights * source_mask[:, None, :].to(weights.dtype)
    history = weights.cumsum(dim=1) - weights
    overlap = torch.minimum(weights, history).sum(dim=-1)
    valid = (labels != -100).to(overlap.dtype)
    return (overlap * valid).sum() / valid.sum().clamp_min(1)
