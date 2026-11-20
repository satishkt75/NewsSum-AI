import torch
from newsum.models.losses import entity_mass_loss, coverage_loss, sequence_loss


def test_entity_mass_penalty_and_grad():
    logits = torch.randn(2, 3, 10, requires_grad=True)
    labels = torch.tensor([[1,2,-100],[2,3,4]])
    loss = entity_mass_loss(logits, labels, [2,3,5], [[2],[5]])
    assert loss.item() >= 0
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_coverage_nonnegative():
    attention = torch.tensor([[[[0.7,0.3],[0.6,0.4],[0.2,0.8]]]])
    labels = torch.tensor([[1,2,-100]])
    mask = torch.tensor([[1,1]])
    got = coverage_loss(attention, labels, mask)
    assert got.item() > 0


def test_sequence_ignores_padding():
    logits = torch.randn(2, 4, 11, requires_grad=True)
    labels = torch.tensor([[1,2,3,-100],[1,2,-100,-100]])
    result = sequence_loss(logits, labels)
    result.backward()
    assert torch.isfinite(result)
