import torch
from newsum.models.planner import SaliencePlanner, fallback_segments


def test_planner_probabilities_and_gradients():
    torch.manual_seed(1)
    planner = SaliencePlanner(8)
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    segments = torch.tensor([[0,0,1,1,-1,-1],[0,1,1,-1,-1,-1]])
    mask = (segments >= 0).long()
    plan, scores, valid = planner(hidden, segments, mask)
    assert plan.shape == (2,8)
    assert torch.allclose(scores.sum(1), torch.ones(2), atol=1e-6)
    assert valid.sum().item() == 4
    plan.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_fallback_segments_padding():
    mask = torch.tensor([[1,1,1,0]])
    assert fallback_segments(mask,2).tolist() == [[0,0,1,-1]]
