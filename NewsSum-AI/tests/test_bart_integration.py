
import pytest
transformers = pytest.importorskip('transformers')
import torch
from transformers import BartConfig
from newsum.models.salience_faith_net import SalienceFaithNet


def test_tiny_bart_end_to_end_and_generation():
    cfg = BartConfig(vocab_size=100, d_model=32, encoder_layers=2, decoder_layers=2,
                     encoder_attention_heads=4, decoder_attention_heads=4,
                     encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=80,
                     pad_token_id=1, bos_token_id=0, eos_token_id=2,
                     decoder_start_token_id=2)
    model = SalienceFaithNet(cfg).configure_variant('salience_faith')
    model.entity_vocab_ids = [10,11,12]
    model.train()
    source = torch.tensor([[0,5,6,7,2,1],[0,6,9,2,1,1]])
    masks = (source != 1).long()
    segments = torch.tensor([[-1,0,0,1,1,-1],[-1,0,0,0,-1,-1]])
    targets = torch.tensor([[10,12,2,-100],[11,2,-100,-100]])
    model.config._attn_implementation = 'eager'
    outputs, losses = model(input_ids=source, attention_mask=masks,
                            segment_ids=segments, labels=targets,
                            supported_entity_ids=[[10],[11]],
                            compute_research_loss=True)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.plan_gate.weight.grad is not None
    assert model.planner.scorer.weight.grad is not None
    model.eval()
    with torch.inference_mode():
        sequences = model.generate(inputs=source[:1], attention_mask=masks[:1],
                                   segment_ids=segments[:1], num_beams=2,
                                   max_new_tokens=4)
    assert sequences.ndim == 2
    assert model._generation_plan is None
