"""BART-large with decoder-layer salience gates and optional entity/coverage losses.

Supports Hugging Face generation with cached planning signals and beam expansion.
All model variants share one implementation for fair ablations.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from .planner import SaliencePlanner, fallback_segments
from .losses import sequence_loss, entity_mass_loss, coverage_loss


@dataclass
class ResearchLosses:
    total: torch.Tensor
    seq: torch.Tensor
    entity: torch.Tensor
    coverage: torch.Tensor


class SalienceFaithNet(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.planner = SaliencePlanner(config.d_model)
        # Shared across all 12 layers. Approx. 2.16M trainable parameters.
        self.plan_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.plan_projection_in = nn.Linear(config.d_model, 32, bias=False)
        self.plan_projection_out = nn.Linear(32, config.d_model, bias=False)
        self._active_plan = None
        self._generation_plan = None
        self.use_salience = True
        self.use_entity = True
        self.use_coverage = True
        self.entity_weight = 0.5
        self.coverage_weight = 0.2
        self.entity_vocab_ids = []
        self._gate_handles = [layer.register_forward_hook(self._gate_layer)
                              for layer in self.model.decoder.layers]

    def configure_variant(self, variant='salience_faith'):
        options = {
            'baseline': (False, False, False),
            'salience': (True, False, True),
            'faith': (False, True, False),
            'salience_faith': (True, True, True),
        }
        if variant not in options:
            raise ValueError(f'Unknown variant {variant}; choose {list(options)}')
        self.use_salience, self.use_entity, self.use_coverage = options[variant]
        self.variant = variant
        return self

    def _gate_layer(self, module, args, output):
        if not self.use_salience or self._active_plan is None:
            return output
        states = output[0]
        plan = self._active_plan
        if states.shape[0] != plan.shape[0]:
            if states.shape[0] % plan.shape[0] != 0:
                raise RuntimeError('Beam expansion and planning batch sizes do not match')
            plan = plan.repeat_interleave(states.shape[0] // plan.shape[0], dim=0)
        plan = plan.to(dtype=states.dtype, device=states.device)
        expanded = plan.unsqueeze(1).expand(-1, states.shape[1], -1)
        gate = torch.sigmoid(self.plan_gate(torch.cat([states, expanded], dim=-1)))
        projected = self.plan_projection_out(self.plan_projection_in(expanded))
        altered = states + gate * projected
        return (altered,) + tuple(output[1:])

    def _encode_plan(self, input_ids, attention_mask, segment_ids, encoder_outputs):
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                                 return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        if segment_ids is None:
            segment_ids = fallback_segments(attention_mask)
        plan, _, _ = self.planner(hidden, segment_ids, attention_mask)
        return encoder_outputs, plan

    def forward(self, input_ids=None, attention_mask=None, segment_ids=None,
                supported_entity_ids=None, labels=None, encoder_outputs=None,
                compute_research_loss=False, **kwargs):
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        plan = self._generation_plan
        if self.use_salience and plan is None:
            encoder_outputs, plan = self._encode_plan(input_ids, attention_mask, segment_ids,
                                                       encoder_outputs)
        previous = self._active_plan
        self._active_plan = plan
        try:
            if not compute_research_loss:
                return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                       encoder_outputs=encoder_outputs, labels=labels, **kwargs)
            if labels is None:
                raise ValueError('labels required for research training')
            output = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                     encoder_outputs=encoder_outputs, labels=labels,
                                     output_attentions=self.use_coverage,
                                     return_dict=True, **kwargs)
            seq = sequence_loss(output.logits, labels)
            zero = seq * 0
            ent = (entity_mass_loss(output.logits, labels, self.entity_vocab_ids,
                                    supported_entity_ids or [[] for _ in range(labels.shape[0])])
                   if self.use_entity else zero)
            cov = (coverage_loss(output.cross_attentions[-1], labels, attention_mask)
                   if self.use_coverage else zero)
            return output, ResearchLosses(seq + self.entity_weight * ent +
                                          self.coverage_weight * cov, seq, ent, cov)
        finally:
            self._active_plan = previous

    @torch.no_grad()
    def generate(self, inputs=None, segment_ids=None, **kwargs):
        """Precompute the plan exactly once; hooks expand it to all beam hypotheses."""
        if not self.use_salience:
            return super().generate(inputs=inputs, **kwargs)
        if inputs is None:
            inputs = kwargs.pop('input_ids', None)
        if inputs is None:
            raise ValueError('Pass tokenised input_ids to generate')
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is None:
            attention_mask = (inputs != self.config.pad_token_id).long()
            kwargs['attention_mask'] = attention_mask
        supplied_encoder_outputs = kwargs.get('encoder_outputs', None)
        enc, plan = self._encode_plan(inputs, attention_mask, segment_ids, supplied_encoder_outputs)
        kwargs['encoder_outputs'] = enc
        previous = self._generation_plan
        self._generation_plan = plan
        try:
            return super().generate(inputs=inputs, **kwargs)
        finally:
            self._generation_plan = previous

    def save_research_checkpoint(self, folder, tokenizer=None):
        import json
        from pathlib import Path
        folder = Path(folder)
        self.save_pretrained(folder)
        if tokenizer is not None:
            tokenizer.save_pretrained(folder)
        (folder / 'research_config.json').write_text(json.dumps({
            'variant': getattr(self, 'variant', 'salience_faith'),
            'entity_vocab_ids': self.entity_vocab_ids,
            'entity_weight': self.entity_weight,
            'coverage_weight': self.coverage_weight}, indent=2))

    @classmethod
    def load_research_checkpoint(cls, folder, **kwargs):
        import json
        from pathlib import Path
        model = cls.from_pretrained(folder, **kwargs)
        config_path = Path(folder) / 'research_config.json'
        if config_path.exists():
            data = json.loads(config_path.read_text())
            model.configure_variant(data.get('variant', 'salience_faith'))
            model.entity_vocab_ids = data.get('entity_vocab_ids', [])
            model.entity_weight = data.get('entity_weight', 0.5)
            model.coverage_weight = data.get('coverage_weight', 0.2)
        return model
