"""Transparent research training loop with per-loss logs and resumable checkpoints."""
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from newsum.data.pipeline import NewsCollator


def set_seed(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def train(model, tokenizer, training_set, validation_set, output_dir,
          epochs=10, batch_size=1, accumulation=4, learning_rate=3e-5,
          warmup_steps=500, patience=2, seed=42, device=None,
          max_train_steps=None, validation_samples=128, resume=None):
    from transformers import get_linear_schedule_with_warmup
    from newsum.evaluation.metrics import rouge_scores
    set_seed(seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    model.train()
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    collator = NewsCollator(tokenizer)
    loader = DataLoader(training_set, batch_size=batch_size, shuffle=True,
                        collate_fn=collator, num_workers=0)
    valid = DataLoader(validation_set, batch_size=batch_size, shuffle=False,
                       collate_fn=collator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max_train_steps or epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp and not torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    global_step = 0
    start_epoch = 0
    best = -float('inf')
    stale = 0
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        global_step = int(state['global_step'])
        start_epoch = int(state['epoch'])
        best = float(state['best'])
    log_path = folder / 'train_log.jsonl'
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = {'seq': 0., 'entity': 0., 'coverage': 0., 'total': 0.}
        batches = 0
        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                _, losses = model(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'],
                                  segment_ids=batch['segment_ids'],
                                  labels=batch['labels'],
                                  supported_entity_ids=batch['supported_entity_ids'],
                                  compute_research_loss=True)
                # A partial final accumulation group receives its true divisor.
                divisor = (accumulation if i + accumulation <= len(loader) else
                           min(accumulation, len(loader) - ((i // accumulation)*accumulation)))
                loss = losses.total / divisor
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            for name in sums:
                sums[name] += float(getattr(losses, name).detach())
            batches += 1
            if (i+1) % accumulation == 0 or i+1 == len(loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if max_train_steps and global_step >= max_train_steps:
                    break
        scores = validate(model, tokenizer, valid, device, max_samples=validation_samples)
        record = {'epoch': epoch+1, 'global_step': global_step,
                  **{f'loss_{k}': v / max(1, batches) for k, v in sums.items()},
                  **scores}
        with log_path.open('a', encoding='utf8') as fp:
            fp.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        # ROUGE is used for fast epoch selection; AlignScore is evaluated separately.
        criterion = scores['rougeLsum']
        if criterion > best:
            best = criterion
            stale = 0
            model.save_research_checkpoint(folder / 'best', tokenizer)
        else:
            stale += 1
        model.save_research_checkpoint(folder / 'last', tokenizer)
        state = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'global_step': global_step, 'epoch': epoch+1, 'best': best}
        torch.save(state, folder / 'trainer_state.pt')
        if stale >= patience or (max_train_steps and global_step >= max_train_steps):
            break
    return folder / 'best'


@torch.no_grad()
def validate(model, tokenizer, loader, device, max_samples=128):
    from newsum.evaluation.metrics import rouge_scores
    model.eval()
    predictions, references = [], []
    for batch in loader:
        batch = to_device(batch, device)
        output = model.generate(inputs=batch['input_ids'], attention_mask=batch['attention_mask'],
                                segment_ids=batch['segment_ids'], num_beams=2,
                                max_new_tokens=64)
        predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        references.extend(tokenizer.batch_decode(batch['labels'].clamp_min(0),
                                                  skip_special_tokens=True))
        if len(predictions) >= max_samples:
            break
    return rouge_scores(predictions[:max_samples], references[:max_samples])
