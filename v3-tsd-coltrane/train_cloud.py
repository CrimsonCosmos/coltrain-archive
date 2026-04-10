#!/usr/bin/env python3
"""
Cloud training script for Vast.ai A100.
Jazz-only fine-tuning with MidiTok TSD tokenizer.

Usage:
    python -u train_cloud.py 2>&1 | tee training.log

Expects:
    - jazz_miditok.npy in current directory (pretokenized jazz dataset)
    - jazz_miditok_groups.json in current directory (song group mapping)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'transformer'))

from model.transformer import MusicTransformer
from data.tokenizer_miditok import MidiTokWrapper
from data.dataset import create_pretrain_dataloader


# ============================================================
# Configuration
# ============================================================

MODEL = {
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 8,
    'd_ff': 2048,
    'max_seq_len': 1024,
    'dropout': 0.2,
}

TRAIN_CONFIG = {
    **MODEL,
    'pretokenized_path': 'jazz_miditok.npy',
    'groups_path': 'jazz_miditok_groups.json',
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 3e-4,
    'weight_decay': 0.05,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0,
    'checkpoint_dir': 'checkpoints',
    'prefix': 'miditok_jazz',
    'tokenizer_type': 'miditok_tsd',
    'use_amp': True,
    'val_ratio': 0.05,
}


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def update_lr(optimizer, step, total_steps, warmup, base_lr):
    if step < warmup:
        lr = base_lr * (step / max(warmup, 1))
    else:
        progress = (step - warmup) / max(total_steps - warmup, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(min(progress, 1.0) * math.pi))
        lr = max(lr, base_lr * 0.01)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def save_checkpoint(model, optimizer, epoch, val_loss, current_step, config, checkpoint_dir, prefix):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'current_step': current_step,
        'config': config,
        'tokenizer_type': 'miditok_tsd',
    }
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save(ckpt, checkpoint_dir / f'{prefix}_epoch{epoch}.pt')
    torch.save(ckpt, checkpoint_dir / f'{prefix}_latest.pt')

    return ckpt


def train(config):
    device = get_device()
    print(f'\n{"="*70}')
    print(f'  Coltrain — MidiTok TSD Cloud Training (device: {device})')
    print(f'{"="*70}')

    for k, v in config.items():
        print(f'  {k}: {v}')

    # --- Tokenizer ---
    tokenizer = MidiTokWrapper()
    print(f'\nVocab size: {tokenizer.vocab_size}')

    # --- Data ---
    npy_path = config['pretokenized_path']
    groups_path = config.get('groups_path')

    if not os.path.exists(npy_path):
        print(f'\nERROR: {npy_path} not found.')
        return

    train_loader, val_loader = create_pretrain_dataloader(
        bin_path=npy_path,
        seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        num_workers=4,
        val_ratio=config.get('val_ratio', 0.05),
        groups_path=groups_path,
    )

    # --- Model ---
    print('\nCreating model...')
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            betas=(0.9, 0.98), weight_decay=config['weight_decay'])

    # --- Resume from checkpoint if exists ---
    start_epoch = 1
    best_val_loss = float('inf')
    current_step = 0
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    prefix = config['prefix']

    resume_path = checkpoint_dir / f'{prefix}_latest.pt'
    if resume_path.exists():
        print(f'\nResuming from {resume_path}...')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        current_step = ckpt.get('current_step', 0)
        print(f'Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}')

    # --- Training loop ---
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['num_epochs']
    warmup = config['warmup_steps']
    base_lr = config['learning_rate']
    use_amp = config.get('use_amp', False) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f'\nSteps per epoch: {steps_per_epoch}')
    print(f'Total steps: {total_steps}')
    print(f'AMP (float16): {use_amp}')
    print(f'Starting from epoch {start_epoch}...')
    print('=' * 70, flush=True)

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_batches = 0

        for batch_idx, (inp, tgt) in enumerate(train_loader):
            inp = inp.to(device)
            tgt = tgt.to(device)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(inp)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inp)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            optimizer.zero_grad()
            lr = update_lr(optimizer, current_step, total_steps, warmup, base_lr)
            current_step += 1

            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / total_batches
                ppl = min(math.exp(avg), 99999)
                elapsed = time.time() - epoch_start
                eta_epoch = elapsed / (batch_idx + 1) * (steps_per_epoch - batch_idx - 1)
                print(f'  [{batch_idx+1}/{steps_per_epoch}] '
                      f'loss={avg:.4f} ppl={ppl:.1f} lr={lr:.2e} '
                      f'ETA={eta_epoch/60:.0f}min', flush=True)

        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits = model(inp)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                else:
                    logits = model(inp)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                val_loss += loss.item()
                val_batches += 1

        train_avg = total_loss / total_batches
        val_avg = val_loss / max(val_batches, 1)
        elapsed = (time.time() - epoch_start) / 60

        print(f'\nEpoch {epoch}/{config["num_epochs"]} ({elapsed:.1f} min):')
        print(f'  Train: {train_avg:.4f} (ppl {min(math.exp(train_avg), 99999):.1f})')
        print(f'  Val:   {val_avg:.4f} (ppl {min(math.exp(val_avg), 99999):.1f})', flush=True)

        # Save checkpoint every epoch
        ckpt = save_checkpoint(model, optimizer, epoch, val_avg,
                               current_step, config, checkpoint_dir, prefix)
        print(f'  Saved checkpoint')

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt, checkpoint_dir / f'{prefix}_best.pt')
            print(f'  New best!')

    print(f'\nTraining complete! Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints in: {checkpoint_dir}')


if __name__ == '__main__':
    train(TRAIN_CONFIG)
