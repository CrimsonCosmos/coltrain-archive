#!/usr/bin/env python3
"""
Coltrain v3 — Two-phase training for multi-model cascaded jazz generation.

Phase 1: Shared pretraining on all track-sequential data (standard CE loss)
Phase 2: Specialist fine-tuning with masked loss (one model per role)

Usage:
    # Phase 1: Shared pretraining
    python -u train_v3.py --phase shared 2>&1 | tee training_shared.log

    # Phase 2: Fine-tune specialists (run 4x, once per role)
    python -u train_v3.py --phase specialist --role lead 2>&1 | tee training_lead.log
    python -u train_v3.py --phase specialist --role comping 2>&1 | tee training_comping.log
    python -u train_v3.py --phase specialist --role bass 2>&1 | tee training_bass.log
    python -u train_v3.py --phase specialist --role drums 2>&1 | tee training_drums.log

Expects:
    - jazz_v3.npy + jazz_v3_groups.json (from pretokenize_v3.py)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'transformer'))

from model.transformer import MusicTransformer
from data.tokenizer_v3 import TrackSeqTokenizer, CHORDS_START, LEAD_START, COMPING_START, BASS_START, DRUMS_START
from data.dataset import create_pretrain_dataloader


# ============================================================
# Configuration
# ============================================================

MODEL = {
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 8,
    'd_ff': 2048,
    'max_seq_len': 2048,
    'dropout': 0.2,
}

SHARED_CONFIG = {
    **MODEL,
    'pretokenized_path': 'jazz_v3_2k.npy',
    'groups_path': 'jazz_v3_2k_groups.json',
    'batch_size': 36,
    'num_epochs': 5,
    'learning_rate': 3e-4,
    'weight_decay': 0.05,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0,
    'checkpoint_dir': 'checkpoints_v3',
    'prefix': 'v3_shared',
    'use_amp': True,
    'val_ratio': 0.05,
}

SPECIALIST_CONFIG = {
    **MODEL,
    'pretokenized_path': 'jazz_v3_2k.npy',
    'groups_path': 'jazz_v3_2k_groups.json',
    'batch_size': 36,
    'num_epochs': 5,
    'learning_rate': 5e-5,
    'weight_decay': 0.05,
    'warmup_steps': 500,
    'max_grad_norm': 1.0,
    'dropout': 0.15,
    'checkpoint_dir': 'checkpoints_v3',
    'shared_checkpoint': 'checkpoints_v3/v3_shared_best.pt',
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


def save_checkpoint(model, optimizer, epoch, val_loss, current_step,
                    config, checkpoint_dir, prefix, role=None):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'current_step': current_step,
        'config': config,
        'tokenizer_type': 'v3_trackseq',
        'role': role,
    }
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save(ckpt, checkpoint_dir / f'{prefix}_epoch{epoch}.pt')
    torch.save(ckpt, checkpoint_dir / f'{prefix}_latest.pt')
    return ckpt


# ============================================================
# On-the-fly role mask computation
# ============================================================

EOS_ID = 2  # From MidiTok TSD vocab (unchanged)

ROLE_DELIMS = {
    'lead': (LEAD_START, COMPING_START),       # 428, 429
    'comping': (COMPING_START, BASS_START),     # 429, 430
    'bass': (BASS_START, DRUMS_START),          # 430, 431
    'drums': (DRUMS_START, EOS_ID),             # 431, 2
}


def compute_batch_mask(inp, tgt, role, device):
    """Compute role mask for a batch — fully vectorized, no Python loops.

    Args:
        inp: (B, seq_len) input token IDs
        tgt: (B, seq_len) target token IDs (= shifted by 1)
        role: 'lead', 'comping', 'bass', or 'drums'
        device: torch device

    Returns:
        mask: (B, seq_len) bool tensor. True where target belongs to the role.
    """
    start_delim, end_delim = ROLE_DELIMS[role]
    B, S = tgt.shape

    # Reconstruct full sequence: positions 0..S
    full_seq = torch.cat([inp[:, :1], tgt], dim=1)  # (B, S+1)

    is_start = (full_seq == start_delim).int()  # (B, S+1)
    is_end = (full_seq == end_delim).int()      # (B, S+1)

    # cumsum tells us how many start/end delimiters we've passed
    cum_starts = torch.cumsum(is_start, dim=1)  # ≥1 from start_delim position onwards
    cum_ends = torch.cumsum(is_end, dim=1)      # ≥1 from end_delim position onwards

    # "After start" = shift cum_starts right by 1 (effect begins AFTER start_delim)
    cum_starts_shifted = torch.cat(
        [torch.zeros(B, 1, dtype=torch.int, device=device), cum_starts[:, :-1]], dim=1
    )
    # "After end" = shift cum_ends right by 1 (end_delim itself is still "in section")
    cum_ends_shifted = torch.cat(
        [torch.zeros(B, 1, dtype=torch.int, device=device), cum_ends[:, :-1]], dim=1
    )

    # In section: we've passed a start but not yet passed an end
    in_section = cum_starts_shifted > cum_ends_shifted

    # Include end_delim itself (it's where the model must predict "stop")
    mask_full = in_section | (is_end.bool() & (cum_starts_shifted > 0))

    # Target mask = positions 1..S of full_seq
    return mask_full[:, 1:]


# ============================================================
# Training functions
# ============================================================

def train_shared(config):
    """Phase 1: Shared pretraining with standard cross-entropy loss."""
    device = get_device()
    print(f'\n{"="*70}')
    print(f'  Coltrain v3 — Shared Pretraining (device: {device})')
    print(f'{"="*70}')
    for k, v in config.items():
        print(f'  {k}: {v}')

    tokenizer = TrackSeqTokenizer()
    print(f'\nVocab size: {tokenizer.vocab_size}')

    npy_path = config['pretokenized_path']
    if not os.path.exists(npy_path):
        print(f'\nERROR: {npy_path} not found.')
        return

    train_loader, val_loader = create_pretrain_dataloader(
        bin_path=npy_path,
        seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        num_workers=4,
        val_ratio=config.get('val_ratio', 0.05),
        groups_path=config.get('groups_path'),
    )

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

    # Resume
    start_epoch = 1
    best_val_loss = float('inf')
    current_step = 0
    checkpoint_dir = Path(config['checkpoint_dir'])
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

    _run_training_loop(model, criterion, optimizer, train_loader, val_loader,
                       config, device, start_epoch, best_val_loss, current_step,
                       checkpoint_dir, prefix, role=None)


def train_specialist(config, role):
    """Phase 2: Fine-tune a specialist model with masked loss."""
    device = get_device()
    print(f'\n{"="*70}')
    print(f'  Coltrain v3 — {role.upper()} Specialist (device: {device})')
    print(f'{"="*70}')
    for k, v in config.items():
        print(f'  {k}: {v}')

    tokenizer = TrackSeqTokenizer()
    prefix = f'v3_{role}'

    npy_path = config['pretokenized_path']
    if not os.path.exists(npy_path):
        print(f'\nERROR: {npy_path} not found.')
        return

    train_loader, val_loader = create_pretrain_dataloader(
        bin_path=npy_path,
        seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        num_workers=4,
        val_ratio=config.get('val_ratio', 0.05),
        groups_path=config.get('groups_path'),
    )

    # Load shared checkpoint
    shared_path = config['shared_checkpoint']
    if not os.path.exists(shared_path):
        print(f'\nERROR: Shared checkpoint not found: {shared_path}')
        return

    print(f'\nLoading shared checkpoint: {shared_path}')
    ckpt = torch.load(shared_path, map_location=device, weights_only=False)

    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id,
                                     label_smoothing=0.1, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            betas=(0.9, 0.98), weight_decay=config['weight_decay'])

    # Check for specialist resume
    checkpoint_dir = Path(config['checkpoint_dir'])
    start_epoch = 1
    best_val_loss = float('inf')
    current_step = 0

    resume_path = checkpoint_dir / f'{prefix}_latest.pt'
    if resume_path.exists():
        print(f'\nResuming specialist from {resume_path}...')
        sp_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(sp_ckpt['model_state_dict'])
        optimizer.load_state_dict(sp_ckpt['optimizer_state_dict'])
        start_epoch = sp_ckpt['epoch'] + 1
        best_val_loss = sp_ckpt['val_loss']
        current_step = sp_ckpt.get('current_step', 0)
        print(f'Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}')

    _run_training_loop(model, criterion, optimizer, train_loader, val_loader,
                       config, device, start_epoch, best_val_loss, current_step,
                       checkpoint_dir, prefix, role=role)


def _run_training_loop(model, criterion, optimizer, train_loader, val_loader,
                       config, device, start_epoch, best_val_loss, current_step,
                       checkpoint_dir, prefix, role=None):
    """Shared training loop for both phases."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['num_epochs']
    warmup = config['warmup_steps']
    base_lr = config['learning_rate']
    use_amp = config.get('use_amp', False) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    is_specialist = role is not None

    print(f'\nSteps per epoch: {steps_per_epoch}')
    print(f'Total steps: {total_steps}')
    print(f'AMP (float16): {use_amp}')
    print(f'Specialist mode: {role or "off"}')
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
                    loss = _compute_loss(logits, inp, tgt, criterion,
                                         is_specialist, role, device)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inp)
                loss = _compute_loss(logits, inp, tgt, criterion,
                                     is_specialist, role, device)
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
                eta = elapsed / (batch_idx + 1) * (steps_per_epoch - batch_idx - 1)
                print(f'  [{batch_idx+1}/{steps_per_epoch}] '
                      f'loss={avg:.4f} ppl={ppl:.1f} lr={lr:.2e} '
                      f'ETA={eta/60:.0f}min', flush=True)

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
                        loss = _compute_loss(logits, inp, tgt, criterion,
                                             is_specialist, role, device)
                else:
                    logits = model(inp)
                    loss = _compute_loss(logits, inp, tgt, criterion,
                                         is_specialist, role, device)
                val_loss += loss.item()
                val_batches += 1

        train_avg = total_loss / total_batches
        val_avg = val_loss / max(val_batches, 1)
        elapsed = (time.time() - epoch_start) / 60

        print(f'\nEpoch {epoch}/{config["num_epochs"]} ({elapsed:.1f} min):')
        print(f'  Train: {train_avg:.4f} (ppl {min(math.exp(train_avg), 99999):.1f})')
        print(f'  Val:   {val_avg:.4f} (ppl {min(math.exp(val_avg), 99999):.1f})', flush=True)

        ckpt = save_checkpoint(model, optimizer, epoch, val_avg, current_step,
                               config, checkpoint_dir, prefix, role)
        print(f'  Saved checkpoint')

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt, checkpoint_dir / f'{prefix}_best.pt')
            print(f'  New best!')

    print(f'\nTraining complete! Best val loss: {best_val_loss:.4f}')


def _compute_loss(logits, inp, tgt, criterion, is_specialist, role, device):
    """Compute loss — standard CE for shared, masked CE for specialist."""
    if not is_specialist:
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        if loss.dim() > 0:
            loss = loss.mean()
        return loss

    # Specialist: compute mask on-the-fly from batch data
    mask = compute_batch_mask(inp, tgt, role, device)  # (B, seq_len)

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_tgt = tgt.reshape(-1)
    flat_mask = mask.reshape(-1)

    per_token_loss = criterion(flat_logits, flat_tgt)
    masked_loss = per_token_loss * flat_mask.float()
    n_active = flat_mask.sum().clamp(min=1)

    return masked_loss.sum() / n_active


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Coltrain v3 Training')
    parser.add_argument('--phase', required=True, choices=['shared', 'specialist'])
    parser.add_argument('--role', choices=['lead', 'comping', 'bass', 'drums'],
                        help='Role for specialist training')
    args = parser.parse_args()

    if args.phase == 'shared':
        train_shared(SHARED_CONFIG)
    elif args.phase == 'specialist':
        if not args.role:
            parser.error('--role required for specialist training')
        train_specialist(SPECIALIST_CONFIG, args.role)


if __name__ == '__main__':
    main()
