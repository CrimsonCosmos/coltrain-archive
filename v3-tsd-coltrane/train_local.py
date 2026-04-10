#!/usr/bin/env python3
"""
Local two-phase training: pretrain on Lakh, fine-tune on jazz.
Designed for Mac M4 with MPS. Saves checkpoints every epoch.

Usage:
    # Phase 1: Pretrain (run once, ~4 days on M4)
    python train_local.py pretrain

    # Phase 2: Fine-tune on jazz (~1 day on M4)
    python train_local.py finetune
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
from data.tokenizer import MIDITokenizer
from data.dataset import create_dataloaders, create_pretrain_dataloader


# ============================================================
# Configuration
# ============================================================

MODEL = {
    'd_model': 384,
    'num_heads': 8,
    'num_layers': 4,
    'd_ff': 1536,
    'max_seq_len': 1024,
    'dropout': 0.3,
}

PRETRAIN_CONFIG = {
    **MODEL,
    'pretokenized_path': str(Path(__file__).parent / 'lakh_pretokenized_500k.npy'),
    'batch_size': 8,
    'num_epochs': 10,
    'learning_rate': 3e-4,
    'weight_decay': 0.05,
    'warmup_steps': 2000,
    'max_grad_norm': 1.0,
    'checkpoint_dir': str(Path(__file__).parent / 'checkpoints_pretrain'),
    'prefix': 'pretrain',
}

FINETUNE_CONFIG = {
    **MODEL,
    'midi_dir': str(Path(__file__).parent / '..' / 'augmented_dataset_v2'),
    'batch_size': 8,
    'num_epochs': 15,
    'learning_rate': 5e-5,
    'weight_decay': 0.05,
    'warmup_steps': 500,
    'max_grad_norm': 1.0,
    'max_steps_per_epoch': 5000,
    'checkpoint_dir': str(Path(__file__).parent / 'checkpoints_finetune'),
    'prefix': 'finetune',
    'pretrain_checkpoint': str(Path(__file__).parent / 'checkpoints_pretrain' / 'pretrain_best.pt'),
}


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
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
    }
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save(ckpt, checkpoint_dir / f'{prefix}_epoch{epoch}.pt')
    torch.save(ckpt, checkpoint_dir / f'{prefix}_latest.pt')

    return ckpt


def train(config, phase):
    device = get_device()
    print(f'\n{"="*70}')
    print(f'  Coltrain v3 — {phase.upper()} (device: {device})')
    print(f'{"="*70}')

    for k, v in config.items():
        print(f'  {k}: {v}')

    tokenizer = MIDITokenizer()

    # --- Data ---
    if phase == 'pretrain':
        npy_path = config['pretokenized_path']
        if not os.path.exists(npy_path):
            print(f'\nERROR: {npy_path} not found.')
            print('Run pretokenize_lakh.py first:')
            print(f'  python pretokenize_lakh.py --input_dir /path/to/lmd_full --output {npy_path}')
            return
        train_loader, val_loader = create_pretrain_dataloader(
            bin_path=npy_path,
            seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            num_workers=2,
        )
    else:
        midi_dir = config['midi_dir']
        if not os.path.exists(midi_dir):
            print(f'\nERROR: {midi_dir} not found.')
            return
        train_loader, val_loader = create_dataloaders(
            train_dir=midi_dir,
            val_dir=midi_dir,
            tokenizer=tokenizer,
            seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            num_workers=2,
            stride=512,
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

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            betas=(0.9, 0.98), weight_decay=config['weight_decay'])

    # --- Load weights ---
    start_epoch = 1
    best_val_loss = float('inf')
    current_step = 0
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    prefix = config['prefix']

    # For finetuning: load pretrained model weights (fresh optimizer)
    if phase == 'finetune':
        pt_path = config.get('pretrain_checkpoint', '')
        if os.path.exists(pt_path):
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f'Loaded pretrained weights from {pt_path}')
            print(f'  Pretrain val loss: {ckpt.get("val_loss", "N/A")}')
        else:
            print(f'WARNING: No pretrain checkpoint at {pt_path}')
            print('Training from scratch.')

    # Resume from same-phase checkpoint if it exists (crash recovery)
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
    max_steps_epoch = config.get('max_steps_per_epoch', len(train_loader))
    steps_per_epoch = min(len(train_loader), max_steps_epoch)
    total_steps = steps_per_epoch * config['num_epochs']
    warmup = config['warmup_steps']
    base_lr = config['learning_rate']

    print(f'\nDataloader steps per epoch: {len(train_loader)}')
    print(f'Effective steps per epoch: {steps_per_epoch}')
    print(f'Total steps: {total_steps}')
    print(f'Starting from epoch {start_epoch}...')
    print('=' * 70, flush=True)

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_batches = 0

        for batch_idx, (inp, tgt) in enumerate(train_loader):
            if batch_idx >= max_steps_epoch:
                break

            inp = inp.to(device)
            tgt = tgt.to(device)

            logits = model(inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()

            total_loss += loss.item()
            total_batches += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()
            lr = update_lr(optimizer, current_step, total_steps, warmup, base_lr)
            current_step += 1

            if (batch_idx + 1) % 200 == 0:
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

        # Save EVERY epoch (crash recovery — max 1 epoch of lost work)
        ckpt = save_checkpoint(model, optimizer, epoch, val_avg,
                               current_step, config, checkpoint_dir, prefix)
        print(f'  Saved checkpoint')

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt, checkpoint_dir / f'{prefix}_best.pt')
            print(f'  New best!')

    print(f'\n{phase.upper()} complete! Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints in: {checkpoint_dir}')


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('pretrain', 'finetune'):
        print('Usage:')
        print('  python train_local.py pretrain   # Phase 1: pretrain on Lakh')
        print('  python train_local.py finetune   # Phase 2: fine-tune on jazz')
        sys.exit(1)

    phase = sys.argv[1]

    if phase == 'pretrain':
        train(PRETRAIN_CONFIG, phase)
    else:
        train(FINETUNE_CONFIG, phase)


if __name__ == '__main__':
    main()
