#!/usr/bin/env python3
"""
Training Script for Music Transformer v2
All improvements included:
- Weighted loss (NOTE_ON matters more than TIME_SHIFT)
- Label smoothing (prevents overconfidence / mode collapse)
- Gradient accumulation (bigger effective batch without more memory)
- Cosine annealing with warm restarts
- Proper checkpointing with config
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.transformer import MusicTransformer
from data.tokenizer import MIDITokenizer
from data.dataset import MIDIDataset, create_dataloaders


def build_token_weights(tokenizer, note_weight=3.0, time_weight=1.0, device='cpu'):
    """
    Build per-token loss weights.

    NOTE_ON tokens get higher weight so the model prioritizes getting
    harmony and melody RIGHT over exact timing.

    This prevents the model from being "lazy" and only learning
    easy-to-predict tokens like TIME_SHIFT.
    """
    weights = torch.ones(tokenizer.vocab_size, device=device)

    for token_str, token_id in tokenizer.token_to_id.items():
        if '_ON_' in token_str:
            # NOTE_ON: most important - getting notes right IS the music
            weights[token_id] = note_weight
        elif '_OFF_' in token_str:
            # NOTE_OFF: moderately important - controls note duration / rhythm
            weights[token_id] = note_weight * 0.6
        elif token_str.startswith('T_'):
            # TIME_SHIFT: less important - timing matters but is easier to learn
            weights[token_id] = time_weight

    # PAD token: zero weight (ignore completely)
    weights[0] = 0.0

    on_count = sum(1 for t in tokenizer.token_to_id if '_ON_' in t)
    off_count = sum(1 for t in tokenizer.token_to_id if '_OFF_' in t)
    time_count = sum(1 for t in tokenizer.token_to_id if t.startswith('T_'))

    print(f"  Loss weights:")
    print(f"    NOTE_ON ({on_count} tokens): {note_weight}x")
    print(f"    NOTE_OFF ({off_count} tokens): {note_weight * 0.6}x")
    print(f"    TIME_SHIFT ({time_count} tokens): {time_weight}x")
    print(f"    PAD: 0x (ignored)")

    return weights


class Trainer:
    def __init__(
        self,
        model, train_loader, val_loader,
        tokenizer,
        device='cpu',
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=2000,
        num_epochs=50,
        grad_accum_steps=4,
        checkpoint_dir='checkpoints',
        config=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.config = config

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate
        self.current_step = 0
        total_steps = len(train_loader) * num_epochs // grad_accum_steps
        self.total_steps = total_steps

        # Weighted loss with label smoothing
        # Label smoothing = 0.1 prevents overconfidence (helps avoid mode collapse)
        token_weights = build_token_weights(tokenizer, note_weight=3.0, device=device)
        self.criterion = nn.CrossEntropyLoss(
            weight=token_weights,
            label_smoothing=0.1,
        )

        self.max_grad_norm = max_grad_norm

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'learning_rate': [], 'perplexity': [],
        }

        eff_batch = train_loader.batch_size * grad_accum_steps
        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {train_loader.batch_size} x {grad_accum_steps} accum = {eff_batch} effective")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  Label smoothing: 0.1")

    def _update_learning_rate(self):
        """Warmup + cosine annealing."""
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / max(self.warmup_steps, 1))
        else:
            progress = (self.current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            lr = self.base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
            lr = max(lr, self.base_lr * 0.01)  # Floor at 1% of base LR

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_batches = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        for batch_idx, (input_seq, target_seq) in enumerate(self.train_loader):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Forward
            logits = self.model(input_seq)

            # Weighted cross-entropy loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_seq.reshape(-1)
            )

            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()

            total_loss += loss.item() * self.grad_accum_steps
            total_batches += 1

            # Optimizer step every grad_accum_steps
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                lr = self._update_learning_rate()
                self.current_step += 1

            # Progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / total_batches
                ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 99999)
                elapsed = time.time() - start_time
                batches_per_sec = total_batches / elapsed

                print(f"  [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"loss={avg_loss:.4f} ppl={ppl:.1f} "
                      f"lr={self._get_lr():.2e} "
                      f"batch/s={batches_per_sec:.1f}")

        avg_loss = total_loss / max(total_batches, 1)
        ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 99999)
        return avg_loss, ppl

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_batches = 0

        for input_seq, target_seq in self.val_loader:
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            logits = self.model(input_seq)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_seq.reshape(-1)
            )
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 99999)
        return avg_loss, ppl

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'step': self.current_step,
            'config': self.config,
        }

        path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, path)

        latest = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest)

        # Also save best
        if not hasattr(self, '_best_val_loss') or val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            best = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best)

        print(f"    Saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.history = ckpt.get('history', self.history)
        self.current_step = ckpt.get('step', 0)
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt['epoch']

    def train(self, num_epochs, save_every=5):
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 70)

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)

            train_loss, train_ppl = self.train_epoch(epoch)

            print(f"  Validating...")
            val_loss, val_ppl = self.validate()

            epoch_time = time.time() - epoch_start

            print(f"\n  Epoch {epoch} done in {epoch_time/60:.1f} min:")
            print(f"    Train Loss: {train_loss:.4f} | Perplexity: {train_ppl:.1f}")
            print(f"    Val Loss:   {val_loss:.4f} | Perplexity: {val_ppl:.1f}")

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['perplexity'].append(val_ppl)
            self.history['learning_rate'].append(self._get_lr())

            if epoch % save_every == 0 or val_loss < best_val_loss:
                self.save_checkpoint(epoch, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"    New best!")

        print("\n" + "=" * 70)
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")

        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    print("Music Transformer v2 Training")
    print("=" * 70)

    CONFIG = {
        # Data
        'midi_dir': '~/augmented_dataset',
        'seq_len': 1024,
        'batch_size': 8,
        'num_workers': 4,
        'stride': 512,       # Half overlap for good variety

        # Model
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,

        # Training
        'num_epochs': 50,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'warmup_steps': 2000,
        'max_grad_norm': 1.0,
        'grad_accum_steps': 4,  # Effective batch = 8 * 4 = 32

        # Loss
        'note_weight': 3.0,     # NOTE_ON gets 3x weight
        'label_smoothing': 0.1, # Prevents mode collapse

        # Device
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    }

    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # Tokenizer
    print("\nCreating tokenizer...")
    tokenizer = MIDITokenizer()

    # Data
    print(f"\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_dir=CONFIG['midi_dir'],
        val_dir=CONFIG['midi_dir'],
        tokenizer=tokenizer,
        seq_len=CONFIG['seq_len'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        stride=CONFIG['stride']
    )

    # Model
    print(f"\nCreating model...")
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        d_ff=CONFIG['d_ff'],
        max_seq_len=CONFIG['seq_len'],
        dropout=CONFIG['dropout'],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        max_grad_norm=CONFIG['max_grad_norm'],
        warmup_steps=CONFIG['warmup_steps'],
        num_epochs=CONFIG['num_epochs'],
        grad_accum_steps=CONFIG['grad_accum_steps'],
        checkpoint_dir='checkpoints',
        config=CONFIG,
    )

    # Train
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_every=5
    )


if __name__ == '__main__':
    main()
