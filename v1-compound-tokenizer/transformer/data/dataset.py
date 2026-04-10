#!/usr/bin/env python3
"""
MIDI Dataset for Transformer Training
Loads MIDI files, tokenizes them, creates training sequences.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import List
import random
from data.tokenizer import MIDITokenizer


def _extract_song_group(filename: str) -> str:
    """Extract the base song name, stripping transposition suffixes.

    'AutumnLeaves_milesIntro#p-3.mid' -> 'AutumnLeaves_milesIntro'
    'autumn_leaves_jpa#.mid'          -> 'autumn_leaves_jpa'
    'Sandu.mid'                       -> 'Sandu'
    """
    stem = Path(filename).stem
    if '#' in stem:
        stem = stem[:stem.index('#')]
    return stem


class MIDIDataset(Dataset):
    """
    PyTorch dataset for MIDI sequences.

    Each sample is a subsequence of tokens from a MIDI file.
    Target is the same sequence shifted by 1 (next token prediction).
    """

    def __init__(
        self,
        midi_dir: str,
        tokenizer: MIDITokenizer,
        seq_len: int = 512,
        stride: int = 256,
        augment: bool = True
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment

        # Load and tokenize all MIDI files
        print(f"Loading MIDI files from: {midi_dir}")
        self.sequences = []
        self.file_boundaries = []
        self.song_groups = []  # Track song group for each sequence

        midi_files = sorted(
            list(Path(midi_dir).glob("*.mid")) + list(Path(midi_dir).glob("*.midi"))
        )
        print(f"   Found {len(midi_files)} MIDI files")

        for file_idx, midi_file in enumerate(midi_files):
            try:
                tokens = tokenizer.tokenize_midi(str(midi_file))

                if len(tokens) < seq_len:
                    continue

                song_group = _extract_song_group(midi_file.name)
                num_sequences = (len(tokens) - seq_len) // stride + 1

                for i in range(num_sequences):
                    start = i * stride
                    end = start + seq_len

                    if end <= len(tokens):
                        seq = tokens[start:end]
                        self.sequences.append(seq)
                        self.file_boundaries.append(file_idx)
                        self.song_groups.append(song_group)

                if (file_idx + 1) % 100 == 0:
                    print(f"   Processed {file_idx + 1}/{len(midi_files)} files...")

            except Exception as e:
                print(f"   Error processing {midi_file.name}: {e}")

        unique_groups = len(set(self.song_groups))
        print(f"Dataset created:")
        print(f"   Total sequences: {len(self.sequences)}")
        print(f"   Unique song groups: {unique_groups}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Total tokens: {len(self.sequences) * seq_len:,}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if self.augment and len(seq) > self.seq_len:
            offset = random.randint(0, min(8, len(seq) - self.seq_len))
            seq = seq[offset:offset + self.seq_len]

        if len(seq) < self.seq_len:
            seq = seq + [self.tokenizer.EVENT_PAD] * (self.seq_len - len(seq))
        elif len(seq) > self.seq_len:
            seq = seq[:self.seq_len]

        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)

        return input_seq, target_seq


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    tokenizer: MIDITokenizer,
    seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    stride: int = 256,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Create training and validation dataloaders.

    When train_dir == val_dir, splits by song group so transpositions
    of the same song never appear in both train and val sets.
    """
    if val_dir == train_dir:
        full_dataset = MIDIDataset(
            train_dir, tokenizer, seq_len=seq_len, stride=stride, augment=True
        )

        # Group-aware split
        unique_groups = sorted(set(full_dataset.song_groups))
        rng = random.Random(seed)
        rng.shuffle(unique_groups)

        val_count = max(1, int(len(unique_groups) * val_ratio))
        val_groups = set(unique_groups[:val_count])
        train_groups = set(unique_groups[val_count:])

        train_indices = [i for i, g in enumerate(full_dataset.song_groups) if g in train_groups]
        val_indices = [i for i, g in enumerate(full_dataset.song_groups) if g in val_groups]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        print(f"\n   Song-group split: {len(train_groups)} train / {len(val_groups)} val groups")
        print(f"   Train sequences: {len(train_indices)}, Val sequences: {len(val_indices)}")
    else:
        train_dataset = MIDIDataset(
            train_dir, tokenizer, seq_len=seq_len, stride=stride, augment=True
        )
        val_dataset = MIDIDataset(
            val_dir, tokenizer, seq_len=seq_len, stride=stride, augment=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"\n   Dataloaders created:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Batch size: {batch_size}")

    return train_loader, val_loader


class PreTokenizedDataset(Dataset):
    """Memory-mapped dataset for pre-tokenized binary data (numpy .npy files)."""

    def __init__(self, bin_path: str, seq_len: int = 1024):
        self.seq_len = seq_len
        self.data = np.load(bin_path, mmap_mode='r')
        print(f"PreTokenizedDataset: {self.data.shape[0]:,} sequences, "
              f"seq_len={self.data.shape[1]}, "
              f"total tokens: {self.data.shape[0] * self.data.shape[1]:,}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx].astype(np.int64)
        input_seq = torch.from_numpy(seq[:-1].copy())
        target_seq = torch.from_numpy(seq[1:].copy())
        return input_seq, target_seq


def create_pretrain_dataloader(
    bin_path: str,
    seq_len: int = 1024,
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.02,
    seed: int = 42
):
    """Create dataloaders from a pre-tokenized .npy file."""
    dataset = PreTokenizedDataset(bin_path, seq_len=seq_len)

    n = len(dataset)
    val_size = max(1, int(n * val_ratio))

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"\n   Pretrain dataloaders: {len(train_loader)} train / {len(val_loader)} val batches")
    return train_loader, val_loader


if __name__ == '__main__':
    from data.tokenizer import MIDITokenizer

    tokenizer = MIDITokenizer()
    midi_dir = "/Users/dylangehl/augmented_dataset"

    if not Path(midi_dir).exists():
        print(f"Directory not found: {midi_dir}")
    else:
        train_loader, val_loader = create_dataloaders(
            train_dir=midi_dir,
            val_dir=midi_dir,
            tokenizer=tokenizer,
            seq_len=256,
            batch_size=4,
            num_workers=0,
            stride=128,
        )

        batch_input, batch_target = next(iter(train_loader))
        print(f"\n   Batch input shape: {batch_input.shape}")
        print(f"   Batch target shape: {batch_target.shape}")
        print("\n   Dataset test complete!")
