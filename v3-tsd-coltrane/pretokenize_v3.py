#!/usr/bin/env python3
"""
Pre-tokenize jazz MIDI dataset in track-sequential format for v3 training.

Usage:
    python -u pretokenize_v3.py \
        --input_dir ~/augmented_dataset_v2 \
        --output jazz_v3.npy \
        --seq_len 1024 \
        --stride 512

Output:
    - jazz_v3.npy: shape (N, seq_len) dtype uint16
    - jazz_v3_groups.json: maps sequence index -> song group for train/val splitting
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformer'))

import numpy as np
from pathlib import Path
import argparse
import time
import tempfile
import warnings

warnings.filterwarnings('ignore')

from data.tokenizer_v3 import TrackSeqTokenizer


def _extract_song_group(filename):
    """Extract base song name, stripping transposition suffixes.

    'AutumnLeaves_milesIntro#p-3.mid' -> 'AutumnLeaves_milesIntro'
    'Sandu.mid'                       -> 'Sandu'
    """
    stem = Path(filename).stem
    if '#' in stem:
        stem = stem[:stem.index('#')]
    return stem


def collect_midi_files(input_dir):
    """Recursively find all .mid/.midi files."""
    p = Path(input_dir)
    return sorted(list(p.rglob("*.mid")) + list(p.rglob("*.midi")) + list(p.rglob("*.MID")))


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize MIDI in track-sequential format')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=1024)
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Files per chunk (controls memory)')
    parser.add_argument('--min_roles', type=int, default=2,
                        help='Skip files with fewer than N non-empty roles')
    args = parser.parse_args()

    tok = TrackSeqTokenizer()
    print(f"TrackSeqTokenizer: vocab_size={tok.vocab_size}")

    midi_files = collect_midi_files(args.input_dir)
    print(f"Found {len(midi_files)} MIDI files", flush=True)

    t0 = time.time()
    seq_len = args.seq_len
    stride = args.stride
    pad_id = tok.pad_id

    temp_dir = tempfile.mkdtemp(prefix='pretok_v3_')
    chunk_files = []
    total_sequences = 0
    errors = 0
    skipped_few_roles = 0
    all_groups = []

    # Track role distribution stats
    role_counts = {'lead': 0, 'comping': 0, 'bass': 0, 'drums': 0}
    total_files_ok = 0

    for chunk_start in range(0, len(midi_files), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(midi_files))
        chunk_seqs = []
        chunk_groups = []

        for i in range(chunk_start, chunk_end):
            f = midi_files[i]
            song_group = _extract_song_group(f.name)
            try:
                tokens = tok.tokenize_midi(str(f))

                # Check non-empty roles
                sections = tok._split_at_delimiters(tokens)
                non_empty = sum(1 for r in ['lead', 'comping', 'bass', 'drums']
                                if sections.get(r))
                if non_empty < args.min_roles:
                    skipped_few_roles += 1
                    continue

                # Track role stats
                for role in ['lead', 'comping', 'bass', 'drums']:
                    if sections.get(role):
                        role_counts[role] += 1
                total_files_ok += 1

                # Skip very short sequences
                if len(tokens) < seq_len // 2:
                    continue

                # Pad short sequences
                if len(tokens) < seq_len:
                    tokens = tokens + [pad_id] * (seq_len - len(tokens))
                    chunk_seqs.append(tokens[:seq_len])
                    chunk_groups.append(song_group)
                    continue

                # Sliding window for long sequences
                for start in range(0, len(tokens) - seq_len + 1, stride):
                    chunk_seqs.append(tokens[start:start + seq_len])
                    chunk_groups.append(song_group)

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error: {f.name}: {e}", flush=True)

        if chunk_seqs:
            chunk_arr = np.array(chunk_seqs, dtype=np.uint16)
            chunk_path = os.path.join(temp_dir, f'chunk_{len(chunk_files):04d}.npy')
            np.save(chunk_path, chunk_arr)
            chunk_files.append(chunk_path)
            total_sequences += len(chunk_seqs)
            all_groups.extend(chunk_groups)
            del chunk_seqs, chunk_arr

        elapsed = time.time() - t0
        print(f"  [{chunk_end}/{len(midi_files)}] "
              f"{total_sequences} seqs, {errors} errors, "
              f"{skipped_few_roles} skipped (few roles), "
              f"{elapsed/60:.1f} min", flush=True)

    print(f"\nTokenization complete in {(time.time()-t0)/60:.1f} min")
    print(f"  Files OK: {total_files_ok}")
    print(f"  Sequences: {total_sequences}")
    print(f"  Unique groups: {len(set(all_groups))}")
    print(f"  Errors: {errors}")
    print(f"  Skipped (< {args.min_roles} roles): {skipped_few_roles}")
    print(f"  Role distribution:")
    for role, count in role_counts.items():
        pct = count / max(total_files_ok, 1) * 100
        print(f"    {role:8s}: {count:5d} files ({pct:.0f}%)")
    print(flush=True)

    # Concatenate chunks
    print("Concatenating chunks...", flush=True)
    all_chunks = [np.load(f) for f in chunk_files]
    final = np.concatenate(all_chunks, axis=0)
    del all_chunks

    np.save(args.output, final)
    file_size = os.path.getsize(args.output) / (1024**3)
    print(f"  Saved: {args.output} ({file_size:.2f} GB)")
    print(f"  Shape: {final.shape}", flush=True)

    # Save groups
    groups_path = args.output.replace('.npy', '_groups.json')
    with open(groups_path, 'w') as f:
        json.dump(all_groups, f)
    print(f"  Groups: {groups_path} ({len(set(all_groups))} unique)", flush=True)

    # Cleanup
    for f in chunk_files:
        os.remove(f)
    os.rmdir(temp_dir)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
