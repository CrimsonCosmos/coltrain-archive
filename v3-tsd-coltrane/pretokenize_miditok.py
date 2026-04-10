#!/usr/bin/env python3
"""
Pre-tokenize jazz MIDI dataset with MidiTok TSD -> numpy .npy for cloud training.

Usage:
    python pretokenize_miditok.py \
        --input_dir /Users/dylangehl/augmented_dataset_v2 \
        --output jazz_miditok.npy \
        --seq_len 1536 \
        --stride 768

Output:
    - jazz_miditok.npy: shape (N, seq_len) dtype uint16
    - jazz_miditok_groups.json: maps sequence index -> song group for train/val splitting
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

from data.tokenizer_miditok import MidiTokWrapper


def _extract_song_group(filename: str) -> str:
    """Extract base song name, stripping transposition suffixes.

    'AutumnLeaves_milesIntro#p-3.mid' -> 'AutumnLeaves_milesIntro'
    'autumn_leaves_jpa#.mid'          -> 'autumn_leaves_jpa'
    'Sandu.mid'                       -> 'Sandu'
    """
    stem = Path(filename).stem
    if '#' in stem:
        stem = stem[:stem.index('#')]
    return stem


def collect_midi_files(input_dir):
    """Recursively find all .mid/.midi files."""
    p = Path(input_dir)
    files = sorted(list(p.rglob("*.mid")) + list(p.rglob("*.midi")) + list(p.rglob("*.MID")))
    return files


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize MIDI dataset with MidiTok TSD')
    parser.add_argument('--input_dir', required=True, help='Directory containing MIDI files')
    parser.add_argument('--output', required=True, help='Output .npy file path')
    parser.add_argument('--seq_len', type=int, default=1536)
    parser.add_argument('--stride', type=int, default=768)
    parser.add_argument('--chunk_size', type=int, default=5000,
                        help='Files per chunk (controls memory usage)')
    args = parser.parse_args()

    wrapper = MidiTokWrapper()
    print(f"MidiTok TSD tokenizer: vocab_size={wrapper.vocab_size}")

    print("Collecting MIDI files...")
    midi_files = collect_midi_files(args.input_dir)
    print(f"Found {len(midi_files)} MIDI files", flush=True)

    t0 = time.time()
    seq_len = args.seq_len
    stride = args.stride
    chunk_size = args.chunk_size
    pad_id = wrapper.pad_id

    # Process in chunks, save each chunk as a temp .npy file
    temp_dir = tempfile.mkdtemp(prefix='pretok_miditok_')
    chunk_files = []
    total_sequences = 0
    errors = 0
    # Track song group per sequence (for train/val splitting)
    all_groups = []

    for chunk_start in range(0, len(midi_files), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(midi_files))
        chunk_seqs = []
        chunk_groups = []

        for i in range(chunk_start, chunk_end):
            f = midi_files[i]
            song_group = _extract_song_group(f.name)
            try:
                tokens = wrapper.tokenize_midi(str(f))

                if len(tokens) < seq_len // 2:
                    continue

                if len(tokens) < seq_len:
                    tokens = tokens + [pad_id] * (seq_len - len(tokens))
                    chunk_seqs.append(tokens[:seq_len])
                    chunk_groups.append(song_group)
                    continue

                for start in range(0, len(tokens) - seq_len + 1, stride):
                    chunk_seqs.append(tokens[start:start + seq_len])
                    chunk_groups.append(song_group)
            except Exception:
                errors += 1

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
              f"{total_sequences} sequences, {errors} errors, "
              f"{elapsed/60:.1f} min", flush=True)

    print(f"\nTokenization complete in {(time.time()-t0)/60:.1f} min")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Unique song groups: {len(set(all_groups))}")
    print(f"  Errors: {errors}")
    print(f"  Chunks saved: {len(chunk_files)}", flush=True)

    # Concatenate all chunks into final array
    print("Concatenating chunks into final array...", flush=True)
    all_chunks = [np.load(f) for f in chunk_files]
    final = np.concatenate(all_chunks, axis=0)
    del all_chunks

    np.save(args.output, final)

    file_size = os.path.getsize(args.output) / (1024**3)
    print(f"  Saved: {args.output} ({file_size:.2f} GB)")
    print(f"  Shape: {final.shape}", flush=True)

    # Save song group mapping
    groups_path = args.output.replace('.npy', '_groups.json')
    with open(groups_path, 'w') as f:
        json.dump(all_groups, f)
    print(f"  Groups: {groups_path} ({len(set(all_groups))} unique groups)", flush=True)

    # Cleanup temp files
    for f in chunk_files:
        os.remove(f)
    os.rmdir(temp_dir)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
