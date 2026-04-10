#!/usr/bin/env python3
"""Preprocess jazz MIDI files into AMT training sequences."""

import os
import json
import pickle
from pathlib import Path
from anticipation.convert import midi_to_events
from anticipation.vocab import TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, SEPARATOR, AUTOREGRESS
from anticipation.config import CONTEXT_SIZE, EVENT_SIZE, MAX_PITCH

MIDI_DIR = "/Users/dylangehl/augmented_dataset"
OUTPUT_DIR = "/Users/dylangehl/coltrain/amt_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert all MIDI files to AMT events
print("Converting jazz MIDI files to AMT token format...")
print(f"Source: {MIDI_DIR}")

midi_files = sorted(Path(MIDI_DIR).glob("*.mid"))
print(f"Found {len(midi_files)} MIDI files\n")

all_sequences = []
errors = 0
too_short = 0
total_events = 0

for i, midi_file in enumerate(midi_files):
    try:
        events = midi_to_events(str(midi_file))
        n_events = len(events) // 3

        if n_events < 50:  # Skip very short files
            too_short += 1
            continue

        total_events += n_events

        # Create overlapping windows of 1024 tokens
        # Header: [AUTOREGRESS] (1 token) + events fill remaining 1023 tokens
        # 1023 tokens = 341 event triplets
        M = 341  # max events per window
        stride = M // 2  # 50% overlap

        for start in range(0, n_events - M + 1, stride):
            chunk_start = start * 3
            chunk_end = (start + M) * 3
            chunk = events[chunk_start:chunk_end]

            # Relativize time (subtract minimum time in chunk)
            chunk = list(chunk)
            times = chunk[0::3]
            min_t = min(times)
            for j in range(0, len(chunk), 3):
                chunk[j] = chunk[j] - min_t

            # Prepend mode token
            seq = [AUTOREGRESS] + chunk

            # Should be exactly 1024 tokens
            assert len(seq) == CONTEXT_SIZE, f"Expected {CONTEXT_SIZE}, got {len(seq)}"

            all_sequences.append(seq)

    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  Error: {midi_file.name}: {e}")

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(midi_files)} files ({len(all_sequences)} sequences so far)")

print(f"\nDone!")
print(f"  Files processed: {len(midi_files) - errors - too_short}")
print(f"  Errors: {errors}")
print(f"  Too short: {too_short}")
print(f"  Total events: {total_events:,}")
print(f"  Training sequences: {len(all_sequences)}")
print(f"  Tokens per sequence: {CONTEXT_SIZE}")
print(f"  Total tokens: {len(all_sequences) * CONTEXT_SIZE:,}")

# Analyze instrument distribution
print("\nInstrument distribution across all sequences:")
instr_counts = {}
for seq in all_sequences:
    notes = seq[3::3]  # skip header, get every 3rd token starting from note position
    for n in notes:
        if n >= NOTE_OFFSET:
            instr = (n - NOTE_OFFSET) // MAX_PITCH
            instr_counts[instr] = instr_counts.get(instr, 0) + 1

GM_NAMES = {
    0: "Piano", 24: "Nylon Guitar", 25: "Steel Guitar", 26: "Jazz Guitar",
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)",
    56: "Trumpet", 57: "Trombone",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    73: "Flute", 128: "Drums"
}

for instr, count in sorted(instr_counts.items(), key=lambda x: -x[1])[:20]:
    name = GM_NAMES.get(instr, f"GM Program {instr}")
    print(f"  {name} ({instr}): {count:,} events")

# Save preprocessed data
output_path = os.path.join(OUTPUT_DIR, "jazz_sequences.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(all_sequences, f)
print(f"\nSaved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
