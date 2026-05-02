#!/usr/bin/env python3
"""Generate from AMT medium on CPU to rule out MPS issues."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi
from anticipation.config import MAX_PITCH
from anticipation.vocab import NOTE_OFFSET

print("Loading AMT medium model on CPU...")
model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k')
model = model.to('cpu')
model.eval()

# Check model loaded correctly
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
print(f"Device: {next(model.parameters()).device}")

# Generate 15 seconds (shorter to be faster on CPU)
print("\nGenerating 15 seconds on CPU (top_p=0.95)...")
events = generate(model, start_time=0, end_time=15, top_p=0.95)

n_events = len(events) // 3
print(f"\nGenerated {n_events} events")

if n_events > 0:
    output_path = "~/coltrain/amt_medium_cpu.mid"
    mid = events_to_midi(events)
    mid.save(output_path)
    print(f"Saved to: {output_path}")

    # Report instruments
    notes = events[2::3]
    instruments = set()
    for n in notes:
        instr = (n - NOTE_OFFSET) // MAX_PITCH
        instruments.add(instr)
    print(f"Instruments: {len(instruments)}")
    for i in sorted(instruments):
        print(f"  Program {i}")
else:
    print("No events generated!")
