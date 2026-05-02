#!/usr/bin/env python3
"""Generate a sample from pre-trained AMT (no fine-tuning) to hear the baseline."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi

print("Loading pre-trained AMT (stanford-crfm/music-small-800k)...")
print("This will download ~500MB on first run.")

model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-small-800k')

# Use MPS if available
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Using device: {device}")
model = model.to(device)
model.eval()

# Generate 30 seconds of music from scratch
print("\nGenerating 30 seconds of multi-track music...")
print("(This may take a few minutes on CPU/MPS)")

events = generate(model, start_time=0, end_time=30, top_p=0.98)

print(f"\nGenerated {len(events)//3} events")

# Convert to MIDI
output_path = "~/coltrain/amt_pretrained_sample.mid"
mid = events_to_midi(events)
mid.save(output_path)
print(f"Saved to: {output_path}")

# Count instruments
from anticipation.config import MAX_PITCH
from anticipation.vocab import NOTE_OFFSET
notes = events[2::3]
instruments = set()
for n in notes:
    instr = (n - NOTE_OFFSET) // MAX_PITCH
    instruments.add(instr)

instr_names = {
    0: "Piano", 24: "Guitar", 25: "Steel Guitar", 32: "Acoustic Bass",
    33: "Electric Bass", 40: "Violin", 42: "Cello", 48: "Strings",
    56: "Trumpet", 64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax",
    67: "Baritone Sax", 73: "Flute", 128: "Drums"
}

print(f"\nInstruments used ({len(instruments)}):")
for i in sorted(instruments):
    name = instr_names.get(i, f"GM Program {i}")
    print(f"  {name} (program {i})")

print("\nDone! Open the MIDI file to listen.")
