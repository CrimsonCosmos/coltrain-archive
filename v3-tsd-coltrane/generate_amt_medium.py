#!/usr/bin/env python3
"""Generate from AMT medium model with correct settings."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi
from anticipation.config import MAX_PITCH
from anticipation.vocab import NOTE_OFFSET

print("Loading AMT medium model (360M params)...")
model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k')

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Device: {device}")
model = model.to(device)
model.eval()

# Generate 30 seconds with paper-recommended top_p=0.95
print("\nGenerating 30 seconds (medium model, top_p=0.95)...")
events = generate(model, start_time=0, end_time=30, top_p=0.95)

print(f"\nGenerated {len(events)//3} events")

# Convert to MIDI
output_path = "/Users/dylangehl/coltrain/amt_medium_sample.mid"
mid = events_to_midi(events)
mid.save(output_path)
print(f"Saved to: {output_path}")

# Report instruments
notes = events[2::3]
instruments = set()
for n in notes:
    instr = (n - NOTE_OFFSET) // MAX_PITCH
    instruments.add(instr)

GM_NAMES = {
    0: "Piano", 1: "Bright Piano", 4: "E.Piano", 5: "FM Piano",
    24: "Nylon Guitar", 25: "Steel Guitar", 26: "Jazz Guitar",
    29: "Overdrive Guitar", 30: "Distortion Guitar",
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)",
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    46: "Orchestral Harp", 48: "String Ensemble",
    56: "Trumpet", 57: "Trombone", 60: "French Horn",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    68: "Oboe", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    80: "Square Lead", 89: "Warm Pad",
    128: "Drums"
}

print(f"\nInstruments ({len(instruments)}):")
for i in sorted(instruments):
    name = GM_NAMES.get(i, f"GM Program {i}")
    print(f"  {name} ({i})")
