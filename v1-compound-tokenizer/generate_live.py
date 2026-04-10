#!/usr/bin/env python3
"""
Coltrain Live - Real-time infinite jazz generation
Inspired by the Swift toy but using Coltrain's learned patterns.
"""
import time
import random
import argparse
from pathlib import Path
import mido

# Import from our generator
import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate import Coltrain


def play_live(coltrain, tempo=120):
    """
    Generate and play jazz infinitely in real-time.
    Uses Coltrain's learned patterns but outputs live MIDI.
    """
    print("🎷 Coltrain Live - Infinite Jazz")
    print("=" * 50)
    print(f"Tempo: {tempo} BPM")
    print("Press Ctrl+C to stop")
    print()

    # Try to open MIDI output port
    try:
        available_ports = mido.get_output_names()
        if not available_ports:
            print("❌ No MIDI output ports found!")
            print("   On Mac, use IAC Driver or install a virtual MIDI port")
            return

        print(f"Available MIDI ports:")
        for i, port in enumerate(available_ports):
            print(f"  {i}: {port}")

        # Use first available port
        port_name = available_ports[0]
        print(f"\n✅ Using: {port_name}")
        print()

        outport = mido.open_output(port_name)
    except Exception as e:
        print(f"❌ Error opening MIDI port: {e}")
        return

    # Get chord progression
    if not coltrain.chord_progressions:
        print("❌ No chord progressions learned!")
        return

    # Use a repeating chord progression
    base_progression = random.choice(coltrain.chord_progressions)
    print(f"Chord progression: {base_progression[:8]}... (repeating)")
    print()
    print("🎺 Playing... (Ctrl+C to stop)")
    print()

    measure_count = 0

    try:
        while True:  # Infinite loop
            for chord_root, chord_type in base_progression:
                measure_count += 1

                # Print progress occasionally
                if measure_count % 8 == 0:
                    print(f"  Measure {measure_count}...")

                # Generate and play notes for this measure
                play_measure(outport, coltrain, chord_root, chord_type, tempo)

    except KeyboardInterrupt:
        print("\n\n✨ Stopped!")
        print(f"   Generated {measure_count} measures")

        # Send all notes off
        for channel in range(3):
            outport.send(mido.Message('control_change', channel=channel, control=123, value=0))

        outport.close()


def play_measure(outport, coltrain, chord_root, chord_type, tempo):
    """Play one measure of jazz."""
    beat_duration = 60.0 / tempo  # seconds per beat

    # Simplified: just play melody and bass (drums/piano would need threading)

    # MELODY (8 eighth notes)
    scale = [chord_root + offset for offset in [0, 2, 4, 7, 9, 12, 14, 16]]

    for beat in range(8):
        # Choose note
        if coltrain.melody_patterns and random.random() < 0.7:
            state = random.choice(list(coltrain.melody_patterns.keys()))
            candidates = coltrain.melody_patterns[state]
            note = random.choices(list(candidates.keys()), list(candidates.values()))[0]
            # Fit to scale
            pitch_class = note % 12
            octave = note // 12
            scale_pcs = [n % 12 for n in scale]
            closest = min(scale_pcs, key=lambda x: abs(x - pitch_class))
            note = octave * 12 + closest
        else:
            note = random.choice(scale) + 60

        note = max(60, min(84, note))
        velocity = 90 if beat % 2 == 0 else 70

        # Play note
        outport.send(mido.Message('note_on', note=note, velocity=velocity, channel=0))

        # Duration
        duration = beat_duration / 2  # Eighth note
        time.sleep(duration * 0.9)  # 90% duration

        outport.send(mido.Message('note_off', note=note, channel=0))
        time.sleep(duration * 0.1)  # 10% gap


def main():
    parser = argparse.ArgumentParser(
        description='Coltrain Live - Real-time infinite jazz generation',
        epilog="""
This mode generates jazz infinitely in real-time.
Unlike the file-based generator, this never stops.

Requirements:
  - MIDI output port (IAC Driver on Mac, loopMIDI on Windows)
  - DAW or MIDI monitor to hear output

Examples:
  # Play live jazz at 120 BPM
  python generate_live.py --tempo 120

  # Slower tempo
  python generate_live.py --tempo 90
        """
    )

    parser.add_argument(
        '--midi-dir',
        default='/Users/dylangehl/augmented_dataset',
        help='Training MIDI directory'
    )
    parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='Tempo in BPM'
    )

    args = parser.parse_args()

    # Load Coltrain
    print("Loading Coltrain...")
    coltrain = Coltrain(args.midi_dir)

    if not coltrain.load_and_analyze():
        print("❌ Failed to load training data")
        return

    # Play live
    play_live(coltrain, tempo=args.tempo)


if __name__ == '__main__':
    main()
