#!/usr/bin/env python3
"""
Coltrain - AI Jazz Generation with Harmonic Coordination and Rhythm Variation
Learns chord progressions, melody patterns, and rhythm from your jazz MIDI files.
"""
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class Coltrain:
    """AI Jazz Generator with chord-aware generation and learned rhythm."""

    def __init__(self, midi_dir, order=2):
        self.midi_dir = Path(midi_dir)
        self.order = order

        # Learned structures
        self.chord_progressions = []
        self.melody_patterns = defaultdict(Counter)
        self.melody_rhythms = []  # (duration, rest_after) tuples
        self.piano_rhythms = []
        self.bass_rhythms = []

        # Instruments
        self.instruments = {
            'melody': 64,  # Sax
            'piano': 0,    # Piano
            'bass': 32,    # Bass
        }

    def _detect_chord(self, notes):
        """Detect chord root from simultaneous notes."""
        if not notes:
            return None
        pitch_classes = sorted(set(n % 12 for n in notes))
        if len(pitch_classes) < 2:
            return pitch_classes[0] if pitch_classes else None
        return min(notes) % 12

    def load_and_analyze(self):
        """Load MIDIs and learn patterns + rhythms."""
        print(f"🎵 Loading and analyzing MIDI files...")
        midi_files = list(self.midi_dir.glob("*.mid"))[:100]  # Limit for speed

        if not midi_files:
            print("❌ No MIDI files found")
            return False

        print(f"Analyzing {len(midi_files)} files...")

        for midi_file in midi_files:
            try:
                self._analyze_midi(midi_file)
            except Exception:
                pass

        print(f"\n📊 Learned:")
        print(f"   Chord progressions: {len(self.chord_progressions)}")
        print(f"   Melody patterns: {len(self.melody_patterns)}")
        print(f"   Melody rhythms: {len(self.melody_rhythms)}")
        print(f"   Piano rhythms: {len(self.piano_rhythms)}")
        print(f"   Bass rhythms: {len(self.bass_rhythms)}")

        return len(self.chord_progressions) > 0

    def _analyze_midi(self, midi_path):
        """Analyze MIDI file for patterns and rhythms."""
        mid = MidiFile(midi_path)
        ticks_per_beat = mid.ticks_per_beat

        # Collect track data
        tracks_data = {}
        for track in mid.tracks:
            notes = []
            current_time = 0
            active_notes = {}  # note -> start_time

            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = current_time
                elif msg.type in ['note_off', 'note_on'] and (msg.type == 'note_off' or msg.velocity == 0):
                    if msg.note in active_notes:
                        start = active_notes[msg.note]
                        duration = current_time - start
                        notes.append({
                            'note': msg.note,
                            'start': start,
                            'duration': duration
                        })
                        del active_notes[msg.note]

            if notes:
                # Classify track
                avg_note = sum(n['note'] for n in notes) / len(notes)
                note_count = len(notes)

                if note_count > 500:  # Likely piano/chords
                    tracks_data['piano'] = notes
                elif avg_note > 60 and note_count < 300:  # Melody
                    tracks_data['melody'] = notes
                elif avg_note < 50:  # Bass
                    tracks_data['bass'] = notes

        # Learn chord progression from piano/harmonic track
        if 'piano' in tracks_data:
            progression = self._extract_chord_progression(tracks_data['piano'], ticks_per_beat)
            if progression and len(progression) >= 4:
                self.chord_progressions.append(progression)

            # Learn piano rhythms
            self._learn_rhythms(tracks_data['piano'], self.piano_rhythms, ticks_per_beat)

        # Learn melody patterns and rhythms
        if 'melody' in tracks_data:
            melody_notes = sorted(tracks_data['melody'], key=lambda x: x['start'])

            # Learn note patterns
            note_seq = [n['note'] for n in melody_notes]
            for i in range(len(note_seq) - self.order):
                state = tuple(note_seq[i:i + self.order])
                next_note = note_seq[i + self.order]
                self.melody_patterns[state][next_note] += 1

            # Learn melody rhythms (including rests)
            self._learn_rhythms_with_rests(melody_notes, self.melody_rhythms, ticks_per_beat)

        # Learn bass rhythms
        if 'bass' in tracks_data:
            self._learn_rhythms(tracks_data['bass'], self.bass_rhythms, ticks_per_beat)

    def _extract_chord_progression(self, notes, ticks_per_beat, measure_duration=None):
        """Extract chord progression from harmonic track."""
        if not notes:
            return []

        if measure_duration is None:
            measure_duration = ticks_per_beat * 4  # 4 beats = 1 measure

        sorted_notes = sorted(notes, key=lambda x: x['start'])
        max_time = sorted_notes[-1]['start'] if sorted_notes else 0
        num_measures = max(1, int(max_time / measure_duration))

        progression = []
        for measure in range(num_measures):
            measure_start = measure * measure_duration
            measure_end = (measure + 1) * measure_duration

            measure_notes = [
                n['note'] for n in sorted_notes
                if measure_start <= n['start'] < measure_end
            ]

            if measure_notes:
                chord_root = self._detect_chord(measure_notes)
                if chord_root is not None:
                    progression.append(chord_root)

        return progression

    def _learn_rhythms(self, notes, rhythm_list, ticks_per_beat):
        """Learn rhythm patterns from note durations."""
        for note in notes[:50]:  # Sample
            # Normalize to quarter notes (480 ticks standard)
            normalized_duration = int((note['duration'] / ticks_per_beat) * 480)
            if 60 < normalized_duration < 2000:  # Reasonable range
                rhythm_list.append(normalized_duration)

    def _learn_rhythms_with_rests(self, sorted_notes, rhythm_list, ticks_per_beat):
        """Learn rhythm patterns including rests between notes."""
        for i in range(len(sorted_notes) - 1):
            current = sorted_notes[i]
            next_note = sorted_notes[i + 1]

            # Duration of note
            duration = int((current['duration'] / ticks_per_beat) * 480)
            # Rest after note (time between note end and next note start)
            note_end = current['start'] + current['duration']
            rest = int(((next_note['start'] - note_end) / ticks_per_beat) * 480)

            if 60 < duration < 2000 and -100 < rest < 1000:
                rhythm_list.append((duration, rest))

    def generate(self, num_measures=64, tempo=120, output_file='jazz.mid'):
        """Generate complete jazz piece."""
        print(f"\n🎺 Generating jazz...")
        print(f"   Measures: {num_measures}")
        print(f"   Tempo: {tempo} BPM")

        if not self.chord_progressions:
            print("❌ No training data!")
            return None

        # Choose and extend chord progression
        base_progression = random.choice(self.chord_progressions)
        chord_progression = []
        while len(chord_progression) < num_measures:
            chord_progression.extend(base_progression)
        chord_progression = chord_progression[:num_measures]

        print(f"   Chords: {chord_progression[:8]}...")

        # Create MIDI
        mid = MidiFile(ticks_per_beat=480)

        # Tempo track
        tempo_track = MidiTrack()
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        mid.tracks.append(tempo_track)

        # Generate tracks
        print("   🎤 Creating melody...")
        mid.tracks.append(self._create_melody(chord_progression))

        print("   🎹 Creating piano...")
        mid.tracks.append(self._create_piano(chord_progression))

        print("   🎸 Creating bass...")
        mid.tracks.append(self._create_bass(chord_progression))

        print("   🥁 Creating drums...")
        mid.tracks.append(self._create_drums(num_measures))

        # Save
        mid.save(output_file)
        duration = mid.length
        print(f"💾 Saved: {output_file}")
        print(f"   Duration: {duration:.1f}s ({duration/60:.1f} minutes)")

        return output_file

    def _create_melody(self, chord_progression):
        """Generate melody with learned rhythms and chord awareness."""
        track = MidiTrack()
        track.append(MetaMessage('track_name', name='Melody'))
        track.append(Message('program_change', program=self.instruments['melody'], channel=0))

        for chord_root in chord_progression:
            # Scale for this chord
            scale = [chord_root + offset for offset in [0, 2, 4, 7, 9, 12, 14, 16, 19, 21]]

            # Generate 4-8 notes per measure with varied rhythms
            notes_in_measure = random.randint(4, 8)

            for _ in range(notes_in_measure):
                # Choose note from scale
                if self.melody_patterns and random.random() < 0.7:
                    state = random.choice(list(self.melody_patterns.keys()))
                    candidates = self.melody_patterns[state]
                    note = random.choices(list(candidates.keys()), list(candidates.values()))[0]
                    note = self._fit_note_to_scale(note, scale)
                else:
                    note = random.choice(scale) + 60

                note = max(60, min(84, note))  # Melody range

                # Use learned rhythm or defaults
                if self.melody_rhythms and random.random() < 0.8:
                    rhythm = random.choice(self.melody_rhythms)
                    if isinstance(rhythm, tuple):
                        duration, rest = rhythm
                    else:
                        duration = rhythm
                        rest = 0
                else:
                    # Default jazz rhythms (swing eighths, quarters)
                    duration = random.choice([180, 240, 360, 480])  # Triplet, eighth, dotted, quarter
                    rest = random.choice([0, 0, 60, 120])  # Occasional rests

                # Clamp values
                duration = max(120, min(960, duration))
                rest = max(0, min(480, rest))

                velocity = random.randint(70, 100)

                # Note on
                track.append(Message('note_on', note=note, velocity=velocity, time=0, channel=0))
                # Note off
                track.append(Message('note_off', note=note, time=duration, channel=0))

                # Rest (if any) - time before next note
                if rest > 0:
                    # Add to the time of next note_on
                    pass  # Will be handled by next note's time=0 + cumulative

        return track

    def _create_piano(self, chord_progression):
        """Generate piano comping with rhythm variation."""
        track = MidiTrack()
        track.append(MetaMessage('track_name', name='Piano'))
        track.append(Message('program_change', program=self.instruments['piano'], channel=1))

        voicing = [0, 4, 7, 10]  # 7th chord

        for chord_root in chord_progression:
            # Comp 2-4 times per measure with varied rhythms
            comps_per_measure = random.randint(2, 4)

            for _ in range(comps_per_measure):
                chord_notes = [48 + chord_root + interval for interval in voicing]

                # Use learned piano rhythm or default
                if self.piano_rhythms and random.random() < 0.8:
                    duration = random.choice(self.piano_rhythms)
                else:
                    duration = random.choice([240, 360, 480])  # Eighth, dotted, quarter

                duration = max(180, min(720, duration))

                # Play chord
                for note in chord_notes:
                    note = max(36, min(72, note))
                    track.append(Message('note_on', note=note, velocity=random.randint(50, 70), time=0, channel=1))

                # Release chord
                for i, note in enumerate(chord_notes):
                    note = max(36, min(72, note))
                    time = duration if i == len(chord_notes) - 1 else 0
                    track.append(Message('note_off', note=note, time=time, channel=1))

                # Gap before next comp
                gap = random.choice([120, 240, 360])
                # (gap will be time=0 on next note_on)

        return track

    def _create_bass(self, chord_progression):
        """Generate walking bass with rhythm variation."""
        track = MidiTrack()
        track.append(MetaMessage('track_name', name='Bass'))
        track.append(Message('program_change', program=self.instruments['bass'], channel=2))

        for i, chord_root in enumerate(chord_progression):
            # Walking bass: 4 quarter notes per measure (classic)
            bass_pattern = [
                36 + chord_root,        # Root
                36 + chord_root + 7,    # 5th
                36 + chord_root + 4,    # 3rd
                36 + chord_root + 10    # 7th
            ]

            # Chromatic approach to next chord
            if i < len(chord_progression) - 1:
                next_root = chord_progression[i + 1]
                bass_pattern[3] = 36 + next_root - 1  # Half-step below next root

            for bass_note in bass_pattern:
                bass_note = max(28, min(55, bass_note))

                # Use learned bass rhythm or default quarter notes
                if self.bass_rhythms and random.random() < 0.7:
                    duration = random.choice(self.bass_rhythms)
                else:
                    duration = 480  # Quarter note (walking)

                duration = max(360, min(600, duration))

                velocity = random.randint(75, 95)

                track.append(Message('note_on', note=bass_note, velocity=velocity, time=0, channel=2))
                track.append(Message('note_off', note=bass_note, time=duration, channel=2))

        return track

    def _create_drums(self, num_measures):
        """Generate swing drums with rhythm variation."""
        track = MidiTrack()
        track.append(MetaMessage('track_name', name='Drums'))

        # Drum notes
        kick = 36
        snare = 38
        closed_hi_hat = 42
        open_hi_hat = 46
        ride = 51

        for measure in range(num_measures):
            # Vary pattern every few measures
            pattern_type = 'ride' if measure % 4 < 3 else 'hi_hat'

            if pattern_type == 'ride':
                # Ride cymbal swing pattern
                pattern = [
                    # Beat 1
                    [(ride, 75, 0)],
                    [(closed_hi_hat, 50, 240)],  # & of 1
                    # Beat 2
                    [(ride, 80, 0), (snare, 70, 0)],
                    [(closed_hi_hat, 55, 240)],  # & of 2
                    # Beat 3
                    [(ride, 75, 0), (kick, 80, 0)],
                    [(closed_hi_hat, 50, 240)],  # & of 3
                    # Beat 4
                    [(ride, 80, 0), (snare, 75, 0)],
                    [(closed_hi_hat, 55, 240)],  # & of 4
                ]
            else:
                # Hi-hat variation
                pattern = [
                    [(closed_hi_hat, 70, 0), (kick, 75, 0)],
                    [(closed_hi_hat, 55, 240)],
                    [(closed_hi_hat, 65, 0), (snare, 70, 0)],
                    [(open_hi_hat, 60, 240)],
                    [(closed_hi_hat, 70, 0)],
                    [(closed_hi_hat, 55, 240)],
                    [(closed_hi_hat, 65, 0), (snare, 75, 0)],
                    [(closed_hi_hat, 50, 240)],
                ]

            for hit_group in pattern:
                for drum, velocity, time in hit_group:
                    track.append(Message('note_on', note=drum, velocity=velocity, time=time, channel=9))
                    track.append(Message('note_off', note=drum, time=60, channel=9))

        return track

    def _fit_note_to_scale(self, note, scale):
        """Transpose note to fit within scale."""
        pitch_class = note % 12
        octave = note // 12
        scale_pcs = [n % 12 for n in scale]
        closest = min(scale_pcs, key=lambda x: abs(x - pitch_class))
        return octave * 12 + closest


def main():
    parser = argparse.ArgumentParser(
        description='Coltrain - AI Jazz Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 6-minute standard jazz
  python generate.py --measures 150 --tempo 120

  # Fast bebop
  python generate.py --measures 150 --tempo 160 --output bebop.mid

  # Slow ballad
  python generate.py --measures 150 --tempo 90 --output ballad.mid

  # Use original (non-augmented) dataset
  python generate.py --midi-dir ~/midi_dataset
        """
    )

    parser.add_argument(
        '--midi-dir',
        default='~/augmented_dataset',
        help='Training MIDI directory'
    )
    parser.add_argument(
        '--measures', '-m',
        type=int,
        default=64,
        help='Number of measures (64 ≈ 3 min, 150 ≈ 6 min)'
    )
    parser.add_argument(
        '--tempo', '-t',
        type=int,
        default=120,
        help='Tempo in BPM (90=ballad, 120=standard, 160=bebop)'
    )
    parser.add_argument(
        '--output', '-o',
        default='coltrain_jazz.mid',
        help='Output filename'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=2,
        help='Markov chain order (default: 2)'
    )

    args = parser.parse_args()

    # Create generator
    print("🎷 Coltrain - AI Jazz Generation")
    print("=" * 50)

    coltrain = Coltrain(args.midi_dir, order=args.order)

    # Load and analyze
    if not coltrain.load_and_analyze():
        print("❌ Failed to load training data")
        return

    # Generate
    coltrain.generate(
        num_measures=args.measures,
        tempo=args.tempo,
        output_file=args.output
    )

    print("\n✨ Done!")
    print("\n💡 Tips:")
    print("   • Open in GarageBand/Logic/Ableton")
    print("   • Assign jazz instruments (sax, piano, bass, drums)")
    print("   • Add reverb and compression")
    print("   • Generate multiple variations and pick the best")


if __name__ == '__main__':
    main()
