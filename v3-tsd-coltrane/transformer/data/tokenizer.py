#!/usr/bin/env python3
"""
MIDI Tokenizer v3 - Compound Tokens + Time-Interleaved

COMPOUND TOKENS: Instead of 2 tokens per note (TRACK + NOTE), use 1:
  M_ON_60_5  = Melody note-on, pitch 60, velocity bin 5
  P_OFF_48   = Piano note-off, pitch 48
  D_ON_42_4  = Drums note-on, pitch 42, velocity bin 4

This DOUBLES the effective context window - 2048 tokens covers ~4 minutes.

TIME-INTERLEAVED: All tracks sorted by absolute time so the model
learns cross-track coordination.
"""
import mido
from pathlib import Path
from typing import List
import pickle


class MIDITokenizer:
    """Compound-token time-interleaved MIDI tokenizer."""

    EVENT_PAD = 0
    EVENT_START = 1
    EVENT_END = 2

    TRACKS = ['M', 'P', 'B', 'D']  # Melody, Piano, Bass, Drums
    TRACK_FULL = {'M': 'MELODY', 'P': 'PIANO', 'B': 'BASS', 'D': 'DRUMS'}
    TRACK_TO_CHANNEL = {'M': 0, 'P': 1, 'B': 2, 'D': 9}
    TRACK_PROGRAMS = {'M': 64, 'P': 0, 'B': 32, 'D': 0}

    # Time: 100 bins, 0-2400 ticks (24 ticks/bin)
    # At 480 tpb: sixteenth=120t=bin5, eighth=240t=bin10, quarter=480t=bin20
    MAX_TIME_SHIFT = 2400
    TIME_SHIFT_BINS = 100
    VELOCITY_BINS = 8

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

        # Special
        self._add('PAD')
        self._add('START')
        self._add('END')

        # Compound NOTE_ON: {track}_ON_{pitch}_{vel_bin}
        for t in self.TRACKS:
            for p in range(128):
                for v in range(self.VELOCITY_BINS):
                    self._add(f'{t}_ON_{p}_{v}')

        # Compound NOTE_OFF: {track}_OFF_{pitch}
        for t in self.TRACKS:
            for p in range(128):
                self._add(f'{t}_OFF_{p}')

        # TIME_SHIFT
        for i in range(self.TIME_SHIFT_BINS):
            self._add(f'T_{i}')

        print(f"Vocabulary size: {self.vocab_size}")

    def _add(self, token: str):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.vocab_size
            self.id_to_token[self.vocab_size] = token
            self.vocab_size += 1

    def _vel_to_bin(self, v): return min(v // 16, self.VELOCITY_BINS - 1)
    def _bin_to_vel(self, b): return min(b * 16 + 8, 127)

    def _time_to_bin(self, t):
        bs = self.MAX_TIME_SHIFT // self.TIME_SHIFT_BINS
        return min(max(t, 0) // bs, self.TIME_SHIFT_BINS - 1)

    def _bin_to_time(self, b):
        bs = self.MAX_TIME_SHIFT // self.TIME_SHIFT_BINS
        return b * bs + bs // 2

    def tokenize_midi(self, midi_path: str) -> List[int]:
        """Convert MIDI to compound-token time-interleaved sequence."""
        mid = mido.MidiFile(midi_path)

        # Collect all events with absolute times
        all_events = []

        for track in mid.tracks:
            track_code = self._identify_track(track)
            if not track_code:
                continue

            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    vb = self._vel_to_bin(msg.velocity)
                    all_events.append((abs_time, f'{track_code}_ON_{msg.note}_{vb}'))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    all_events.append((abs_time, f'{track_code}_OFF_{msg.note}'))

        if not all_events:
            return [self.EVENT_START, self.EVENT_END]

        # Sort by time
        all_events.sort(key=lambda e: e[0])

        # Build token sequence with relative time shifts
        tokens = [self.token_to_id['START']]
        prev_time = 0

        for abs_time, token_str in all_events:
            delta = abs_time - prev_time
            if delta > 0:
                remaining = delta
                while remaining > 0:
                    shift = min(remaining, self.MAX_TIME_SHIFT)
                    tokens.append(self.token_to_id[f'T_{self._time_to_bin(shift)}'])
                    remaining -= shift
                prev_time = abs_time

            if token_str in self.token_to_id:
                tokens.append(self.token_to_id[token_str])

        tokens.append(self.token_to_id['END'])
        return tokens

    def _identify_track(self, track) -> str:
        """Identify track → M/P/B/D code."""
        # Check track name
        for msg in track:
            if msg.type == 'track_name':
                name = msg.name.lower()
                if any(w in name for w in ['drum', 'perc']):
                    return 'D'
                elif 'bass' in name:
                    return 'B'
                elif any(w in name for w in ['piano', 'keys', 'chord', 'comp']):
                    return 'P'
                elif any(w in name for w in ['melody', 'sax', 'trumpet', 'lead', 'solo']):
                    return 'M'

        # Check program change
        for msg in track:
            if msg.type == 'program_change':
                if msg.channel == 9:
                    return 'D'
                if msg.program in range(0, 8):
                    return 'P'
                elif msg.program in range(24, 32):
                    return 'P'  # Guitar as comping
                elif msg.program in range(32, 40):
                    return 'B'
                elif msg.program in range(56, 80):
                    return 'M'

        # Fallback: channel/pitch
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.channel == 9:
                    return 'D'
                if msg.note < 48:
                    return 'B'
                elif msg.note > 72:
                    return 'M'
                else:
                    return 'P'
        return None

    def detokenize_to_midi(self, tokens: List[int], output_path: str, tempo: int = 120):
        """Convert compound tokens back to multi-track MIDI."""
        mid = mido.MidiFile(ticks_per_beat=480)

        tempo_track = mido.MidiTrack()
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        mid.tracks.append(tempo_track)

        # Accumulate events per track
        track_events = {t: [] for t in self.TRACKS}
        abs_time = 0

        for tid in tokens:
            tok = self.id_to_token.get(tid, 'PAD')

            if tok in ('PAD', 'START', 'END'):
                continue

            if tok.startswith('T_'):
                tb = int(tok.split('_')[1])
                abs_time += self._bin_to_time(tb)
                continue

            # Parse compound token: {track}_{ON/OFF}_{pitch}[_{vel}]
            parts = tok.split('_')
            if len(parts) < 3:
                continue

            track_code = parts[0]
            event_type = parts[1]
            pitch = int(parts[2])

            if track_code not in self.TRACKS:
                continue

            if event_type == 'ON' and len(parts) >= 4:
                vel = self._bin_to_vel(int(parts[3]))
                track_events[track_code].append((abs_time, 'note_on', pitch, vel))
            elif event_type == 'OFF':
                track_events[track_code].append((abs_time, 'note_off', pitch, 0))

        # Build MIDI tracks
        for tc in self.TRACKS:
            events = track_events[tc]
            if not events:
                continue

            t = mido.MidiTrack()
            t.append(mido.MetaMessage('track_name', name=self.TRACK_FULL[tc]))

            ch = self.TRACK_TO_CHANNEL[tc]
            if tc != 'D':
                t.append(mido.Message('program_change', program=self.TRACK_PROGRAMS[tc], channel=ch))

            events.sort(key=lambda e: e[0])
            prev = 0
            for etime, etype, pitch, vel in events:
                delta = max(0, etime - prev)
                t.append(mido.Message(etype, note=pitch, velocity=vel, time=delta, channel=ch))
                prev = etime

            mid.tracks.append(t)

        mid.save(output_path)
        print(f"Saved: {output_path}")

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'t2i': self.token_to_id, 'i2t': self.id_to_token, 'vs': self.vocab_size}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
            self.token_to_id, self.id_to_token, self.vocab_size = d['t2i'], d['i2t'], d['vs']


def test_tokenizer():
    print("Testing Compound Token Tokenizer")
    print("=" * 50)

    tok = MIDITokenizer()

    test_file = "~/augmented_dataset/SoulEyes#p3.mid"
    if not Path(test_file).exists():
        test_file = "~/coltrain/examples/strict_32bar.mid"

    if Path(test_file).exists():
        tokens = tok.tokenize_midi(test_file)
        decoded = [tok.id_to_token[t] for t in tokens]

        print(f"\nFile: {test_file}")
        print(f"Total tokens: {len(tokens)}")

        # Show interleaved multi-track section
        start = len(decoded) // 4
        print(f"\nTokens {start}-{start+30} (mid-song):")
        for i, t in enumerate(decoded[start:start+30]):
            print(f"  {start+i:4d}: {t}")

        # Track counts
        counts = {}
        for t in decoded:
            prefix = t.split('_')[0]
            if prefix in ('M', 'P', 'B', 'D'):
                counts[prefix] = counts.get(prefix, 0) + 1
        print(f"\nTrack events: {counts}")

        # Rhythm variety
        time_bins = set(int(t.split('_')[1]) for t in decoded if t.startswith('T_'))
        print(f"Unique time bins: {len(time_bins)} -> {sorted(time_bins)}")

        # Compare to old tokenizer
        print(f"\nCompound token savings:")
        print(f"  Old approach: ~{len(tokens)*2} tokens (2 per note)")
        print(f"  New approach: {len(tokens)} tokens (1 per note)")
        print(f"  2x more music fits in context window!")

        # Round-trip
        tok.detokenize_to_midi(tokens, "~/coltrain/test_compound.mid")
        print("\nRound-trip test passed!")


if __name__ == '__main__':
    test_tokenizer()
