#!/usr/bin/env python3
"""
Coltrain v3 — Cascaded 4-model jazz generation.

Generates multi-track jazz MIDI by running 4 specialist models in sequence:
  1. Lead model generates melody → until [COMPING]
  2. Comping model continues → until [BASS]
  3. Bass model continues → until [DRUMS]
  4. Drums model continues → until [EOS]

Each model sees ALL previous tracks as context but only generates its own section.

Usage:
    python generate_v3.py \
        --lead-ckpt checkpoints_v3/v3_lead_best.pt \
        --comping-ckpt checkpoints_v3/v3_comping_best.pt \
        --bass-ckpt checkpoints_v3/v3_bass_best.pt \
        --drums-ckpt checkpoints_v3/v3_drums_best.pt \
        --output generated_v3.mid \
        --temperature 0.9 --top-k 50 --top-p 0.95

    # Single shared model (no specialist fine-tuning)
    python generate_v3.py \
        --shared-ckpt checkpoints_v3/v3_shared_best.pt \
        --output generated_v3.mid
"""
import torch
import torch.nn.functional as F
import argparse
import sys
import os
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import math

sys.path.insert(0, str(Path(__file__).parent / 'transformer'))

from model.transformer import MusicTransformer
from data.tokenizer_v3 import (
    TrackSeqTokenizer, CHORDS_START, LEAD_START, COMPING_START,
    BASS_START, DRUMS_START,
    PITCH_MIN, PITCH_MAX, PITCHDRUM_MIN, PITCHDRUM_MAX,
    TS_MIN, TS_MAX, ROOT_MIN, ROOT_MAX, QUAL_MIN, QUAL_MAX,
    CHORD_TEMPLATES, QUALITY_NAMES,
)

# ── Realistic instrument pitch ranges (token IDs) ──
# Formula: token_id = MIDI_pitch - 18
PITCH_RANGES = {
    'lead_sax':     (31, 62),   # Alto sax: Db3-Ab5 (MIDI 49-80)
    'lead_trumpet': (34, 64),   # Trumpet: E3-Bb5 (MIDI 52-82)
    'lead_piano':   (18, 78),   # Piano melody: C2-C7 (MIDI 36-96)
    'comping':      (18, 66),   # Piano comp: C2-C6 (MIDI 36-84)
    'bass':         (10, 49),   # Acoustic bass: E1-G4 (MIDI 28-67)
}

# Standard GM drum kit tokens (MIDI 35-59 → token IDs 244-268)
# Kick, snare, toms, hi-hat, ride, crash, etc.
JAZZ_DRUM_TOKENS = set(range(244, 269))

# ── Jazz scale mappings (quality_idx -> scale intervals from root) ──
# Used for theory-guided sampling: boost chord/scale tones, penalize avoid notes
JAZZ_SCALES = {
    0:  frozenset({0, 2, 4, 5, 7, 9, 11}),   # maj     -> Ionian
    1:  frozenset({0, 2, 3, 5, 7, 9, 10}),    # min     -> Dorian
    2:  frozenset({0, 2, 4, 5, 7, 9, 10}),    # 7       -> Mixolydian
    3:  frozenset({0, 2, 4, 5, 7, 9, 11}),    # maj7    -> Ionian
    4:  frozenset({0, 2, 3, 5, 7, 9, 10}),    # min7    -> Dorian
    5:  frozenset({0, 2, 3, 5, 6, 8, 9, 11}), # dim     -> Whole-half diminished
    6:  frozenset({0, 2, 4, 6, 8, 10}),        # aug     -> Whole tone
    7:  frozenset({0, 1, 3, 5, 6, 8, 10}),    # m7b5    -> Locrian
    8:  frozenset({0, 2, 3, 5, 6, 8, 9, 11}), # dim7    -> Whole-half diminished
    9:  frozenset({0, 2, 4, 5, 7, 9, 10}),    # sus4    -> Mixolydian
    10: frozenset({0, 2, 4, 5, 7, 9, 10}),    # sus2    -> Mixolydian
    11: frozenset({0, 2, 4, 5, 7, 9, 11}),    # 6       -> Ionian
    12: frozenset({0, 2, 3, 5, 7, 9, 10}),    # min6    -> Dorian
}

# Chord tones (subset of scale tones): the intervals in the chord itself
# Built from CHORD_TEMPLATES: quality_idx -> frozenset of intervals
CHORD_TONE_INTERVALS = {}
for _template, _qual_idx in CHORD_TEMPLATES.items():
    CHORD_TONE_INTERVALS[_qual_idx] = _template


# ── Beat Grid: Structural awareness for jazz generation ──

@dataclass
class BeatInfo:
    """Per-beat metadata for grid-based generation."""
    beat_number: int          # Absolute beat (0, 1, 2, ...)
    time_units: int           # beat_number * 8
    bar_number: int           # 0-indexed
    beat_in_bar: int          # 0=beat1, 1=beat2, 2=beat3, 3=beat4
    section_label: str        # 'A1', 'A2', 'B', 'A3', 'A'
    phrase_position: int      # Beat within 4-bar phrase (0-15)
    is_phrase_boundary: bool  # True on first beat of new 4-bar phrase
    chord_root_pc: int        # Pitch class 0-11
    chord_qual_idx: int       # Quality index 0-12
    next_chord_root_pc: int   # For approach notes (None if last chord)
    next_chord_qual_idx: int  # (None if last chord)
    is_last_beat_before_change: bool  # True when next beat has different chord
    key_center_pc: int = None           # Key center for geometric tonal awareness


class BeatGrid:
    """Temporal scaffold mapping every beat to structural metadata."""

    def __init__(self, beats, total_beats, form_name=''):
        self.beats = beats
        self.total_beats = total_beats
        self.total_time_units = total_beats * 8
        self.form_name = form_name
        self._by_time = {b.time_units: b for b in beats}

    def at_time(self, time_units):
        """O(1) lookup: find BeatInfo at or just before time_units."""
        beat_time = (time_units // 8) * 8
        beat_time = min(beat_time, self.beats[-1].time_units)
        return self._by_time.get(beat_time, self.beats[-1])


def _fill_next_chord_info(beats):
    """Post-pass: fill next_chord and is_last_beat_before_change fields."""
    for i in range(len(beats)):
        b = beats[i]
        # Find next different chord
        next_root = None
        next_qual = None
        is_last = False
        if i + 1 < len(beats):
            nb = beats[i + 1]
            if nb.chord_root_pc != b.chord_root_pc or nb.chord_qual_idx != b.chord_qual_idx:
                is_last = True
            # Scan forward for next chord change
            for j in range(i + 1, len(beats)):
                if beats[j].chord_root_pc != b.chord_root_pc or beats[j].chord_qual_idx != b.chord_qual_idx:
                    next_root = beats[j].chord_root_pc
                    next_qual = beats[j].chord_qual_idx
                    break
        b.next_chord_root_pc = next_root
        b.next_chord_qual_idx = next_qual
        b.is_last_beat_before_change = is_last


@dataclass
class FormTemplate:
    """Jazz form definition with transposable chord progressions."""
    name: str
    bars: int
    sections: list    # [(label, start_bar, end_bar), ...]
    changes: list     # [(bar, beat, root_offset, qual_name), ...]
    key_center_offsets: list = None  # Parallel to changes: key center offset per chord

    def transpose(self, key_pc):
        """Return changes transposed to key_pc as (bar, beat, root_pc, qual_idx)."""
        qual_map = {name: i for i, name in enumerate(QUALITY_NAMES)}
        qual_map.update({
            'M7': qual_map['maj7'], 'm7': qual_map['min7'],
            'm': qual_map['min'], 'M': qual_map['maj'],
            'dom7': qual_map['7'],
        })
        result = []
        for bar, beat, offset, qual_name in self.changes:
            root_pc = (key_pc + offset) % 12
            qual_idx = qual_map.get(qual_name, 0)
            result.append((bar, beat, root_pc, qual_idx))
        return result

    def _section_for_bar(self, bar):
        """Return section label for a given bar number."""
        for label, start, end in self.sections:
            if start <= bar < end:
                return label
        return '?'

    def _chord_at(self, bar, beat_in_bar, transposed_changes):
        """Find the active chord at (bar, beat_in_bar) from transposed changes."""
        active_root, active_qual = transposed_changes[0][2], transposed_changes[0][3]
        for c_bar, c_beat, c_root, c_qual in transposed_changes:
            if (c_bar, c_beat) <= (bar, beat_in_bar):
                active_root, active_qual = c_root, c_qual
            else:
                break
        return active_root, active_qual

    def _kc_at(self, bar, beat_in_bar, kc_map):
        """Find the active key center at (bar, beat_in_bar)."""
        active_kc = kc_map[0][2]
        for c_bar, c_beat, kc_pc in kc_map:
            if (c_bar, c_beat) <= (bar, beat_in_bar):
                active_kc = kc_pc
            else:
                break
        return active_kc

    def build_grid(self, key_pc, num_choruses=1):
        """Construct the full BeatGrid for this form in the given key."""
        transposed = self.transpose(key_pc)
        # Key center map: transpose offsets to absolute pitch classes
        kc_map = None
        if self.key_center_offsets:
            kc_map = [(self.changes[i][0], self.changes[i][1],
                        (key_pc + self.key_center_offsets[i]) % 12)
                       for i in range(len(self.key_center_offsets))]
        all_beats = []

        for chorus in range(num_choruses):
            bar_offset = chorus * self.bars
            beat_offset = chorus * self.bars * 4

            for bar in range(self.bars):
                abs_bar = bar_offset + bar
                section_label = self._section_for_bar(bar)

                for beat_in_bar in range(4):
                    abs_beat = beat_offset + bar * 4 + beat_in_bar
                    time_units = abs_beat * 8
                    root_pc, qual_idx = self._chord_at(bar, beat_in_bar, transposed)
                    phrase_position = abs_beat % 16
                    key_center = self._kc_at(bar, beat_in_bar, kc_map) if kc_map else key_pc

                    all_beats.append(BeatInfo(
                        beat_number=abs_beat,
                        time_units=time_units,
                        bar_number=abs_bar,
                        beat_in_bar=beat_in_bar,
                        section_label=section_label,
                        phrase_position=phrase_position,
                        is_phrase_boundary=(phrase_position == 0),
                        chord_root_pc=root_pc,
                        chord_qual_idx=qual_idx,
                        next_chord_root_pc=None,
                        next_chord_qual_idx=None,
                        is_last_beat_before_change=False,
                        key_center_pc=key_center,
                    ))

        _fill_next_chord_info(all_beats)
        total_beats = num_choruses * self.bars * 4
        return BeatGrid(all_beats, total_beats,
                        f'{self.name}x{num_choruses}' if num_choruses > 1 else self.name)


# ── Predefined jazz form templates ──
# Chord changes: (bar, beat, root_offset_from_key, quality_name)
# root_offset: 0=I, 2=ii, 4=iii, 5=IV, 7=V, 9=vi, 11=vii

FORM_TEMPLATES = {
    'blues12': FormTemplate(
        name='blues12',
        bars=12,
        sections=[('A', 0, 4), ('B', 4, 8), ('A', 8, 12)],
        changes=[
            (0,  0, 0,  '7'),    # I7
            (4,  0, 5,  '7'),    # IV7
            (6,  0, 0,  '7'),    # I7
            (8,  0, 7,  '7'),    # V7
            (9,  0, 5,  '7'),    # IV7
            (10, 0, 0,  '7'),    # I7
            (11, 0, 7,  '7'),    # V7 (turnaround)
        ],
    ),
    'aaba32': FormTemplate(
        name='aaba32',
        bars=32,
        sections=[('A1', 0, 8), ('A2', 8, 16), ('B', 16, 24), ('A3', 24, 32)],
        changes=[
            # A1 section (bars 0-7): ii-V-I-VI | ii-V-I-I
            (0,  0, 2,  'min7'),   # ii (Dm7 in C)
            (1,  0, 7,  '7'),      # V  (G7)
            (2,  0, 0,  'maj7'),   # I  (Cmaj7)
            (3,  0, 9,  '7'),      # VI (A7)
            (4,  0, 2,  'min7'),   # ii
            (5,  0, 7,  '7'),      # V
            (6,  0, 0,  'maj7'),   # I
            (7,  0, 0,  'maj7'),   # I
            # A2 section (bars 8-15): same changes
            (8,  0, 2,  'min7'),
            (9,  0, 7,  '7'),
            (10, 0, 0,  'maj7'),
            (11, 0, 9,  '7'),
            (12, 0, 2,  'min7'),
            (13, 0, 7,  '7'),
            (14, 0, 0,  'maj7'),
            (15, 0, 0,  'maj7'),
            # B section (bars 16-23): bridge - modulate to IV
            (16, 0, 5,  'maj7'),   # IV  (Fmaj7)
            (17, 0, 5,  'maj7'),   # IV
            (18, 0, 0,  '7'),      # I7  (C7 - secondary dominant)
            (19, 0, 0,  '7'),      # I7
            (20, 0, 9,  'min7'),   # vi  (Am7)
            (21, 0, 2,  'min7'),   # ii
            (22, 0, 7,  '7'),      # V
            (23, 0, 7,  '7'),      # V
            # A3 section (bars 24-31): same as A1
            (24, 0, 2,  'min7'),
            (25, 0, 7,  '7'),
            (26, 0, 0,  'maj7'),
            (27, 0, 9,  '7'),
            (28, 0, 2,  'min7'),
            (29, 0, 7,  '7'),
            (30, 0, 0,  'maj7'),
            (31, 0, 0,  'maj7'),
        ],
    ),
    # ── Coltrane's Giant Steps: 3 key centers separated by major thirds ──
    # Key centers at offsets 0, 8, 4 (e.g. B, G, Eb when --key B)
    # Each tonic approached by its V7; harmonic rhythm = 2 beats per chord
    'giantsteps': FormTemplate(
        name='giantsteps',
        bars=16,
        sections=[('A1', 0, 4), ('A2', 4, 8), ('B1', 8, 12), ('B2', 12, 16)],
        changes=[
            # A1: I → bIII-key → bVI-key → bIII-key
            (0, 0, 0, 'maj7'),   (0, 2, 3, '7'),       # Bmaj7  D7
            (1, 0, 8, 'maj7'),   (1, 2, 11, '7'),      # Gmaj7  Bb7
            (2, 0, 4, 'maj7'),                          # Ebmaj7
            (3, 0, 10, 'min7'),  (3, 2, 3, '7'),       # Am7    D7
            # A2: bIII-key → bVI-key → I → bVI-key
            (4, 0, 8, 'maj7'),   (4, 2, 11, '7'),      # Gmaj7  Bb7
            (5, 0, 4, 'maj7'),   (5, 2, 7, '7'),       # Ebmaj7 F#7
            (6, 0, 0, 'maj7'),                          # Bmaj7
            (7, 0, 6, 'min7'),   (7, 2, 11, '7'),      # Fm7    Bb7
            # B1: bVI-key → bIII-key → bIII-key → I
            (8, 0, 4, 'maj7'),                          # Ebmaj7
            (9, 0, 10, 'min7'),  (9, 2, 3, '7'),       # Am7    D7
            (10, 0, 8, 'maj7'),                         # Gmaj7
            (11, 0, 2, 'min7'),  (11, 2, 7, '7'),      # C#m7   F#7
            # B2: I → bVI-key → bVI-key → I (turnaround)
            (12, 0, 0, 'maj7'),                         # Bmaj7
            (13, 0, 6, 'min7'),  (13, 2, 11, '7'),     # Fm7    Bb7
            (14, 0, 4, 'maj7'),                         # Ebmaj7
            (15, 0, 2, 'min7'),  (15, 2, 7, '7'),      # C#m7   F#7
        ],
        # Key center offsets: 0=I(B), 8=bIII(G), 4=bVI(Eb) — the 3 major thirds
        key_center_offsets=[
            0, 8,    # Bar 0:  Bmaj7→B,   D7→G
            8, 4,    # Bar 1:  Gmaj7→G,   Bb7→Eb
            4,       # Bar 2:  Ebmaj7→Eb
            8, 8,    # Bar 3:  Am7→G,     D7→G
            8, 4,    # Bar 4:  Gmaj7→G,   Bb7→Eb
            4, 0,    # Bar 5:  Ebmaj7→Eb, F#7→B
            0,       # Bar 6:  Bmaj7→B
            4, 4,    # Bar 7:  Fm7→Eb,    Bb7→Eb
            4,       # Bar 8:  Ebmaj7→Eb
            8, 8,    # Bar 9:  Am7→G,     D7→G
            8,       # Bar 10: Gmaj7→G
            0, 0,    # Bar 11: C#m7→B,    F#7→B
            0,       # Bar 12: Bmaj7→B
            4, 4,    # Bar 13: Fm7→Eb,    Bb7→Eb
            4,       # Bar 14: Ebmaj7→Eb
            0, 0,    # Bar 15: C#m7→B,    F#7→B
        ],
    ),
}


def parse_chord_context(tokens, ts_id_to_units):
    """Extract chord progression from chord section tokens in a sequence.

    Scans tokens between CHORDS_START and LEAD_START for Root/Quality/TimeShift.

    Returns:
        list of (time_units, root_pc, qual_idx) tuples, sorted by time
    """
    chords = []
    time = 0
    in_chords = False
    i = 0

    while i < len(tokens):
        tid = tokens[i]

        if tid == CHORDS_START:
            in_chords = True
            i += 1
            continue
        if tid == LEAD_START:
            break
        if not in_chords:
            i += 1
            continue

        if ROOT_MIN <= tid <= ROOT_MAX:
            root = tid - ROOT_MIN
            qual = 0
            if i + 1 < len(tokens) and QUAL_MIN <= tokens[i + 1] <= QUAL_MAX:
                qual = tokens[i + 1] - QUAL_MIN
                i += 1
            chords.append((time, root, qual))
            i += 1
        elif TS_MIN <= tid <= TS_MAX:
            time += ts_id_to_units[tid]
            i += 1
        else:
            i += 1

    return chords


def build_theory_masks(root_pc, qual_idx, vocab_size, device):
    """Build chord-tone, scale-tone, and avoid-note masks for a given chord.

    Returns:
        (chord_tone_mask, scale_tone_mask, avoid_mask) — each a bool tensor of shape (vocab_size,)
    """
    chord_intervals = CHORD_TONE_INTERVALS.get(qual_idx, frozenset({0, 4, 7}))
    scale_intervals = JAZZ_SCALES.get(qual_idx, frozenset({0, 2, 4, 5, 7, 9, 11}))

    chord_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    scale_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    avoid_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    for t in range(PITCH_MIN, PITCH_MAX + 1):
        pc = (t + 18) % 12
        interval = (pc - root_pc) % 12
        if interval in chord_intervals:
            chord_mask[t] = True
        elif interval in scale_intervals:
            scale_mask[t] = True
        else:
            avoid_mask[t] = True

    return chord_mask, scale_mask, avoid_mask


# ── Grid helper functions ──

def build_grid_from_chords(chord_context, total_beats=None):
    """Build a BeatGrid from a raw chord_context list (no form template).

    Args:
        chord_context: list of (time_units, root_pc, qual_idx)
        total_beats: override total beats; otherwise inferred from last chord + 8 beats
    """
    if not chord_context:
        return None

    if total_beats is None:
        last_time = chord_context[-1][0]
        total_beats = last_time // 8 + 8  # Add 8 beats after last chord

    beats = []
    for abs_beat in range(total_beats):
        time_units = abs_beat * 8
        # Find active chord at this time
        root_pc, qual_idx = chord_context[0][1], chord_context[0][2]
        for ct, cr, cq in chord_context:
            if ct <= time_units:
                root_pc, qual_idx = cr, cq
            else:
                break

        beats.append(BeatInfo(
            beat_number=abs_beat,
            time_units=time_units,
            bar_number=abs_beat // 4,
            beat_in_bar=abs_beat % 4,
            section_label='A',
            phrase_position=abs_beat % 16,
            is_phrase_boundary=(abs_beat % 16 == 0),
            chord_root_pc=root_pc,
            chord_qual_idx=qual_idx,
            next_chord_root_pc=None,
            next_chord_qual_idx=None,
            is_last_beat_before_change=False,
            key_center_pc=infer_key_center(root_pc, qual_idx),
        ))

    _fill_next_chord_info(beats)
    return BeatGrid(beats, total_beats, 'chords')


def beat_grid_to_chord_tokens(grid, tokenizer):
    """Convert a BeatGrid's chord progression into chord section tokens."""
    tokens = []
    prev_chord = None
    chord_start_time = 0

    for beat in grid.beats:
        chord = (beat.chord_root_pc, beat.chord_qual_idx)
        if chord != prev_chord:
            if prev_chord is not None:
                delta = beat.time_units - chord_start_time
                if delta > 0:
                    tokens.extend(tokenizer._delta_to_ts_tokens(delta))
            tokens.append(ROOT_MIN + beat.chord_root_pc)
            tokens.append(QUAL_MIN + beat.chord_qual_idx)
            chord_start_time = beat.time_units
            prev_chord = chord

    # Final chord duration
    if prev_chord is not None:
        delta = grid.total_time_units - chord_start_time
        if delta > 0:
            tokens.extend(tokenizer._delta_to_ts_tokens(delta))

    return tokens


# ── Coltrane features: Key centers, Motivic memory, Tension curves ──

def infer_key_center(root_pc, qual_idx):
    """Infer the tonal key center for a chord based on its quality.

    Heuristic mapping from chord function to likely key center:
    - Major/maj7/6 → tonic, key center = root
    - min/min7/min6 → likely ii, key center = root - 2 semitones
    - dom7/sus → likely V, key center = root - 7 (down a fifth)
    - m7b5/dim → likely vii, key center = root + 1 semitone
    """
    if qual_idx in (0, 3, 11):      # maj, maj7, 6
        return root_pc
    elif qual_idx in (1, 4, 12):    # min, min7, min6
        return (root_pc + 10) % 12  # ii → I (down whole step)
    elif qual_idx in (2, 9, 10):    # 7, sus4, sus2
        return (root_pc + 5) % 12   # V → I (down a 5th)
    elif qual_idx in (7, 5, 8):     # m7b5, dim, dim7
        return (root_pc + 1) % 12   # vii → I (up half step)
    else:                           # aug, other
        return root_pc


class MotivicMemory:
    """Tracks pitch intervals from the first phrase for motivic development.

    Emulates Coltrane's approach: take a tiny musical cell (4-8 notes)
    and develop it across the solo — transposing to new key centers,
    inverting, varying rhythm while keeping the interval DNA recognizable.

    During the first 4-bar phrase, we "collect" the interval pattern.
    During subsequent phrases, we boost tokens that would continue
    a recognized interval match from the stored motif.
    """

    def __init__(self, motif_notes=12, boost=1.5):
        self.motif_intervals = []      # Intervals from first phrase
        self.recent_pitches = []       # All pitches (MIDI) generated
        self.collecting = True         # True during first phrase
        self.max_motif_notes = motif_notes
        self.boost = boost

    def add_note(self, pitch_token):
        """Record a note pitch. During collection, builds the motif."""
        midi = pitch_token + 18
        if self.recent_pitches:
            interval = midi - self.recent_pitches[-1]
            if self.collecting and len(self.motif_intervals) < self.max_motif_notes:
                self.motif_intervals.append(interval)
        self.recent_pitches.append(midi)

    def stop_collecting(self):
        """Called when first phrase ends. Freezes the motif."""
        self.collecting = False

    def get_boost_tensor(self, vocab_size, device):
        """Compute logit boosts based on motivic pattern matching.

        Checks if the recent pitch intervals match any sub-pattern of the
        stored motif. If the last K intervals match the start of a motif
        fragment, boost the token that would produce the next interval.
        Also matches transposed echoes (same interval sequence at any
        starting point in the motif).
        """
        if not self.motif_intervals or len(self.recent_pitches) < 2 or self.collecting:
            return None

        boost = torch.zeros(vocab_size, device=device)
        last_midi = self.recent_pitches[-1]

        # Recent intervals (last N notes)
        window = min(len(self.recent_pitches) - 1, len(self.motif_intervals))
        recent_iv = []
        for i in range(len(self.recent_pitches) - window, len(self.recent_pitches) - 1):
            if i >= 0:
                recent_iv.append(self.recent_pitches[i + 1] - self.recent_pitches[i])

        if not recent_iv:
            return None

        best_boost = 0.0
        best_target = None

        # Match tail of recent intervals against any sub-sequence of the motif
        for motif_start in range(len(self.motif_intervals)):
            remaining = self.motif_intervals[motif_start:]
            if len(remaining) < 2:
                break

            for match_len in range(min(len(recent_iv), len(remaining) - 1), 0, -1):
                tail = recent_iv[-match_len:]
                motif_sub = remaining[:match_len]

                if tail == motif_sub and motif_start + match_len < len(self.motif_intervals):
                    next_iv = self.motif_intervals[motif_start + match_len]
                    target_midi = last_midi + next_iv
                    target_token = target_midi - 18

                    this_boost = self.boost * (0.5 + match_len * 0.5)
                    if this_boost > best_boost and PITCH_MIN <= target_token <= PITCH_MAX:
                        best_boost = this_boost
                        best_target = target_token
                    break  # Longest match at this motif_start

        if best_target is not None:
            boost[best_target] = best_boost
            # Octave transpositions (weaker echo)
            for shift in [-12, 12]:
                alt = best_target + shift
                if PITCH_MIN <= alt <= PITCH_MAX:
                    boost[alt] = best_boost * 0.4

        return boost if boost.any() else None


class TensionCurve:
    """Parameterized tension arc modulating generation across a piece.

    Models Coltrane's solo architecture:
    - Exposition (0-25%):  Lyrical, consonant, moderate pace
    - Development (25-50%): Adventurous, chromatic exploration
    - Climax (50-85%):     Sheets of sound — rapid chord-tone arpeggiation
    - Resolution (85-100%): Return to melody, strong harmonic resolution

    The curve modulates chord_tone_boost, avoid_penalty, note speed,
    and arpeggio intensity to create a natural narrative shape.
    """

    def __init__(self, total_time_units, arpeggio_boost=2.0):
        self.total_time = max(total_time_units, 1)
        self.arpeggio_boost = arpeggio_boost

    def get_modulation(self, current_time):
        """Return modulation dict based on position in the piece.

        Returns:
            chord_tone_boost_mult: multiplier for chord tone logit boost
            avoid_penalty_mult: multiplier for avoid note penalty
            short_duration_boost: logit boost for short TimeShift tokens
            arpeggio_boost: extra chord-tone boost (sheets of sound effect)
            resolution_boost: extra chord-tone boost near ending
        """
        progress = min(current_time / self.total_time, 1.0)

        if progress < 0.25:
            # Exposition: consonant, lyrical
            ct_mult = 1.2
            avoid_mult = 1.2
            short_boost = 0.0
            arp_boost = 0.0
        elif progress < 0.50:
            # Development: chromatic exploration
            t = (progress - 0.25) / 0.25
            ct_mult = 1.2 - 0.5 * t      # 1.2 → 0.7
            avoid_mult = 1.2 - 0.7 * t    # 1.2 → 0.5
            short_boost = 0.3 * t          # 0 → 0.3
            arp_boost = 0.0
        elif progress < 0.85:
            # Climax: sheets of sound — fast chord-tone arpeggiation
            intensity = min((progress - 0.50) / 0.15, 1.0)  # ramp to full by 0.65
            ct_mult = 1.4                  # Strong chord tones
            avoid_mult = 1.3               # Stay harmonic
            short_boost = 1.0              # Fast movement
            arp_boost = self.arpeggio_boost * intensity
        else:
            # Resolution: consonant, slower, resolving
            ct_mult = 1.3
            avoid_mult = 1.0
            short_boost = -0.3             # Slow down
            arp_boost = 0.0

        resolution = max(0, (progress - 0.85) / 0.15) * 2.5 if progress > 0.85 else 0.0

        return {
            'chord_tone_boost_mult': ct_mult,
            'avoid_penalty_mult': avoid_mult,
            'short_duration_boost': short_boost,
            'arpeggio_boost': arp_boost,
            'resolution_boost': resolution,
        }


def apply_key_center_bias(logits, key_center_pc, boost=0.5):
    """Boost notes from the key center's major scale.

    Captures Coltrane's geometric thinking: he navigated key centers
    (B, G, Eb in Giant Steps), not individual chords. The key center's
    scale provides broader tonal context beyond the current chord.
    """
    major_scale = {0, 2, 4, 5, 7, 9, 11}
    for t in range(PITCH_MIN, PITCH_MAX + 1):
        pc = (t + 18) % 12
        interval = (pc - key_center_pc) % 12
        if interval in major_scale:
            logits[0, t] += boost


# ── Beat-position bias functions ──

def apply_beat_position_bias(logits, beat_info, role_name, params, device):
    """Apply role-specific logit biases based on beat position in the bar.

    Args:
        logits: shape (1, vocab_size), already temperature-scaled
        beat_info: BeatInfo for current position
        role_name: 'lead', 'comping', 'bass', 'drums'
        params: dict with 'bass_root_boost', 'phrase_breathing', etc.
        device: torch device
    """
    bib = beat_info.beat_in_bar  # 0, 1, 2, 3

    if role_name == 'bass':
        root_pc = beat_info.chord_root_pc
        # Bass root in octave 2: token = pitch_class + 18
        root_token = root_pc + 18
        fifth_pc = (root_pc + 7) % 12
        fifth_token = fifth_pc + 18
        third_pc = beat_info.chord_qual_idx  # Need to compute 3rd from chord
        # Major 3rd = +4, minor 3rd = +3
        chord_intervals = CHORD_TONE_INTERVALS.get(beat_info.chord_qual_idx, frozenset({0, 4, 7}))
        third_interval = 4 if 4 in chord_intervals else 3
        third_token = ((root_pc + third_interval) % 12) + 18
        seventh_interval = 11 if 11 in chord_intervals else 10
        seventh_token = ((root_pc + seventh_interval) % 12) + 18

        bass_root_boost = params.get('bass_root_boost', 3.0)

        if bib == 0:  # Beat 1: strong root
            logits[0, root_token] += bass_root_boost
            # Also boost octave above
            if root_token + 12 <= PITCH_MAX:
                logits[0, root_token + 12] += bass_root_boost * 0.5
        elif bib == 1:  # Beat 2: 5th
            logits[0, fifth_token] += bass_root_boost * 0.67
            if fifth_token + 12 <= PITCH_MAX:
                logits[0, fifth_token + 12] += bass_root_boost * 0.33
        elif bib == 2:  # Beat 3: 3rd
            logits[0, third_token] += bass_root_boost * 0.5
            if third_token + 12 <= PITCH_MAX:
                logits[0, third_token + 12] += bass_root_boost * 0.25
        elif bib == 3:  # Beat 4: 7th (passing tone)
            logits[0, seventh_token] += bass_root_boost * 0.33

    elif role_name == 'lead':
        phrase_breathing = params.get('phrase_breathing', 1.5)

        # Phrase breathing at phrase boundaries
        if beat_info.is_phrase_boundary and bib == 0 and phrase_breathing > 0:
            for ts_id in range(TS_MIN, TS_MAX + 1):
                logits[0, ts_id] += phrase_breathing
            # Slightly discourage immediate note attacks
            for t in range(PITCH_MIN, PITCH_MAX + 1):
                logits[0, t] -= phrase_breathing * 0.33

        # Chord-tone resolution at phrase endings (last 2 beats of 4-bar phrase)
        if beat_info.phrase_position >= 14:
            ct_mask, _, _ = build_theory_masks(
                beat_info.chord_root_pc, beat_info.chord_qual_idx,
                logits.size(-1), device)
            logits[0, ct_mask] += 2.0

    elif role_name == 'comping':
        # Subtle backbeat emphasis on beats 2 and 4
        if bib in (1, 3):
            for t in range(PITCH_MIN, PITCH_MAX + 1):
                logits[0, t] += 0.3


def apply_approach_note_bias(logits, beat_info, role_name, params):
    """Boost half-step neighbors of the next chord's root before chord changes.

    Creates the chromatic approach that jazz musicians use to smoothly
    transition between chords.
    """
    if not beat_info.is_last_beat_before_change:
        return
    if beat_info.next_chord_root_pc is None:
        return
    if role_name == 'drums':
        return

    approach_boost = params.get('approach_boost', 1.5)
    if approach_boost <= 0:
        return

    next_root = beat_info.next_chord_root_pc
    below_pc = (next_root - 1) % 12  # Half step below
    above_pc = (next_root + 1) % 12  # Half step above

    if role_name == 'bass':
        # Approach in octave 2 only
        logits[0, below_pc + 18] += approach_boost
        logits[0, above_pc + 18] += approach_boost
    else:
        # Melodic approach across all octaves in range
        for t in range(PITCH_MIN, PITCH_MAX + 1):
            pc = (t + 18) % 12
            if pc == below_pc or pc == above_pc:
                logits[0, t] += approach_boost


def load_model(checkpoint_path, vocab_size, device='cpu'):
    """Load a MusicTransformer from a v3 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded {checkpoint_path} (epoch {ckpt["epoch"]}, '
          f'val_loss={ckpt["val_loss"]:.4f}, role={ckpt.get("role", "shared")})')
    return model


@torch.no_grad()
def generate_section(model, context, stop_token, max_tokens=400, min_tokens=50,
                     temperature=0.9, top_k=50, top_p=0.95,
                     repetition_penalty=1.2, ngram_block=4,
                     pitch_range=None, allowed_drum_tokens=None,
                     chord_context=None, theory_params=None,
                     beat_grid=None, role_name=None, grid_params=None,
                     tension_curve=None, motivic_memory=None,
                     coltrane_params=None,
                     ts_id_to_units=None,
                     max_seq_len=1024, device='cpu'):
    """Generate tokens for a single role section.

    Args:
        model: The specialist (or shared) model
        context: List of token IDs generated so far (includes [BOS], previous sections)
        stop_token: The delimiter token ID that ends this section
        max_tokens: Maximum tokens to generate for this section
        min_tokens: Minimum tokens before stop_token is allowed
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        repetition_penalty: Divide logits of recent tokens by this value
        ngram_block: Block n-grams of this length from repeating
        pitch_range: (min_token, max_token) for allowed Pitch tokens, or None
        allowed_drum_tokens: set of allowed PitchDrum token IDs, or None
        chord_context: list of (time_units, root_pc, qual_idx) for theory-guided sampling
        theory_params: dict with 'chord_tone_boost', 'scale_tone_boost', 'avoid_penalty'
                       or None to disable theory-guided sampling
        beat_grid: BeatGrid for structural awareness, or None
        role_name: 'lead', 'comping', 'bass', 'drums' — needed for grid biases
        grid_params: dict with 'bass_root_boost', 'phrase_breathing', 'approach_boost'
        ts_id_to_units: dict mapping TimeShift token ID -> time units (for time tracking)
        max_seq_len: Model's maximum context window
        device: Device for inference

    Returns:
        List of generated token IDs (not including context, but including stop_token)
    """
    generated = []
    recent_tokens = []  # Track recent tokens for repetition penalty

    # Pitch mask will be built on first step (needs vocab size from logits)
    pitch_mask = None
    need_pitch_mask = pitch_range is not None or allowed_drum_tokens is not None

    # Theory-guided sampling state
    use_theory = (chord_context is not None and theory_params is not None
                  and ts_id_to_units is not None and len(chord_context) > 0)

    # Beat grid state
    use_grid = beat_grid is not None and role_name is not None and grid_params is not None
    target_time = beat_grid.total_time_units if use_grid else None

    # Time tracking (needed for theory, grid, and tension)
    current_time = 0
    current_chord_idx = 0
    theory_masks = None  # (chord_tone_mask, scale_tone_mask, avoid_mask)
    track_time = use_theory or use_grid or tension_curve is not None

    for step in range(max_tokens):
        # Sliding window: use last max_seq_len tokens
        window = (context + generated)[-max_seq_len:]
        x = torch.tensor([window], dtype=torch.long, device=device)

        logits = model(x)
        logits = logits[:, -1, :] / temperature

        # ── Build pitch mask on first step ──
        if need_pitch_mask and pitch_mask is None:
            pitch_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=device)
            if pitch_range is not None:
                lo, hi = pitch_range
                for t in range(PITCH_MIN, PITCH_MAX + 1):
                    if t < lo or t > hi:
                        pitch_mask[t] = True
            if allowed_drum_tokens is not None:
                for t in range(PITCHDRUM_MIN, PITCHDRUM_MAX + 1):
                    if t not in allowed_drum_tokens:
                        pitch_mask[t] = True

        # ── Pitch range clamping ──
        if pitch_mask is not None:
            logits[0, pitch_mask] = float('-inf')

        # ── Theory-guided sampling: boost chord/scale tones ──
        if use_theory:
            # Update chord if time has advanced past current chord
            while (current_chord_idx + 1 < len(chord_context)
                   and current_time >= chord_context[current_chord_idx + 1][0]):
                current_chord_idx += 1

            # Build/rebuild masks when chord changes
            _, root_pc, qual_idx = chord_context[current_chord_idx]
            if theory_masks is None or theory_masks[3] != (root_pc, qual_idx):
                ct_mask, st_mask, av_mask = build_theory_masks(
                    root_pc, qual_idx, logits.size(-1), device)
                theory_masks = (ct_mask, st_mask, av_mask, (root_pc, qual_idx))

            ct_mask, st_mask, av_mask, _ = theory_masks

            # Apply with tension curve modulation if active
            effective_ct = theory_params['chord_tone_boost']
            effective_avoid = theory_params['avoid_penalty']
            if tension_curve:
                mod = tension_curve.get_modulation(current_time)
                effective_ct *= mod['chord_tone_boost_mult']
                effective_ct += mod['arpeggio_boost']    # Sheets of sound
                effective_ct += mod['resolution_boost']  # Strong ending
                effective_avoid *= mod['avoid_penalty_mult']

            logits[0, ct_mask] += effective_ct
            logits[0, st_mask] += theory_params['scale_tone_boost']
            logits[0, av_mask] -= effective_avoid

        # ── Tension curve: short duration boost (sheets of sound velocity) ──
        if tension_curve and ts_id_to_units:
            mod = tension_curve.get_modulation(current_time)
            sdb = mod['short_duration_boost']
            if abs(sdb) > 0.01:
                for ts_id, units in ts_id_to_units.items():
                    if units <= 4:       # Short: ≤ half a beat
                        logits[0, ts_id] += sdb
                    elif units >= 16:    # Long: ≥ 2 beats
                        logits[0, ts_id] -= sdb * 0.5

        # ── Beat grid: position-dependent biases ──
        beat_info = None
        if beat_grid:
            beat_info = beat_grid.at_time(current_time)
        if use_grid and beat_info:
            apply_beat_position_bias(logits, beat_info, role_name, grid_params, device)
            apply_approach_note_bias(logits, beat_info, role_name, grid_params)

        # ── Key center awareness (Coltrane's geometric thinking) ──
        if coltrane_params and coltrane_params.get('key_center_boost', 0) > 0:
            kc_pc = None
            if beat_info and beat_info.key_center_pc is not None:
                kc_pc = beat_info.key_center_pc
            elif use_theory:
                _, root_pc, qual_idx = chord_context[current_chord_idx]
                kc_pc = infer_key_center(root_pc, qual_idx)
            if kc_pc is not None:
                apply_key_center_bias(logits, kc_pc, coltrane_params['key_center_boost'])

        # ── Motivic memory: boost tokens continuing motif patterns ──
        if motivic_memory and not motivic_memory.collecting:
            motif_boost = motivic_memory.get_boost_tensor(logits.size(-1), device)
            if motif_boost is not None:
                logits += motif_boost.unsqueeze(0)

        # ── Repetition penalty ──
        if repetition_penalty > 1.0 and recent_tokens:
            token_counts = Counter(recent_tokens[-100:])
            for tok, count in token_counts.items():
                logits[0, tok] /= repetition_penalty ** min(count, 3)

        # ── N-gram blocking ──
        if ngram_block > 0 and len(recent_tokens) >= ngram_block:
            prefix = tuple(recent_tokens[-(ngram_block - 1):])
            for i in range(len(recent_tokens) - ngram_block):
                candidate = tuple(recent_tokens[i:i + ngram_block - 1])
                if candidate == prefix:
                    blocked = recent_tokens[i + ngram_block - 1]
                    logits[0, blocked] -= 5.0

        # ── Duration-based termination (grid) ──
        time_done = use_grid and current_time >= target_time
        if time_done:
            logits[0, stop_token] += 10.0
        elif step < min_tokens:
            logits[0, stop_token] = float('-inf')

        # ── Top-k filtering ──
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = float('-inf')

        # ── Top-p (nucleus) filtering ──
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = 0
            logits[0, sorted_indices[0][remove[0]]] = float('-inf')

        # ── Sample ──
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        recent_tokens.append(next_token)

        # Track time for theory/grid/tension
        if track_time and TS_MIN <= next_token <= TS_MAX:
            current_time += ts_id_to_units[next_token]

        # Update motivic memory with new note
        if motivic_memory and PITCH_MIN <= next_token <= PITCH_MAX:
            motivic_memory.add_note(next_token)
            # Stop collecting after first 4-bar phrase (16 beats = 128 time units)
            if motivic_memory.collecting and current_time >= 128:
                motivic_memory.stop_collecting()

        # Stop if we generated the stop token
        if next_token == stop_token:
            break

    # If we hit max_tokens without emitting stop_token, append it
    if not generated or generated[-1] != stop_token:
        generated.append(stop_token)

    return generated


def generate_cascaded(models, tokenizer, temperature=0.9, top_k=50, top_p=0.95,
                      repetition_penalty=1.2, ngram_block=4,
                      role_settings=None, lead_instrument='sax',
                      chord_tokens=None, theory_params=None,
                      beat_grid=None, grid_params=None,
                      coltrane_params=None, device='cpu'):
    """Generate a full multi-track jazz piece using cascaded specialist models.

    Args:
        models: dict with keys 'lead', 'comping', 'bass', 'drums'
                (can all be the same shared model)
        tokenizer: TrackSeqTokenizer instance
        role_settings: dict of per-role overrides
        lead_instrument: 'sax', 'trumpet', or 'piano' — determines pitch range
        theory_params: dict with chord_tone_boost, scale_tone_boost, avoid_penalty
        beat_grid: BeatGrid for form-aware generation, or None
        grid_params: dict with bass_root_boost, phrase_breathing, approach_boost
        coltrane_params: dict with tension_curve, motivic_memory, key_center_boost,
                         arpeggio_boost, motif_boost — or None to disable

    Returns:
        List of token IDs in track-sequential format
    """
    max_seq_len = 2048  # v3.1: doubled for longer musical form

    # If beat_grid provided but no chord_tokens, derive them from the grid
    if beat_grid and not chord_tokens:
        chord_tokens = beat_grid_to_chord_tokens(beat_grid, tokenizer)
        print(f'\nChord tokens derived from {beat_grid.form_name} grid: '
              f'{len(chord_tokens)} tokens')

    # Per-role defaults
    defaults = {
        'lead':    {'min_tokens': 50, 'max_tokens': 400},
        'comping': {'min_tokens': 80, 'max_tokens': 400},
        'bass':    {'min_tokens': 50, 'max_tokens': 400},
        'drums':   {'min_tokens': 50, 'max_tokens': 400},
    }

    # Scale max_tokens with grid duration if present
    if beat_grid:
        grid_max = beat_grid.total_beats * 5
        for role in defaults:
            defaults[role]['max_tokens'] = max(defaults[role]['max_tokens'], grid_max)

    # Per-role pitch constraints
    lead_range_key = f'lead_{lead_instrument}'
    if lead_range_key not in PITCH_RANGES:
        lead_range_key = 'lead_sax'
    pitch_constraints = {
        'lead':    {'pitch_range': PITCH_RANGES[lead_range_key]},
        'comping': {'pitch_range': PITCH_RANGES['comping']},
        'bass':    {'pitch_range': PITCH_RANGES['bass']},
        'drums':   {'allowed_drum_tokens': JAZZ_DRUM_TOKENS},
    }

    # Merge user overrides
    if role_settings:
        for role, overrides in role_settings.items():
            if role in defaults:
                defaults[role].update(overrides)

    roles = [
        ('lead', LEAD_START, COMPING_START),
        ('comping', COMPING_START, BASS_START),
        ('bass', BASS_START, DRUMS_START),
        ('drums', DRUMS_START, tokenizer.eos_id),
    ]

    # Start with [BOS] and optional chord section
    sequence = [tokenizer.bos_id]
    if chord_tokens:
        sequence.append(CHORDS_START)
        sequence.extend(chord_tokens)
        print(f'\nChord section: {len(chord_tokens)} tokens prepended')

    # Parse chord context for theory-guided sampling
    chord_context = None
    ts_id_to_units = tokenizer._ts_id_to_units
    if theory_params and chord_tokens:
        chord_context = parse_chord_context(sequence, ts_id_to_units)
        if chord_context:
            print(f'  Theory-guided: {len(chord_context)} chords parsed '
                  f'(boost={theory_params["chord_tone_boost"]}, '
                  f'scale={theory_params["scale_tone_boost"]}, '
                  f'avoid={theory_params["avoid_penalty"]})')
        else:
            print('  Theory-guided: no chords found, disabled')

    if beat_grid:
        print(f'  Beat grid: {beat_grid.form_name}, '
              f'{beat_grid.total_beats} beats ({beat_grid.total_beats // 4} bars)')

    # Context window warning
    if beat_grid:
        est_total = len(chord_tokens or []) + 10 + beat_grid.total_beats * 5 * 4
        if est_total > 1800:
            print(f'  WARNING: Estimated {est_total} tokens may exceed 2048 context window. '
                  f'Consider fewer choruses.')

    # ── Coltrane features ──
    tension_curve = None
    lead_motivic_memory = None
    if coltrane_params:
        total_time = beat_grid.total_time_units if beat_grid else 384  # ~48 beats default
        if coltrane_params.get('tension_curve'):
            arp_boost = coltrane_params.get('arpeggio_boost', 2.0)
            tension_curve = TensionCurve(total_time, arpeggio_boost=arp_boost)
            print(f'  Tension curve: ON (arpeggio_boost={arp_boost})')
        if coltrane_params.get('motivic_memory'):
            motif_boost = coltrane_params.get('motif_boost', 1.5)
            lead_motivic_memory = MotivicMemory(motif_notes=12, boost=motif_boost)
            print(f'  Motivic memory: ON (boost={motif_boost})')
        if coltrane_params.get('key_center_boost', 0) > 0:
            print(f'  Key center awareness: ON (boost={coltrane_params["key_center_boost"]})')

    for role_name, start_delim, stop_delim in roles:
        model = models[role_name]
        role_def = defaults[role_name]
        role_pitch = pitch_constraints[role_name]

        # Append role start delimiter
        sequence.append(start_delim)

        # Theory-guided only for melodic roles (not drums)
        role_theory = theory_params if role_name != 'drums' else None
        role_chords = chord_context if role_name != 'drums' else None

        # Grid for all roles (drums get grid for duration control, no biases applied)
        role_grid = beat_grid
        role_grid_params = grid_params

        # Coltrane features: tension for melodic roles, motivic memory for lead only
        role_tension = tension_curve if role_name != 'drums' else None
        role_motivic = lead_motivic_memory if role_name == 'lead' else None
        role_coltrane = coltrane_params if role_name != 'drums' else None

        print(f'\nGenerating {role_name}...')
        print(f'  Context length: {len(sequence)} tokens')
        print(f'  min_tokens={role_def["min_tokens"]}, max_tokens={role_def["max_tokens"]}')
        if role_theory and role_chords:
            print(f'  Theory-guided: ON')
        if role_grid:
            print(f'  Beat grid: ON ({beat_grid.total_beats} beats target)')
        if role_tension:
            print(f'  Tension curve: ON')
        if role_motivic:
            print(f'  Motivic memory: ON')

        section_tokens = generate_section(
            model=model,
            context=sequence,
            stop_token=stop_delim,
            max_tokens=role_def['max_tokens'],
            min_tokens=role_def['min_tokens'],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            ngram_block=ngram_block,
            pitch_range=role_pitch.get('pitch_range'),
            allowed_drum_tokens=role_pitch.get('allowed_drum_tokens'),
            chord_context=role_chords,
            theory_params=role_theory,
            beat_grid=role_grid,
            role_name=role_name,
            grid_params=role_grid_params,
            tension_curve=role_tension,
            motivic_memory=role_motivic,
            coltrane_params=role_coltrane,
            ts_id_to_units=ts_id_to_units,
            max_seq_len=max_seq_len,
            device=device,
        )

        # The stop_delim for 'lead' is COMPING_START (next section's start).
        # Remove it since the next iteration adds the start delim.
        if section_tokens and section_tokens[-1] == stop_delim:
            sequence.extend(section_tokens[:-1])
        else:
            sequence.extend(section_tokens)

        n_notes = sum(1 for t in section_tokens if PITCH_MIN <= t <= PITCH_MAX
                      or PITCHDRUM_MIN <= t <= PITCHDRUM_MAX)
        print(f'  Generated: {len(section_tokens)} tokens, ~{n_notes} notes')

    # Final [EOS]
    sequence.append(tokenizer.eos_id)

    return sequence


def analyze_output(tokens, tokenizer):
    """Print analysis of the generated track-sequential sequence."""
    print(f'\n{"="*50}')
    print(f'Generated sequence analysis:')
    print(f'{"="*50}')
    tokenizer.describe(tokens)

    # Check for repetition
    sections = tokenizer._split_at_delimiters(tokens)
    for role in ['lead', 'comping', 'bass', 'drums']:
        sec = sections.get(role, [])
        if len(sec) < 20:
            continue
        # Check 4-gram repetitions
        ngrams = []
        for i in range(len(sec) - 3):
            ngrams.append(tuple(sec[i:i+4]))
        counts = Counter(ngrams)
        top_repeat = counts.most_common(1)[0] if counts else (None, 0)
        if top_repeat[1] > 3:
            print(f'  WARNING: {role} has 4-gram repeated {top_repeat[1]}x')


def main():
    parser = argparse.ArgumentParser(description='Coltrain v3 Cascaded Generation')
    parser.add_argument('--output', default='generated_v3.mid')
    parser.add_argument('--shared-ckpt', help='Use one shared model for all roles')
    parser.add_argument('--lead-ckpt', help='Lead specialist checkpoint')
    parser.add_argument('--comping-ckpt', help='Comping specialist checkpoint')
    parser.add_argument('--bass-ckpt', help='Bass specialist checkpoint')
    parser.add_argument('--drums-ckpt', help='Drums specialist checkpoint')
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.2)
    parser.add_argument('--ngram-block', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=400,
                        help='Default max tokens per role section')
    parser.add_argument('--lead-min-tokens', type=int, default=50)
    parser.add_argument('--comping-min-tokens', type=int, default=80,
                        help='Comping needs more min tokens to prevent early stopping')
    parser.add_argument('--bass-min-tokens', type=int, default=50)
    parser.add_argument('--drums-min-tokens', type=int, default=50)
    parser.add_argument('--tempo', type=int, default=140)
    parser.add_argument('--lead-instrument', default='sax',
                        choices=['sax', 'trumpet', 'piano'],
                        help='Lead instrument (determines pitch range clamping)')
    parser.add_argument('--lead-program', type=int, default=65,
                        help='MIDI program for lead (default: 65=alto sax)')
    parser.add_argument('--comping-program', type=int, default=0,
                        help='MIDI program for comping (default: 0=piano)')
    parser.add_argument('--bass-program', type=int, default=32,
                        help='MIDI program for bass (default: 32=acoustic bass)')
    parser.add_argument('--chords', type=str, default=None,
                        help='Chord progression e.g. "Dm7:4 G7:4 Cmaj7:4 Cmaj7:4"')
    parser.add_argument('--chord-tone-boost', type=float, default=2.0,
                        help='Logit boost for chord tones (theory-guided sampling)')
    parser.add_argument('--scale-tone-boost', type=float, default=1.0,
                        help='Logit boost for scale tones (theory-guided sampling)')
    parser.add_argument('--avoid-penalty', type=float, default=1.5,
                        help='Logit penalty for avoid notes (theory-guided sampling)')
    parser.add_argument('--no-theory', action='store_true',
                        help='Disable theory-guided sampling (for A/B comparison)')
    # Form-based generation
    parser.add_argument('--form', type=str, default=None,
                        choices=list(FORM_TEMPLATES.keys()),
                        help='Jazz form template (e.g., blues12, aaba32)')
    parser.add_argument('--key', type=str, default='C',
                        help='Key for form template (e.g., C, Bb, F#)')
    parser.add_argument('--choruses', type=int, default=1,
                        help='Number of choruses (repeats of the form)')
    parser.add_argument('--bass-root-boost', type=float, default=3.0,
                        help='Beat-1 root boost for bass (grid)')
    parser.add_argument('--approach-boost', type=float, default=1.5,
                        help='Approach note boost before chord changes (grid)')
    parser.add_argument('--phrase-breathing', type=float, default=1.5,
                        help='Rest boost at phrase boundaries for lead (grid)')
    # Coltrane features
    parser.add_argument('--coltrane', action='store_true',
                        help='Enable all Coltrane features (tension curve, motivic memory, '
                             'key centers, sheets of sound)')
    parser.add_argument('--tension-curve', action='store_true',
                        help='Enable tension arc: exposition→development→climax→resolution')
    parser.add_argument('--motivic-memory', action='store_true',
                        help='Enable motivic development (lead captures first phrase, '
                             'develops it throughout)')
    parser.add_argument('--key-center-boost', type=float, default=0.0,
                        help='Boost for key center scale tones (geometric awareness, '
                             '0=off, 0.5=default with --coltrane)')
    parser.add_argument('--arpeggio-boost', type=float, default=0.0,
                        help='Peak arpeggio boost at climax — sheets of sound effect '
                             '(0=off, 2.0=default with --coltrane)')
    parser.add_argument('--motif-boost', type=float, default=0.0,
                        help='Boost when motif interval pattern matches '
                             '(0=off, 1.5=default with --coltrane)')
    parser.add_argument('--device', default=None,
                        help='Device (auto-detected if not set)')
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Device: {device}')

    # Tokenizer (needed for vocab_size and model loading)
    tokenizer = TrackSeqTokenizer()
    vocab_size = tokenizer.vocab_size

    # Load models
    if args.shared_ckpt:
        print(f'\nUsing shared model for all roles')
        shared = load_model(args.shared_ckpt, vocab_size, device)
        models = {r: shared for r in ['lead', 'comping', 'bass', 'drums']}
    else:
        required = {'lead': args.lead_ckpt, 'comping': args.comping_ckpt,
                     'bass': args.bass_ckpt, 'drums': args.drums_ckpt}
        missing = [r for r, p in required.items() if not p]
        if missing:
            parser.error(f'Missing checkpoint(s): {missing}. '
                         f'Use --shared-ckpt or provide all 4 specialist checkpoints.')
        models = {r: load_model(p, vocab_size, device) for r, p in required.items()}

    # Per-role token settings
    role_settings = {
        'lead':    {'min_tokens': args.lead_min_tokens, 'max_tokens': args.max_tokens},
        'comping': {'min_tokens': args.comping_min_tokens, 'max_tokens': args.max_tokens},
        'bass':    {'min_tokens': args.bass_min_tokens, 'max_tokens': args.max_tokens},
        'drums':   {'min_tokens': args.drums_min_tokens, 'max_tokens': args.max_tokens},
    }

    print(f'\nVocab: {tokenizer.vocab_size}')
    print(f'Lead instrument: {args.lead_instrument} '
          f'(pitch range: {PITCH_RANGES.get(f"lead_{args.lead_instrument}", "default")})')

    # Encode chord progression if provided
    chord_tokens = None
    if args.chords:
        chord_tokens = tokenizer.encode_chord_string(args.chords)
        print(f'Chord input: "{args.chords}" -> {len(chord_tokens)} tokens')

    # Build beat grid from form template (if specified)
    beat_grid = None
    grid_params = None

    KEY_MAP = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }

    if args.form:
        template = FORM_TEMPLATES[args.form]
        key_pc = KEY_MAP.get(args.key, 0)
        beat_grid = template.build_grid(key_pc, args.choruses)
        grid_params = {
            'bass_root_boost': args.bass_root_boost,
            'approach_boost': args.approach_boost,
            'phrase_breathing': args.phrase_breathing,
        }
        print(f'Form: {args.form} in {args.key} ({args.choruses} chorus(es))')
        print(f'  {beat_grid.total_beats} beats, {beat_grid.total_beats // 4} bars')
        print(f'  Grid params: bass_root={args.bass_root_boost}, '
              f'approach={args.approach_boost}, breathing={args.phrase_breathing}')

        # If no --chords provided, chord_tokens will be derived from grid in generate_cascaded
        # If --chords provided, they override the form's default chords

    # Theory-guided sampling params (enabled when chords are available)
    # With --form, chords come from the grid, so theory is enabled automatically
    theory_params = None
    has_chords = chord_tokens is not None or beat_grid is not None
    if not args.no_theory and has_chords:
        theory_params = {
            'chord_tone_boost': args.chord_tone_boost,
            'scale_tone_boost': args.scale_tone_boost,
            'avoid_penalty': args.avoid_penalty,
        }
        print(f'Theory-guided sampling: ON '
              f'(chord={args.chord_tone_boost}, scale={args.scale_tone_boost}, '
              f'avoid={args.avoid_penalty})')
    elif args.no_theory:
        print('Theory-guided sampling: OFF (--no-theory)')
    else:
        print('Theory-guided sampling: OFF (no chords provided)')

    # Coltrane features (--coltrane enables all with sensible defaults)
    coltrane_params = None
    use_tension = args.tension_curve or args.coltrane
    use_motivic = args.motivic_memory or args.coltrane
    kc_boost = args.key_center_boost if args.key_center_boost > 0 else (0.5 if args.coltrane else 0)
    arp_boost = args.arpeggio_boost if args.arpeggio_boost > 0 else (2.0 if args.coltrane else 0)
    motif_boost = args.motif_boost if args.motif_boost > 0 else (1.5 if args.coltrane else 0)

    if use_tension or use_motivic or kc_boost > 0:
        coltrane_params = {
            'tension_curve': use_tension,
            'motivic_memory': use_motivic,
            'key_center_boost': kc_boost,
            'arpeggio_boost': arp_boost,
            'motif_boost': motif_boost,
        }
        features = []
        if use_tension: features.append(f'tension(arp={arp_boost})')
        if use_motivic: features.append(f'motivic(boost={motif_boost})')
        if kc_boost > 0: features.append(f'key_centers({kc_boost})')
        print(f'Coltrane features: {", ".join(features)}')
    else:
        print('Coltrane features: OFF')

    tokens = generate_cascaded(
        models=models,
        tokenizer=tokenizer,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        ngram_block=args.ngram_block,
        role_settings=role_settings,
        lead_instrument=args.lead_instrument,
        chord_tokens=chord_tokens,
        theory_params=theory_params,
        beat_grid=beat_grid,
        grid_params=grid_params,
        coltrane_params=coltrane_params,
        device=device,
    )

    # Analyze
    analyze_output(tokens, tokenizer)

    # Detokenize to MIDI
    tokenizer.detokenize_to_midi(
        tokens, args.output, tempo=args.tempo,
        lead_program=args.lead_program,
        comping_program=args.comping_program,
        bass_program=args.bass_program,
    )
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
