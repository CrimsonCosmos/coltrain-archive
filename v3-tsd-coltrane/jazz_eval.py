#!/usr/bin/env python3
"""
Jazz Evaluation Script — Quantitative scoring of generated jazz MIDI.

Computes 7 metrics across multiple musical dimensions and produces a composite score.

Usage:
    python jazz_eval.py generated_v3.mid
    python jazz_eval.py generated_v3.mid --chords "Dm7:4 G7:4 Cmaj7:4 Cmaj7:4"
    python jazz_eval.py generated_v3.mid real_jazz.mid --compare

Metrics:
    CTR  Chord-Tone Ratio     % of lead notes that are chord tones
    STR  Scale-Tone Ratio     % of lead notes on scale tones
    VLS  Voice Leading         % of melodic intervals <= 2 semitones
    NDC  Note Density          Consistency of note density over time
    PRU  Pitch Range Usage     Fraction of expected range used
    RPS  Repetition Score      Absence of excessive n-gram repetition
    RHV  Rhythmic Variety      Entropy of inter-onset interval distribution
"""
import sys
import os
import argparse
import math
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformer'))

from data.tokenizer_v3 import (
    TrackSeqTokenizer, CHORD_TEMPLATES,
    PITCH_MIN, PITCH_MAX, PITCHDRUM_MIN, PITCHDRUM_MAX,
    TS_MIN, TS_MAX, ROOT_MIN, ROOT_MAX, QUAL_MIN, QUAL_MAX,
)

# ── Jazz scale mappings (same as generate_v3.py) ──
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

CHORD_TONE_INTERVALS = {}
for _template, _qual_idx in CHORD_TEMPLATES.items():
    CHORD_TONE_INTERVALS[_qual_idx] = _template

# Expected pitch ranges per role (MIDI pitch)
EXPECTED_RANGES = {
    'lead':    (49, 80),   # Alto sax range
    'comping': (36, 84),   # Piano comp range
    'bass':    (28, 67),   # Acoustic bass range
    'drums':   (35, 59),   # GM drum kit
}

# Scoring targets: (target_lo, target_hi, max_for_ramp)
# If raw metric is in [target_lo, target_hi] -> score = 1.0
# Below target_lo: linear ramp from 0
# Above target_hi (where applicable): linear ramp down
METRIC_TARGETS = {
    'CTR': (0.35, 0.60, 0.85),   # 35-60% chord tones ideal, >85% too consonant
    'STR': (0.65, 0.95, 1.00),   # 65-95% scale tones
    'VLS': (0.10, 0.45, 0.70),   # 10-45% stepwise (real jazz: 0.04-0.46)
    'NDC': (0.50, 1.00, 1.00),   # >50% density consistency
    'PRU': (0.40, 1.00, 1.00),   # >40% range usage
    'RPS': (0.80, 1.00, 1.00),   # >80% repetition score
    'RHV': (0.50, 1.00, 1.00),   # >50% rhythmic variety
}

WEIGHTS = {
    'CTR': 0.20, 'STR': 0.15, 'VLS': 0.15,
    'NDC': 0.15, 'PRU': 0.10, 'RPS': 0.15, 'RHV': 0.10,
}


def score_metric(raw, metric_name):
    """Convert raw metric value to 0-1 score using target ranges."""
    lo, hi, max_val = METRIC_TARGETS[metric_name]

    if raw < 0:
        return 0.0
    if lo <= raw <= hi:
        return 1.0
    if raw < lo:
        return raw / lo if lo > 0 else 0.0
    # raw > hi
    if max_val <= hi:
        return 1.0
    return max(0.0, 1.0 - (raw - hi) / (max_val - hi))


def load_and_parse(midi_path):
    """Load a MIDI file, tokenize, and extract per-role note events + chords.

    Returns:
        (role_notes, chords, tokens)
        role_notes: dict of role -> list of {time, midi_pitch, ...}
        chords: list of (time_units, root_pc, qual_idx)
        tokens: full token sequence
    """
    tok = TrackSeqTokenizer()
    tokens = tok.tokenize_midi(midi_path)
    sections = tok._split_at_delimiters(tokens)

    # Parse chord section
    chord_sec = sections.get('chords', [])
    chords = []
    time = 0
    i = 0
    while i < len(chord_sec):
        tid = chord_sec[i]
        if ROOT_MIN <= tid <= ROOT_MAX:
            root = tid - ROOT_MIN
            qual = 0
            if i + 1 < len(chord_sec) and QUAL_MIN <= chord_sec[i + 1] <= QUAL_MAX:
                qual = chord_sec[i + 1] - QUAL_MIN
                i += 1
            chords.append((time, root, qual))
            i += 1
        elif TS_MIN <= tid <= TS_MAX:
            time += tok._ts_id_to_units[tid]
            i += 1
        else:
            i += 1

    # Parse role sections into note events with MIDI pitch
    role_notes = {}
    for role in ['lead', 'comping', 'bass', 'drums']:
        sec = sections.get(role, [])
        if not sec:
            role_notes[role] = []
            continue

        notes = []
        t = 0
        j = 0
        while j < len(sec):
            tid = sec[j]
            if TS_MIN <= tid <= TS_MAX:
                t += tok._ts_id_to_units[tid]
                j += 1
            elif PITCH_MIN <= tid <= PITCH_MAX:
                midi_pitch = tid + 18
                notes.append({'time': t, 'midi_pitch': midi_pitch, 'token': tid})
                j += 3  # Skip velocity + duration
            elif PITCHDRUM_MIN <= tid <= PITCHDRUM_MAX:
                midi_pitch = tid - 209
                notes.append({'time': t, 'midi_pitch': midi_pitch, 'token': tid})
                j += 3
            else:
                j += 1
        role_notes[role] = notes

    return role_notes, chords, tokens


def get_chord_at_time(chords, time):
    """Find the active chord at a given time."""
    if not chords:
        return None
    active = chords[0]
    for chord_time, root, qual in chords:
        if chord_time <= time:
            active = (chord_time, root, qual)
        else:
            break
    return active


# ── Metric implementations ──

def compute_ctr(notes, chords):
    """Chord-Tone Ratio: fraction of notes that are chord tones."""
    if not notes or not chords:
        return 0.0

    chord_tone_count = 0
    for note in notes:
        chord = get_chord_at_time(chords, note['time'])
        if chord is None:
            continue
        _, root, qual = chord
        intervals = CHORD_TONE_INTERVALS.get(qual, frozenset({0, 4, 7}))
        pc = note['midi_pitch'] % 12
        interval = (pc - root) % 12
        if interval in intervals:
            chord_tone_count += 1

    return chord_tone_count / len(notes)


def compute_str(notes, chords):
    """Scale-Tone Ratio: fraction of notes on the scale for the current chord."""
    if not notes or not chords:
        return 0.0

    scale_tone_count = 0
    for note in notes:
        chord = get_chord_at_time(chords, note['time'])
        if chord is None:
            continue
        _, root, qual = chord
        scale = JAZZ_SCALES.get(qual, frozenset({0, 2, 4, 5, 7, 9, 11}))
        pc = note['midi_pitch'] % 12
        interval = (pc - root) % 12
        if interval in scale:
            scale_tone_count += 1

    return scale_tone_count / len(notes)


def compute_vls(notes):
    """Voice Leading Smoothness: fraction of consecutive intervals <= 2 semitones."""
    if len(notes) < 2:
        return 0.0

    stepwise = 0
    for i in range(1, len(notes)):
        interval = abs(notes[i]['midi_pitch'] - notes[i - 1]['midi_pitch'])
        if interval <= 2:
            stepwise += 1

    return stepwise / (len(notes) - 1)


def compute_ndc(notes, window_units=32):
    """Note Density Consistency: 1 - coefficient of variation of notes per window.

    window_units=32 = 4 beats (one bar at 4/4).
    Only considers the active region (first note to last note), trimming
    empty leading/trailing silence.
    """
    if not notes:
        return 0.0

    min_time = min(n['time'] for n in notes)
    max_time = max(n['time'] for n in notes)
    if max_time == min_time:
        return 1.0

    # Count notes per window (relative to first note)
    span = max_time - min_time
    n_windows = max(1, span // window_units + 1)
    counts = [0] * n_windows
    for note in notes:
        win = min((note['time'] - min_time) // window_units, n_windows - 1)
        counts[win] += 1

    if len(counts) < 2:
        return 1.0

    mean = sum(counts) / len(counts)
    if mean == 0:
        return 0.0

    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    cv = math.sqrt(variance) / mean

    return max(0.0, 1.0 - cv)


def compute_pru(notes, role):
    """Pitch Range Usage: fraction of expected range actually used."""
    if not notes:
        return 0.0

    expected = EXPECTED_RANGES.get(role, (0, 127))
    expected_span = expected[1] - expected[0]
    if expected_span == 0:
        return 1.0

    pitches = [n['midi_pitch'] for n in notes]
    actual_span = max(pitches) - min(pitches)

    return min(1.0, actual_span / expected_span)


def compute_rps(notes):
    """Repetition Score: 1 - (max 4-gram repeat ratio).

    Uses pitch token sequences to detect melodic repetition.
    """
    if len(notes) < 5:
        return 1.0

    pitches = [n['midi_pitch'] for n in notes]
    ngrams = []
    for i in range(len(pitches) - 3):
        ngrams.append(tuple(pitches[i:i + 4]))

    if not ngrams:
        return 1.0

    counts = Counter(ngrams)
    max_count = counts.most_common(1)[0][1]

    return max(0.0, 1.0 - max_count / len(ngrams))


def compute_rhv(notes):
    """Rhythmic Variety: normalized entropy of inter-onset interval distribution."""
    if len(notes) < 2:
        return 0.0

    # Compute inter-onset intervals
    iois = []
    for i in range(1, len(notes)):
        ioi = notes[i]['time'] - notes[i - 1]['time']
        iois.append(ioi)

    if not iois:
        return 0.0

    # Compute entropy of IOI distribution
    counts = Counter(iois)
    total = len(iois)
    probs = [c / total for c in counts.values()]

    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Normalize by max possible entropy (log2 of number of unique values)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0

    return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


# ── Evaluation ──

def evaluate(midi_path, chord_string=None):
    """Evaluate a MIDI file on all jazz metrics.

    Returns:
        dict with 'metrics' (raw values), 'scores' (0-1), 'composite', 'grade'
    """
    role_notes, chords, tokens = load_and_parse(midi_path)

    # If chord_string provided and no chords from MIDI, use manual chords
    if not chords and chord_string:
        tok = TrackSeqTokenizer()
        chord_tokens = tok.encode_chord_string(chord_string)
        # Parse the chord tokens
        time = 0
        i = 0
        while i < len(chord_tokens):
            tid = chord_tokens[i]
            if ROOT_MIN <= tid <= ROOT_MAX:
                root = tid - ROOT_MIN
                qual = 0
                if i + 1 < len(chord_tokens) and QUAL_MIN <= chord_tokens[i + 1] <= QUAL_MAX:
                    qual = chord_tokens[i + 1] - QUAL_MIN
                    i += 1
                chords.append((time, root, qual))
                i += 1
            elif TS_MIN <= tid <= TS_MAX:
                time += tok._ts_id_to_units[tid]
                i += 1
            else:
                i += 1

    lead = role_notes.get('lead', [])

    # Compute metrics (lead-focused for harmonic metrics)
    metrics = {
        'CTR': compute_ctr(lead, chords),
        'STR': compute_str(lead, chords),
        'VLS': compute_vls(lead),
        'NDC': compute_ndc(lead),
        'PRU': compute_pru(lead, 'lead'),
        'RPS': compute_rps(lead),
        'RHV': compute_rhv(lead),
    }

    # Per-role secondary metrics
    role_metrics = {}
    for role in ['comping', 'bass', 'drums']:
        notes = role_notes.get(role, [])
        if notes:
            role_metrics[role] = {
                'VLS': compute_vls(notes),
                'NDC': compute_ndc(notes),
                'PRU': compute_pru(notes, role),
                'RPS': compute_rps(notes),
                'RHV': compute_rhv(notes),
                'n_notes': len(notes),
            }

    # Score each metric
    scores = {k: score_metric(v, k) for k, v in metrics.items()}

    # Composite score
    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    # Letter grade
    if composite >= 0.90:
        grade = 'A'
    elif composite >= 0.80:
        grade = 'B+'
    elif composite >= 0.70:
        grade = 'B'
    elif composite >= 0.60:
        grade = 'B-'
    elif composite >= 0.50:
        grade = 'C+'
    elif composite >= 0.40:
        grade = 'C'
    elif composite >= 0.30:
        grade = 'C-'
    else:
        grade = 'D'

    return {
        'metrics': metrics,
        'scores': scores,
        'composite': composite,
        'grade': grade,
        'role_metrics': role_metrics,
        'n_lead_notes': len(lead),
        'n_chords': len(chords),
    }


def print_report(result, midi_path):
    """Print a formatted evaluation report."""
    print(f'\n{"=" * 55}')
    print(f'  Jazz Evaluation: {os.path.basename(midi_path)}')
    print(f'{"=" * 55}')
    print(f'  Lead notes: {result["n_lead_notes"]}')
    print(f'  Chords detected: {result["n_chords"]}')

    print(f'\n  Lead Metrics:')
    print(f'  {"Metric":<30s} {"Raw":>6s}  {"Target":>12s}  {"Score":>6s}')
    print(f'  {"-"*60}')

    metric_names = {
        'CTR': 'Chord-Tone Ratio',
        'STR': 'Scale-Tone Ratio',
        'VLS': 'Voice Leading Smoothness',
        'NDC': 'Note Density Consistency',
        'PRU': 'Pitch Range Usage',
        'RPS': 'Repetition Score',
        'RHV': 'Rhythmic Variety',
    }

    for key in ['CTR', 'STR', 'VLS', 'NDC', 'PRU', 'RPS', 'RHV']:
        name = metric_names[key]
        raw = result['metrics'][key]
        lo, hi, _ = METRIC_TARGETS[key]
        score = result['scores'][key]
        target_str = f'[{lo:.2f}-{hi:.2f}]'
        print(f'  {name:<30s} {raw:>6.3f}  {target_str:>12s}  {score:>6.3f}')

    print(f'\n  {"─" * 55}')
    print(f'  Composite Score: {result["composite"]:.3f} / 1.000  '
          f'(Grade: {result["grade"]})')

    # Per-role summaries
    for role, rm in result.get('role_metrics', {}).items():
        print(f'\n  {role.capitalize()} ({rm["n_notes"]} notes):')
        print(f'    VLS={rm["VLS"]:.3f}  NDC={rm["NDC"]:.3f}  '
              f'PRU={rm["PRU"]:.3f}  RPS={rm["RPS"]:.3f}  RHV={rm["RHV"]:.3f}')

    print()


def main():
    parser = argparse.ArgumentParser(description='Jazz MIDI Evaluation')
    parser.add_argument('midi_files', nargs='+', help='MIDI file(s) to evaluate')
    parser.add_argument('--chords', type=str, default=None,
                        help='Chord progression e.g. "Dm7:4 G7:4 Cmaj7:4 Cmaj7:4"')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple files side-by-side')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    args = parser.parse_args()

    results = []
    for midi_path in args.midi_files:
        if not os.path.exists(midi_path):
            print(f'File not found: {midi_path}', file=sys.stderr)
            continue
        result = evaluate(midi_path, args.chords)
        result['file'] = midi_path
        results.append(result)

    if args.json:
        import json
        # Convert frozensets for JSON serialization
        print(json.dumps([{
            'file': r['file'],
            'metrics': r['metrics'],
            'scores': r['scores'],
            'composite': r['composite'],
            'grade': r['grade'],
        } for r in results], indent=2))
        return

    for result in results:
        print_report(result, result['file'])

    if args.compare and len(results) > 1:
        print(f'\n{"=" * 55}')
        print(f'  Comparison')
        print(f'{"=" * 55}')
        print(f'  {"File":<30s} {"Composite":>10s} {"Grade":>6s}')
        print(f'  {"-" * 50}')
        for r in sorted(results, key=lambda x: -x['composite']):
            name = os.path.basename(r['file'])[:28]
            print(f'  {name:<30s} {r["composite"]:>10.3f} {r["grade"]:>6s}')
        print()


if __name__ == '__main__':
    main()
