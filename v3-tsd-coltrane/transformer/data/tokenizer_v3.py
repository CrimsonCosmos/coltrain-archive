#!/usr/bin/env python3
"""
Track-sequential tokenizer for Coltrain v3.1.

Converts multi-track jazz MIDI into track-sequential format with chord section:
  [BOS] [CHORDS] chords... [LEAD] ... [COMPING] ... [BASS] ... [DRUMS] ... [EOS]

Each section contains MidiTok TSD tokens (TimeShift, Pitch/PitchDrum, Velocity,
Duration) with Program tokens preserved for multi-instrument sections.

The chord section contains Root + Quality + TimeShift tokens extracted from
comping instrument notes (or inferred from bass if no comping).

Token layout (vocab=457):
  0-426:   Base MidiTok TSD tokens
  427:     CHORDS_START
  428:     LEAD_START
  429:     COMPING_START
  430:     BASS_START
  431:     DRUMS_START
  432-443: Root_C, Root_Db, Root_D, Root_Eb, Root_E, Root_F,
           Root_Gb, Root_G, Root_Ab, Root_A, Root_Bb, Root_B
  444-456: Qual_maj, Qual_min, Qual_7, Qual_maj7, Qual_min7,
           Qual_dim, Qual_aug, Qual_m7b5, Qual_dim7,
           Qual_sus4, Qual_sus2, Qual_6, Qual_min6

Role assignment heuristic:
  - If horns/reeds (programs 56-79) exist -> lead=horns, comping=piano+guitar+organ
  - If no horns -> lead=piano or guitar, comping=empty
  - Bass always = programs 32-39
  - Drums always = percussion (program -1)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.tokenizer_miditok import MidiTokWrapper

# ── Token ID ranges from MidiTok TSD (vocab=427) ──
PITCH_MIN, PITCH_MAX = 3, 91
VEL_MIN, VEL_MAX = 92, 107
DUR_MIN, DUR_MAX = 108, 171
TS_MIN, TS_MAX = 172, 235
PITCHDRUM_MIN, PITCHDRUM_MAX = 236, 297
PROG_MIN, PROG_MAX = 298, 426

# ── Section delimiter token IDs ──
CHORDS_START = 427
LEAD_START = 428
COMPING_START = 429
BASS_START = 430
DRUMS_START = 431

# ── Chord tokens ──
ROOT_MIN = 432
ROOT_MAX = 443
QUAL_MIN = 444
QUAL_MAX = 456

# Root names indexed by pitch class (0=C, 1=Db, ..., 11=B)
ROOT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Quality names — 13 common jazz chord types
QUALITY_NAMES = [
    'maj', 'min', '7', 'maj7', 'min7',
    'dim', 'aug', 'm7b5', 'dim7',
    'sus4', 'sus2', '6', 'min6',
]

# Chord templates: frozenset of intervals from root (in semitones)
# Maps interval set -> quality index
CHORD_TEMPLATES = {
    frozenset({0, 4, 7}):       0,   # maj
    frozenset({0, 3, 7}):       1,   # min
    frozenset({0, 4, 7, 10}):   2,   # 7 (dominant)
    frozenset({0, 4, 7, 11}):   3,   # maj7
    frozenset({0, 3, 7, 10}):   4,   # min7
    frozenset({0, 3, 6}):       5,   # dim
    frozenset({0, 4, 8}):       6,   # aug
    frozenset({0, 3, 6, 10}):   7,   # m7b5 (half-dim)
    frozenset({0, 3, 6, 9}):    8,   # dim7
    frozenset({0, 5, 7}):       9,   # sus4
    frozenset({0, 2, 7}):       10,  # sus2
    frozenset({0, 4, 7, 9}):    11,  # 6
    frozenset({0, 3, 7, 9}):    12,  # min6
}

# ── Program classification ranges ──
HORN_PROGRAMS = set(range(56, 80))
PIANO_PROGRAMS = set(range(0, 8))
GUITAR_PROGRAMS = set(range(24, 32))
BASS_PROGRAMS = set(range(32, 40))
ORGAN_PROGRAMS = set(range(16, 24))


def _is_pitch(tid):     return PITCH_MIN <= tid <= PITCH_MAX
def _is_vel(tid):       return VEL_MIN <= tid <= VEL_MAX
def _is_dur(tid):       return DUR_MIN <= tid <= DUR_MAX
def _is_ts(tid):        return TS_MIN <= tid <= TS_MAX
def _is_pitchdrum(tid): return PITCHDRUM_MIN <= tid <= PITCHDRUM_MAX
def _is_prog(tid):      return PROG_MIN <= tid <= PROG_MAX
def _is_root(tid):      return ROOT_MIN <= tid <= ROOT_MAX
def _is_qual(tid):      return QUAL_MIN <= tid <= QUAL_MAX


def classify_programs(programs):
    """Classify a set of MIDI programs into 4 roles.

    Returns dict mapping role name -> set of program numbers.
    """
    has_horns = bool(programs & HORN_PROGRAMS)

    roles = {'lead': set(), 'comping': set(), 'bass': set(), 'drums': set()}

    for prog in programs:
        if prog == -1:
            roles['drums'].add(prog)
        elif prog in BASS_PROGRAMS:
            roles['bass'].add(prog)
        elif has_horns:
            if prog in HORN_PROGRAMS:
                roles['lead'].add(prog)
            else:
                roles['comping'].add(prog)
        else:
            # No horns: piano/guitar -> lead, everything else -> comping
            if prog in PIANO_PROGRAMS or prog in GUITAR_PROGRAMS:
                roles['lead'].add(prog)
            else:
                roles['comping'].add(prog)

    return roles


def _detect_chord(pitch_classes):
    """Detect chord quality from a set of pitch classes.

    Args:
        pitch_classes: set of integers 0-11

    Returns:
        (root_pc, quality_idx) or None if no match
    """
    if len(pitch_classes) < 2:
        return None

    pcs = sorted(pitch_classes)

    # Try each pitch class as potential root
    best_match = None
    best_score = 0

    for root in pcs:
        intervals = frozenset((pc - root) % 12 for pc in pcs)

        # Exact match
        if intervals in CHORD_TEMPLATES:
            return (root, CHORD_TEMPLATES[intervals])

        # Partial match: find template with most overlap
        for template, qual_idx in CHORD_TEMPLATES.items():
            overlap = len(intervals & template)
            # Score: overlap as fraction of template size, penalize extra notes
            score = overlap / len(template) - 0.1 * (len(intervals) - overlap)
            if score > best_score:
                best_score = score
                best_match = (root, qual_idx)

    # Accept partial match if at least 60% overlap
    if best_match and best_score >= 0.5:
        return best_match

    # Fallback: lowest pitch class = root, guess major
    return (pcs[0], 0)


class TrackSeqTokenizer:
    """Track-sequential tokenizer with chord conditioning for cascaded jazz generation."""

    def __init__(self):
        self.base = MidiTokWrapper()
        self.vocab_size = 457  # 427 base + 5 delimiters + 12 roots + 13 qualities
        self.pad_id = self.base.pad_id        # 0
        self.bos_id = self.base.bos_id        # 1
        self.eos_id = self.base.eos_id        # 2
        self.chords_start_id = CHORDS_START   # 427
        self.lead_start_id = LEAD_START       # 428
        self.comping_start_id = COMPING_START # 429
        self.bass_start_id = BASS_START       # 430
        self.drums_start_id = DRUMS_START     # 431

        self._build_timeshift_lookup()

    # ── TimeShift <-> abstract time units ──

    def _build_timeshift_lookup(self):
        """Map TimeShift token IDs <-> abstract time units (8ths of a beat)."""
        self._ts_id_to_units = {}

        for token_id in range(TS_MIN, TS_MAX + 1):
            token_str = self.base.id_to_token_str(token_id)
            # Format: "TimeShift_beats.positions.resolution"
            parts = token_str.split("_")[1].split(".")
            beats, pos, res = int(parts[0]), int(parts[1]), int(parts[2])
            units = beats * 8 + (pos * 8) // res
            self._ts_id_to_units[token_id] = units

        # Sorted descending for greedy decomposition
        self._sorted_ts = sorted(
            [(units, tid) for tid, units in self._ts_id_to_units.items()],
            key=lambda x: -x[0]
        )

    def _delta_to_ts_tokens(self, delta_units):
        """Convert an abstract time delta to a list of TimeShift token IDs."""
        tokens = []
        remaining = delta_units
        while remaining > 0:
            emitted = False
            for units, tid in self._sorted_ts:
                if 0 < units <= remaining:
                    tokens.append(tid)
                    remaining -= units
                    emitted = True
                    break
            if not emitted:
                break  # remaining < smallest TimeShift (shouldn't happen)
        return tokens

    # ── Parsing interleaved TSD tokens ──

    def _parse_interleaved(self, token_ids):
        """Parse interleaved MidiTok TSD tokens into structured note events.

        Returns list of dicts with keys:
          time, program, pitch_tid, vel_tid, dur_tid, is_drum, midi_pitch
        """
        events = []
        current_time = 0
        current_program = 0
        i = 0

        while i < len(token_ids):
            tid = token_ids[i]

            if tid in (self.pad_id, self.bos_id, self.eos_id):
                i += 1
                continue

            if _is_ts(tid):
                current_time += self._ts_id_to_units[tid]
                i += 1
                continue

            if _is_prog(tid):
                prog_str = self.base.id_to_token_str(tid)
                current_program = int(prog_str.split("_")[1])
                i += 1
                continue

            if _is_pitch(tid) or _is_pitchdrum(tid):
                is_drum = _is_pitchdrum(tid)
                vel_tid = token_ids[i + 1] if i + 1 < len(token_ids) else None
                dur_tid = token_ids[i + 2] if i + 2 < len(token_ids) else None

                if (vel_tid is not None and _is_vel(vel_tid)
                        and dur_tid is not None and _is_dur(dur_tid)):
                    # Extract MIDI pitch for chord detection
                    if is_drum:
                        midi_pitch = tid - 209  # PitchDrum token -> MIDI note
                    else:
                        midi_pitch = tid + 18   # Pitch token -> MIDI note
                    events.append({
                        'time': current_time,
                        'program': -1 if is_drum else current_program,
                        'pitch_tid': tid,
                        'vel_tid': vel_tid,
                        'dur_tid': dur_tid,
                        'is_drum': is_drum,
                        'midi_pitch': midi_pitch,
                    })
                    i += 3
                    continue

            i += 1

        return events

    def _events_to_tokens(self, events):
        """Convert sorted note events back into TSD token sequence.

        Emits TimeShift deltas, Program changes, Pitch, Velocity, Duration.
        """
        tokens = []
        current_time = 0
        current_program = None

        for event in events:
            # TimeShift delta
            delta = event['time'] - current_time
            if delta > 0:
                tokens.extend(self._delta_to_ts_tokens(delta))
                current_time = event['time']

            # Program change
            prog = event['program']
            if prog != current_program:
                prog_str = f"Program_{prog}"
                try:
                    prog_tid = self.base.tokenizer[prog_str]
                    tokens.append(prog_tid)
                except KeyError:
                    pass
                current_program = prog

            # Note: Pitch/PitchDrum + Velocity + Duration
            tokens.append(event['pitch_tid'])
            tokens.append(event['vel_tid'])
            tokens.append(event['dur_tid'])

        return tokens

    # ── Chord extraction ──

    def _extract_chords(self, events, window_units=16):
        """Extract chord progression from note events.

        Groups notes into time windows, detects pitch class sets, matches
        against chord templates.

        Args:
            events: list of note event dicts (non-drum only)
            window_units: time units per chord window (16 = 2 beats)

        Returns:
            list of (time_units, root_pc, quality_idx) tuples
        """
        if not events:
            return []

        # Filter out drums
        melodic = [e for e in events if not e.get('is_drum', False)]
        if not melodic:
            return []

        max_time = max(e['time'] for e in melodic)

        chords = []
        for win_start in range(0, max_time + 1, window_units):
            win_end = win_start + window_units

            # Collect pitch classes in this window
            pcs = set()
            for e in melodic:
                if win_start <= e['time'] < win_end:
                    pcs.add(e['midi_pitch'] % 12)

            if len(pcs) < 2:
                continue

            result = _detect_chord(pcs)
            if result is None:
                continue

            root_pc, qual_idx = result

            # Deduplicate consecutive identical chords
            if chords and chords[-1][1] == root_pc and chords[-1][2] == qual_idx:
                continue

            chords.append((win_start, root_pc, qual_idx))

        return chords

    def _chords_to_tokens(self, chords):
        """Convert chord list to token sequence: Root Quality TimeShift ..."""
        tokens = []
        for i, (time, root_pc, qual_idx) in enumerate(chords):
            # Root token
            tokens.append(ROOT_MIN + root_pc)
            # Quality token
            tokens.append(QUAL_MIN + qual_idx)

            # TimeShift to next chord (or end of piece)
            if i + 1 < len(chords):
                delta = chords[i + 1][0] - time
            else:
                delta = 16  # Default: 2 beats for final chord

            if delta > 0:
                tokens.extend(self._delta_to_ts_tokens(delta))

        return tokens

    def encode_chord_string(self, chord_str):
        """Encode a manual chord progression string into tokens.

        Format: "Cmaj7:4 Dm7:4 G7:4 Cmaj7:4"
        Where each chord is Root[Quality]:beats

        Returns: list of token IDs for the chord section (without CHORDS_START)
        """
        # Map common chord names to (root_offset, quality_name)
        root_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
        }
        qual_map = {name: i for i, name in enumerate(QUALITY_NAMES)}
        # Common aliases
        qual_map['M7'] = qual_map['maj7']
        qual_map['m7'] = qual_map['min7']
        qual_map['m'] = qual_map['min']
        qual_map['M'] = qual_map['maj']
        qual_map['dom7'] = qual_map['7']
        qual_map['o7'] = qual_map['dim7']
        qual_map['o'] = qual_map['dim']
        qual_map['+'] = qual_map['aug']
        qual_map['halfdim'] = qual_map['m7b5']

        tokens = []
        for chord_spec in chord_str.strip().split():
            # Parse "Cmaj7:4" or "Dm7:8"
            if ':' in chord_spec:
                chord_part, beats_str = chord_spec.rsplit(':', 1)
                beats = int(beats_str)
            else:
                chord_part = chord_spec
                beats = 4  # Default 4 beats

            # Parse root note
            root_pc = None
            for rname in sorted(root_map.keys(), key=len, reverse=True):
                if chord_part.startswith(rname):
                    root_pc = root_map[rname]
                    quality_str = chord_part[len(rname):]
                    break

            if root_pc is None:
                continue

            # Parse quality
            if not quality_str:
                quality_str = 'maj'
            qual_idx = qual_map.get(quality_str, 0)  # Default to major

            tokens.append(ROOT_MIN + root_pc)
            tokens.append(QUAL_MIN + qual_idx)

            # TimeShift for chord duration
            delta_units = beats * 8  # 8 units per beat
            tokens.extend(self._delta_to_ts_tokens(delta_units))

        return tokens

    # ── Main tokenize / detokenize ──

    def tokenize_midi(self, midi_path):
        """Tokenize a MIDI file to track-sequential format with chord section.

        Returns: list[int] -- token IDs in format:
          [BOS] [CHORDS] chords... [LEAD] lead... [COMPING] comp...
          [BASS] bass... [DRUMS] drums... [EOS]
        """
        # Get interleaved tokens from base MidiTok TSD
        base_tokens = self.base.tokenize_midi(midi_path)

        # Parse into structured events
        events = self._parse_interleaved(base_tokens)

        if not events:
            return [self.bos_id, CHORDS_START, LEAD_START, COMPING_START,
                    BASS_START, DRUMS_START, self.eos_id]

        # Classify programs into roles
        programs = set(e['program'] for e in events)
        role_map = classify_programs(programs)

        # Group events by role
        role_events = {'lead': [], 'comping': [], 'bass': [], 'drums': []}
        for event in events:
            assigned = False
            for role, progs in role_map.items():
                if event['program'] in progs:
                    role_events[role].append(event)
                    assigned = True
                    break
            if not assigned:
                role_events['comping'].append(event)

        # Sort each role's events by time (stable for simultaneous notes)
        for role in role_events:
            role_events[role].sort(key=lambda e: e['time'])

        # Extract chord progression (prefer comping, fallback to bass+lead)
        chord_source = role_events['comping'] or role_events['lead']
        # Include bass notes for better root detection
        if role_events['bass']:
            chord_source = chord_source + role_events['bass']
        chords = self._extract_chords(chord_source)
        chord_tokens = self._chords_to_tokens(chords)

        # Build track-sequential token stream
        tokens = [self.bos_id]

        # Chord section
        tokens.append(CHORDS_START)
        tokens.extend(chord_tokens)

        # Role sections
        for role_name, delimiter in [
            ('lead', LEAD_START),
            ('comping', COMPING_START),
            ('bass', BASS_START),
            ('drums', DRUMS_START),
        ]:
            tokens.append(delimiter)
            tokens.extend(self._events_to_tokens(role_events[role_name]))
        tokens.append(self.eos_id)

        return tokens

    def detokenize_to_midi(self, tokens, output_path, tempo=120,
                           lead_program=0, comping_program=0,
                           bass_program=32):
        """Convert track-sequential tokens back to multi-track MIDI.

        Splits at role delimiters, parses each section (skipping chord section),
        merges into a single interleaved stream, decodes with base MidiTok.
        """
        sections = self._split_at_delimiters(tokens)

        # Parse each section into events with default programs
        default_progs = {
            'lead': lead_program,
            'comping': comping_program,
            'bass': bass_program,
            'drums': -1,
        }

        all_events = []
        for role_name in ['lead', 'comping', 'bass', 'drums']:
            section_tokens = sections.get(role_name, [])
            if not section_tokens:
                continue
            events = self._parse_section(section_tokens, default_progs[role_name])
            all_events.extend(events)

        if not all_events:
            return

        # Merge all events by time -> interleaved TSD stream
        all_events.sort(key=lambda e: e['time'])
        interleaved = [self.bos_id]
        interleaved.extend(self._events_to_tokens(all_events))
        interleaved.append(self.eos_id)

        self.base.detokenize_to_midi(interleaved, output_path, tempo)

    def _split_at_delimiters(self, tokens):
        """Split token sequence at section delimiter tokens."""
        delimiters = {
            CHORDS_START: 'chords',
            LEAD_START: 'lead',
            COMPING_START: 'comping',
            BASS_START: 'bass',
            DRUMS_START: 'drums',
        }

        sections = {}
        current_role = None
        current_tokens = []

        for tid in tokens:
            if tid in delimiters:
                if current_role is not None:
                    sections[current_role] = current_tokens
                current_role = delimiters[tid]
                current_tokens = []
            elif tid in (self.bos_id, self.eos_id, self.pad_id):
                if current_role is not None:
                    sections[current_role] = current_tokens
                    current_role = None
                    current_tokens = []
            elif current_role is not None:
                current_tokens.append(tid)

        if current_role is not None:
            sections[current_role] = current_tokens

        return sections

    def _parse_section(self, section_tokens, default_program):
        """Parse a role section's tokens into structured note events.

        Program tokens in the section are IGNORED -- the role's assigned program
        (default_program) is always used. Root/Quality tokens are also skipped.
        """
        events = []
        current_time = 0
        i = 0

        while i < len(section_tokens):
            tid = section_tokens[i]

            if _is_ts(tid):
                current_time += self._ts_id_to_units[tid]
                i += 1
                continue

            # Skip Program, Root, Quality tokens
            if _is_prog(tid) or _is_root(tid) or _is_qual(tid):
                i += 1
                continue

            if _is_pitch(tid) or _is_pitchdrum(tid):
                is_drum = _is_pitchdrum(tid)
                vel_tid = section_tokens[i + 1] if i + 1 < len(section_tokens) else None
                dur_tid = section_tokens[i + 2] if i + 2 < len(section_tokens) else None

                if (vel_tid is not None and _is_vel(vel_tid)
                        and dur_tid is not None and _is_dur(dur_tid)):
                    events.append({
                        'time': current_time,
                        'program': -1 if is_drum else default_program,
                        'pitch_tid': tid,
                        'vel_tid': vel_tid,
                        'dur_tid': dur_tid,
                        'is_drum': is_drum,
                    })
                    i += 3
                    continue

            i += 1

        return events

    # ── Role masking for specialist training ──

    def get_role_mask(self, tokens, role):
        """Create a boolean mask for specialist fine-tuning.

        Returns a mask of len(tokens). mask[i] = True for tokens that belong
        to the given role's section (content tokens + the end delimiter).
        """
        role_to_delims = {
            'lead': (LEAD_START, COMPING_START),
            'comping': (COMPING_START, BASS_START),
            'bass': (BASS_START, DRUMS_START),
            'drums': (DRUMS_START, self.eos_id),
        }

        start_delim, end_delim = role_to_delims[role]
        mask = [False] * len(tokens)
        in_section = False

        for i, tid in enumerate(tokens):
            if tid == start_delim:
                in_section = True
                continue  # Don't mask the start delimiter
            if in_section and tid == end_delim:
                mask[i] = True  # Mask the end delimiter (model must learn to stop)
                in_section = False
                continue
            if in_section:
                mask[i] = True

        return mask

    # ── Diagnostics ──

    def describe(self, tokens):
        """Print a summary of a track-sequential token sequence."""
        sections = self._split_at_delimiters(tokens)
        total = len(tokens)
        print(f"Total tokens: {total}")

        # Chord section
        chord_sec = sections.get('chords', [])
        n_chords = sum(1 for t in chord_sec if _is_root(t))
        print(f"  chords  : {len(chord_sec):4d} tokens, {n_chords:3d} chords")

        for role in ['lead', 'comping', 'bass', 'drums']:
            sec = sections.get(role, [])
            events = self._parse_section(sec, 0) if sec else []
            programs = set(e['program'] for e in events) if events else set()
            print(f"  {role:8s}: {len(sec):4d} tokens, {len(events):3d} notes, "
                  f"programs={programs or '{}'}")

    def describe_chords(self, tokens):
        """Print the detected chord progression."""
        sections = self._split_at_delimiters(tokens)
        chord_sec = sections.get('chords', [])
        if not chord_sec:
            print("No chord section found")
            return

        i = 0
        beat = 0
        chords_str = []
        while i < len(chord_sec):
            tid = chord_sec[i]
            if _is_root(tid):
                root = ROOT_NAMES[tid - ROOT_MIN]
                qual = ''
                if i + 1 < len(chord_sec) and _is_qual(chord_sec[i + 1]):
                    qual = QUALITY_NAMES[chord_sec[i + 1] - QUAL_MIN]
                    i += 1
                chords_str.append(f"  beat {beat:3d}: {root}{qual}")
                i += 1
            elif _is_ts(tid):
                beat += self._ts_id_to_units[tid] // 8
                i += 1
            else:
                i += 1

        print(f"Chord progression ({len(chords_str)} chords):")
        for s in chords_str:
            print(s)


# ── CLI test ──

def test_tokenizer():
    """Round-trip test: MIDI -> track-seq tokens -> MIDI."""
    import glob

    tok = TrackSeqTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"TimeShift tokens: {len(tok._ts_id_to_units)}")
    print(f"Max TimeShift: {max(tok._ts_id_to_units.values())} units "
          f"({max(tok._ts_id_to_units.values())/8:.1f} beats)")
    print()

    # Find test MIDI files
    test_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'augmented_dataset_v2')
    midis = sorted(glob.glob(os.path.join(test_dir, '*.mid')))[:5]

    if not midis:
        print(f"No MIDI files found in {test_dir}")
        return

    for path in midis:
        name = os.path.basename(path)
        print(f"--- {name} ---")

        # Tokenize with track-sequential + chords
        ts_tokens = tok.tokenize_midi(path)
        tok.describe(ts_tokens)
        tok.describe_chords(ts_tokens)

        # Count total note events across role sections
        sections = tok._split_at_delimiters(ts_tokens)
        total_ts_events = 0
        for role in ['lead', 'comping', 'bass', 'drums']:
            sec = sections.get(role, [])
            if sec:
                total_ts_events += len(tok._parse_section(sec, 0))

        # Compare against base interleaved
        base_tokens = tok.base.tokenize_midi(path)
        base_events = tok._parse_interleaved(base_tokens)
        print(f"  Base events: {len(base_events)}, "
              f"Track-seq events: {total_ts_events}, "
              f"Match: {len(base_events) == total_ts_events}")

        # Round-trip: detokenize to temp MIDI, re-tokenize, compare event counts
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            tmp_path = tmp.name
        tok.detokenize_to_midi(ts_tokens, tmp_path)
        rt_tokens = tok.tokenize_midi(tmp_path)
        rt_sections = tok._split_at_delimiters(rt_tokens)
        rt_events = sum(
            len(tok._parse_section(rt_sections.get(r, []), 0))
            for r in ['lead', 'comping', 'bass', 'drums']
        )
        print(f"  Round-trip events: {rt_events}")
        os.unlink(tmp_path)
        print()


if __name__ == '__main__':
    test_tokenizer()
