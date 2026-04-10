#!/usr/bin/env python3
"""
MidiTok TSD tokenizer wrapper for Coltrain.

Replaces the custom compound-token tokenizer with MidiTok's TSD (Time Shift Duration).
Provides the same interface so the rest of the codebase works unchanged.

Key improvements over the old tokenizer:
- Explicit Duration tokens (no NOTE_OFF ambiguity)
- Program-based instrument identification (no heuristic track guessing)
- Battle-tested tokenization from published research
- 16 velocity bins (vs old 8)
"""
from miditok import TSD
from miditok.classes import TokenizerConfig
from pathlib import Path


def create_tsd_tokenizer() -> TSD:
    """Create the standard Coltrain MidiTok TSD tokenizer."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=16,
        special_tokens=["PAD", "BOS", "EOS"],
        use_velocities=True,
        use_programs=True,
        one_token_stream_for_programs=True,
        use_tempos=False,
        use_chords=False,
        use_rests=False,
        use_time_signatures=False,
    )
    return TSD(config)


class MidiTokWrapper:
    """
    Wrapper around MidiTok TSD providing the same interface as the old MIDITokenizer.

    Properties matching old tokenizer:
        vocab_size, pad_id (EVENT_PAD), bos_id (EVENT_START), eos_id (EVENT_END)

    Methods matching old tokenizer:
        tokenize_midi(path) -> list[int]
        detokenize_to_midi(tokens, output_path, tempo=120)
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or create_tsd_tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer["BOS_None"]
        self.eos_id = self.tokenizer["EOS_None"]

        # Aliases for compatibility with old code that uses EVENT_PAD etc.
        self.EVENT_PAD = self.pad_id
        self.EVENT_START = self.bos_id
        self.EVENT_END = self.eos_id

    def tokenize_midi(self, midi_path: str) -> list[int]:
        """Tokenize a MIDI file to a flat list of token IDs."""
        tok_seq = self.tokenizer.encode(Path(midi_path))
        # one_token_stream_for_programs=True → returns a single TokSequence
        if isinstance(tok_seq, list):
            # Shouldn't happen with one_token_stream, but handle gracefully
            ids = []
            for seq in tok_seq:
                ids.extend(seq.ids)
            return ids
        return tok_seq.ids

    def detokenize_to_midi(self, tokens: list[int], output_path: str, tempo: int = 120):
        """Convert token IDs back to a MIDI file."""
        score = self.tokenizer.decode(tokens)
        # Always set the requested tempo (MidiTok decode inserts default 120)
        from symusic import Tempo as SyTempo
        if len(score.tempos) == 0:
            score.tempos.append(SyTempo(0, tempo))
        else:
            score.tempos[0] = SyTempo(0, tempo)
        score.dump_midi(output_path)

    def id_to_token_str(self, token_id: int) -> str:
        """Convert a token ID to its string representation (for debugging)."""
        return self.tokenizer[token_id]

    def save(self, directory: str):
        """Save tokenizer config to directory."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(Path(directory))

    @classmethod
    def load(cls, directory: str) -> "MidiTokWrapper":
        """Load tokenizer from saved directory."""
        tok = TSD.from_pretrained(Path(directory))
        return cls(tokenizer=tok)
