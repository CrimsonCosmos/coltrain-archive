# Coltrain Archive

**Archived.** This repo preserves the Music Transformer approach to jazz MIDI generation. Superseded by [coltrain-by-theory](https://github.com/CrimsonCosmos/coltrain-by-theory), which uses pure music theory rules instead of neural networks.

## What's Here

### `v1-compound-tokenizer/`
The original Coltrain: 9M-parameter Music Transformer with compound tokenization (vocab 4,711). Trained on 1,876 jazz MIDI files. Produced solo piano output despite multi-track training data. Val loss went down but musical quality degraded after ~10 epochs (Goodhart's Law).

### `v3-tsd-coltrane/`
Coltrain v3.1: 26M-parameter transformer with MidiTok TSD tokenization (vocab 457), cascaded 4-track generation (lead/comping/bass/drums), and Coltrane-specific features:
- Theory-guided sampling (chord/scale/avoid note biases)
- Beat grid with form templates (Giant Steps, blues, AABA)
- Tension curves (exposition/development/climax/resolution)
- Motivic memory and digital pattern matching
- Key center awareness for multi-tonic systems
- Harmonic superimposition (Coltrane matrix)

Reached Jazz Score 0.926 but rhythms never sounded like real jazz. 17 competing soft biases couldn't override the model's learned distribution.

## Why We Moved On

The transformer approach hit a wall: **rhythm in jazz is structural, not probabilistic.** Walking bass is quarter notes. Comping hits beats 2 and 4. Drums play ride + snare + kick in fixed patterns. A neural network token predictor is the wrong tool for these deterministic patterns. Rule-based music theory produces better rhythm from day one.
