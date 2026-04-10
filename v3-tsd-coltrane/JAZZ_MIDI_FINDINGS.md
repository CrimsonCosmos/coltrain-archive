# Jazz MIDI Generation — Findings & Knowledge Base

Accumulated knowledge from the Coltrain project: training Music Transformers
to generate multi-track jazz MIDI.

---

## Model Architecture

### What works
- **Decoder-only transformer** with learned positional embeddings + SDPA attention
- **Pre-norm blocks** (LayerNorm before attention/FFN) — more stable than post-norm
- **GELU activation** in FFN — smoother gradients than ReLU for music
- **Weight tying** (embedding ↔ output projection) — reduces params, helps generalization

### Current model
- 26M params: d_model=512, 8 layers, 8 heads, d_ff=2048, seq_len=1024
- Vocab: 431 tokens (MidiTok TSD + 4 role delimiters)

### Open questions
- Is 26M params enough? Would 50-100M improve musical coherence?
- Would longer context (2048+) help with song structure/form?
- Would relative position encoding (e.g., RoPE, ALiBi) help with musical repetition/structure?

---

## Tokenization

### MidiTok TSD (Time Shift Duration) — current choice
- Vocab size: 427 base tokens
- Token types: Pitch (89), Velocity (16), Duration (64), TimeShift (64), PitchDrum (62), Program (129), PAD/BOS/EOS (3)
- Single interleaved stream with Program tokens for multi-instrument
- Explicit duration tokens (no NOTE_OFF ambiguity)

### Track-sequential format (v3)
- Extends TSD with 4 role delimiter tokens (427 → 431)
- Format: [BOS][LEAD]...[COMPING]...[BASS]...[DRUMS]...[EOS]
- Each section has its own timeline (TimeShifts from beat 0)
- Enables cascaded generation: each instrument model sees all previous tracks
- Typical jazz piece = 800-2000 tokens in this format

### Failed: Compound token approach (v1)
- 4,711 token vocab, compound MIDI tokens
- Too large a vocab for a small model to learn effectively

### Lesson: tokenizer choice matters enormously
- Switching from compound (4,711) to TSD (427) was a bigger improvement than any architecture change

---

## Training

### Goodhart's Law: loss ↓ ≠ quality ↑
**This is our single most important finding.** Validation loss improving does NOT mean musical quality is improving. We have observed this multiple times:

- v1: Weighted loss on NOTE_ON tokens (3x) — loss improved, music became mechanical and repetitive
- v2: Training 30 epochs — loss steadily decreased from 1.70 to 1.53, but epoch 19 output was a 14-note repeating loop stuck on one instrument

**Implications:**
- Keep training short: 5-10 epochs max
- Must listen to outputs — metrics alone are unreliable
- Generate samples from multiple checkpoints and compare by ear
- Early stopping based on human evaluation, not just val loss

### Specialist fine-tuning (v3)
- Phase 1: Shared pretraining on all tracks (5 epochs, lr=3e-4)
- Phase 2: Per-role specialist with masked loss (5 epochs, lr=5e-5)
- Each specialist sees the FULL sequence but only computes loss on its section
- On-the-fly mask computation (vectorized cumsum) avoids precompute/shuffle bugs

### Label smoothing
- Using 0.1 label smoothing — helps prevent overconfident predictions
- May contribute to more "creative" generation (less peaked distributions)

### Learning rate schedule
- Cosine decay with linear warmup (500-1000 steps)
- Specialist fine-tuning uses 6x lower base LR than shared pretraining

---

## Generation Quality Issues

### Repetition loops (the #1 problem)
- Models easily fall into repeating the same 4-16 note pattern indefinitely
- Worse with lower temperature, longer generation, and more training epochs
- **Mitigations that help:**
  - Token-level repetition penalty (divide logits by 1.2 for recent tokens)
  - N-gram blocking (penalize 4-grams that appeared in last 100 tokens)
  - Higher temperature (0.9-1.0 vs 0.7-0.8)
  - Fewer training epochs
- **Mitigations that DON'T help:**
  - Just lowering temperature (makes repetition worse)
  - Training longer (makes repetition worse)

### Instrument collapse
- v2 single-model approach collapsed to solo piano despite multi-track training data
- Root cause: single 26M model can't coordinate multiple instruments simultaneously
- **Fix:** v3 multi-model architecture with track-sequential tokenization
- Each specialist only needs to master one instrument's patterns

### Lack of musical structure
- Generated music often lacks verse/chorus/AABA form
- Likely needs longer context (current 1024 tokens ≈ 30-60 seconds)
- May need explicit structural tokens or conditioning

---

## Dataset

### Jazz corpus
- 1,876 original jazz MIDI songs
- 24,302 files after augmentation (transposition: ±6 semitones)
- Genres: standards, bebop, modal jazz, fusion, Latin jazz

### Augmentation
- Pitch transposition only (no tempo/velocity augmentation)
- Song-group-aware train/val split prevents leakage from transpositions

### Track-sequential dataset (v3)
- 625,232 sequences (seq_len=1024, stride=512)
- 18,915 files with ≥2 roles (5,386 skipped for having <2 roles)
- Role coverage: lead=100%, comping=68%, bass=94%, drums=95%
- 1.19 GB as uint16 numpy

### Role assignment heuristic
- If horns/reeds (programs 56-79) exist → lead=horns, comping=piano+guitar
- If no horns → lead=piano or guitar, comping=empty
- Bass = programs 32-39, Drums = program -1 (percussion)
- Works well for standard jazz combos (quartet, quintet, trio)

---

## Hardware & Infrastructure

### NVIDIA L4 (current — Ada Lovelace, sm89)
- 24GB VRAM, native FP16 + BF16 support
- ~20 min/epoch for our model at batch_size=96
- BF16 preferred (no GradScaler needed, same exponent range as FP32)
- ~$0.70/hr on GCP (g2-standard-4)

### NVIDIA T4 (previous — Turing, sm75)
- 16GB VRAM, FP16 only (BF16 silently falls back to FP32!)
- ~30 min/epoch at batch_size=64
- Must use FP16 + GradScaler
- ~$0.54/hr on GCP but slower, so similar total cost to L4

### Apple M4 (local dev)
- MPS backend, 16GB unified memory
- batch_size=8 max at seq_len=1024 (OOM at 32+)
- Good for quick tests, too slow/small for real training
- Watch out for mmap swap thrashing with large datasets

### GCP tips
- GPU availability is extremely limited — scan many zones
- On-demand instances won't get preempted (spot instances will)
- Always set up auto-shutdown to prevent billing surprises
- Deep learning VM images come with PyTorch + CUDA pre-installed

---

## Training Results Log

| Version | Epochs | Val Loss | Musical Quality | Notes |
|---------|--------|----------|----------------|-------|
| v1 (compound) | ~40 | ~2.5 | Poor | Wrong tokenizer, too-large vocab |
| v2 (TSD, single model) | 19/30 | 1.53 | Bad | Solo piano collapse, repetition loops |
| v3 shared | 5 | 1.78 | Decent | 4 distinct tracks, some wrong programs |
| v3 lead specialist | 5 | 1.69 | TBD | |
| v3 comping specialist | 5 | 1.38 | TBD | |
| v3 bass specialist | 5 | 1.42 | TBD | |
| v3 drums specialist | 5 | TBD | TBD | |

---

## Next Steps / Ideas to Explore

- [ ] Listen to v3 specialist outputs and evaluate musical quality
- [ ] Try 3 specialist epochs vs 5 — is less better?
- [ ] Experiment with sampling parameters (temperature, top-k, top-p, repetition penalty)
- [ ] Try longer context (2048) if VRAM allows
- [ ] Condition generation on chord changes or song structure
- [ ] Evaluate with musicians — get subjective quality ratings
- [ ] Try RoPE or ALiBi for better handling of musical repetition/position
- [ ] Larger model (50-100M params) — would it help or just overfit faster?
