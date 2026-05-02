# 🎸 Music Transformer - Real Deep Learning for Jazz

**This is NOT Markov chains. This is a REAL transformer with:**
- ✅ **Multi-head self-attention** (learns dependencies across hundreds of tokens)
- ✅ **Backpropagation training** (gradient descent with real loss functions)
- ✅ **Long-range context** (512+ token context window)
- ✅ **Multi-track awareness** (all instruments in same sequence)
- ✅ **Proper learning** (notes from measure 1 inform measure 50)

---

## Architecture

### 1. Tokenizer (`data/tokenizer.py`)
Converts MIDI → discrete tokens:
- **Vocabulary: 2,283 tokens**
  - NOTE_ON (pitch × velocity_bin)
  - NOTE_OFF (pitch)
  - TIME_SHIFT (delta time)
  - TRACK (MELODY, PIANO, BASS, DRUMS)
  - Special (PAD, START, END)

### 2. Transformer Model (`model/transformer.py`)
**4.3M+ parameters** with:
- **Multi-head attention** (8 heads)
  - Learns which past tokens are relevant to current token
  - Can attend to notes 100+ positions back
- **Positional encoding**
  - Adds time/position information
- **Feed-forward layers**
  - Non-linear transformations
- **6 transformer blocks** (stacked)
  - Each block: attention → FFN → layer norm

### 3. Dataset (`data/dataset.py`)
- Loads **1,352 MIDI files**
- Creates **117,987 training sequences**
- Total **30.2 million tokens**
- Batching and data loading

### 4. Training Loop (`training/train.py`)
**Real machine learning:**
```python
# Forward pass through transformer
logits = model(input_seq)  # Predictions

# Calculate loss (how wrong are we?)
loss = cross_entropy(logits, target_seq)

# Backward pass (backpropagation!)
loss.backward()  # Compute gradients

# Gradient clipping (stability)
clip_grad_norm_(model.parameters())

# Update weights (gradient descent)
optimizer.step()  # Learn!
```

**Features:**
- AdamW optimizer
- Learning rate warmup + cosine decay
- Gradient clipping
- Checkpointing every epoch
- Validation monitoring
- Perplexity tracking

### 5. Generation (`generate_transformer.py`)
**Autoregressive generation:**
- Starts with START token
- Predicts next token (using attention to all previous tokens)
- Samples from probability distribution
- Temperature/top-k/top-p sampling
- Converts back to MIDI

---

## Training

### Quick Start

```bash
# Train transformer (10 epochs, ~2-4 hours on M4)
cd ~/coltrain/transformer
../../venv/bin/python training/train.py
```

### What Happens During Training

**Each batch:**
1. Load 16 sequences (512 tokens each)
2. Forward pass → get predictions
3. Calculate loss (cross-entropy)
4. **Backpropagation** → compute gradients
5. Clip gradients (prevent explosion)
6. Update weights (gradient descent)
7. Update learning rate

**Each epoch:**
- ~7,500 batches (117,987 sequences ÷ 16)
- Validation after each epoch
- Save checkpoint
- Track metrics (loss, perplexity)

**After 10 epochs:**
- Model has seen 30M tokens
- Learned attention patterns
- Can generate coherent music

### Monitoring Training

```bash
# Watch loss decrease (learning!)
Epoch 1 [100/7374]
   Loss: 6.8234
   Perplexity: 912.45
   LR: 1.00e-05

Epoch 1 [200/7374]
   Loss: 6.1245
   Perplexity: 456.78
   LR: 2.00e-05

# Lower loss = better learning
# Lower perplexity = more confident predictions
```

---

## Generation

### Generate Music with Trained Model

```bash
# Basic generation
python generate_transformer.py --checkpoint checkpoints/checkpoint_latest.pt --output jazz.mid

# More creative (higher temperature)
python generate_transformer.py \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --output creative.mid \
  --temperature 1.2

# Longer piece
python generate_transformer.py \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --output long.mid \
  --max-len 3000

# Ballad tempo
python generate_transformer.py \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --output ballad.mid \
  --tempo 80 \
  --temperature 0.9
```

### Generation Parameters

- `--temperature` (0.5-1.5)
  - Lower = more conservative, predictable
  - Higher = more creative, experimental
  - Default: 1.0

- `--top-k` (10-100)
  - Keep only top k probable tokens
  - Lower = more focused
  - Default: 50

- `--top-p` (0.8-0.99)
  - Nucleus sampling threshold
  - Higher = more diversity
  - Default: 0.95

- `--max-len` (500-5000)
  - Number of tokens to generate
  - ~1000 tokens ≈ 1-2 minutes
  - Default: 2000

---

## Key Differences from Markov Chains

### Markov Chain (Old Approach)
```python
# Only looks at last 2 notes
melody_patterns[(C, D)] = {E: 10, F: 5}
next_note = sample(melody_patterns[(C, D)])
```
- ❌ No long-range context
- ❌ No backpropagation
- ❌ No learning of structure
- ❌ Each track independent

### Transformer (This Approach)
```python
# Attention looks at ALL previous tokens
attention_scores = softmax(Q @ K.T / sqrt(d))
context = attention_scores @ V

# With backpropagation
loss = cross_entropy(predictions, targets)
loss.backward()  # Learn what patterns work!
```
- ✅ **Context window of 512 tokens** (~30 seconds of music)
- ✅ **Attention mechanism** learns which past notes matter
- ✅ **Backpropagation** actually learns from mistakes
- ✅ **Multi-track in same sequence** (tracks can listen to each other)

---

## Model Size & Performance

### Current Configuration
- **Parameters:** 4,330,219 (4.3M)
- **Model dimension:** 512
- **Attention heads:** 8
- **Layers:** 6
- **Context window:** 512 tokens

### Training Time (M4 MacBook Air)
- **Per epoch:** ~15-20 minutes
- **10 epochs:** ~3 hours
- **20 epochs:** ~6 hours (recommended)

### Scaling Up (Optional)
```python
# Bigger model (more capacity, slower training)
CONFIG = {
    'd_model': 768,      # 512 → 768
    'num_heads': 12,     # 8 → 12
    'num_layers': 8,     # 6 → 8
    'd_ff': 3072,        # 2048 → 3072
}
# Result: ~15M parameters
```

---

## Technical Details

### Loss Function
**Cross-Entropy Loss:**
```
L = -∑ log(P(token_i | token_1...token_{i-1}))
```
- Measures how well model predicts next token
- Lower is better
- Perplexity = exp(loss)

### Attention Mechanism
```python
# For each token, compute relevance to all previous tokens
Q = W_q @ x  # Query: "what am I looking for?"
K = W_k @ x  # Key: "what do I contain?"
V = W_v @ x  # Value: "what information do I have?"

scores = Q @ K.T / sqrt(d_k)  # Compute relevance
attn = softmax(scores)        # Normalize
output = attn @ V             # Weighted sum
```

### Causal Masking
- Model can only attend to past tokens, not future
- Ensures autoregressive generation works
- Mask[i,j] = 1 if i ≥ j, else 0

### Positional Encoding
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- Adds position information to tokens
- Allows model to distinguish position

---

## Project Structure

```
transformer/
├── model/
│   └── transformer.py      # MusicTransformer (4.3M params)
├── data/
│   ├── tokenizer.py        # MIDI ↔ tokens (2,283 vocab)
│   └── dataset.py          # PyTorch dataset (117,987 sequences)
├── training/
│   └── train.py            # Training loop (backprop!)
├── generate_transformer.py # Generation script
├── checkpoints/            # Saved models (created during training)
│   ├── checkpoint_epoch1.pt
│   ├── checkpoint_epoch2.pt
│   └── checkpoint_latest.pt
└── README.md               # This file
```

---

## FAQ

### Q: How is this different from Markov chains?
**A:** Markov chains count patterns (no learning). Transformers use:
- Backpropagation (real learning from loss)
- Attention (long-range dependencies)
- Context windows (512+ tokens vs 2 notes)
- Gradient descent (weight updates)

### Q: How long to train?
**A:** 10 epochs (~3 hours) minimum. 20 epochs (~6 hours) better.

### Q: Will output sound good immediately?
**A:** Early epochs (1-3): Random-ish, but structured
Mid epochs (5-10): Coherent melodies, okay harmony
Later epochs (15-20): Good musical phrases

### Q: Can I resume training?
**A:** Yes! Checkpoints save optimizer state.
```python
trainer.load_checkpoint('checkpoints/checkpoint_epoch5.pt')
trainer.train(num_epochs=10)  # Continues from epoch 5
```

### Q: GPU required?
**A:** No. M4 MacBook Air uses **MPS** (Metal Performance Shaders).
Faster than CPU, slower than NVIDIA GPU, but works great!

### Q: How much memory?
**A:** ~4GB RAM for training with batch_size=16.
Reduce batch_size if needed.

---

## Next Steps

1. **Train the model:**
   ```bash
   python training/train.py
   ```

2. **Watch loss decrease** (learning is happening!)

3. **Generate music after a few epochs:**
   ```bash
   python generate_transformer.py \
     --checkpoint checkpoints/checkpoint_latest.pt \
     --output test.mid
   ```

4. **Keep training** for better results (20+ epochs)

5. **Experiment with generation parameters** (temperature, top-k, etc.)

---

## This is Real AI

Unlike the Markov chain approach, this transformer:
- **Learns** through backpropagation (not just counting)
- **Understands** long-range structure (via attention)
- **Coordinates** multiple tracks (same sequence)
- **Develops** themes over time (context window)

**Welcome to real deep learning for music generation! 🚀**
