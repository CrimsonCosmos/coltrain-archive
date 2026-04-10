# 🎷 Coltrain

**AI-powered multi-track jazz generation with harmonic coordination and learned rhythm**

Named after John Coltrane, the legendary jazz saxophonist, and the AI training that powers it.

---

## 🎵 What is Coltrain?

Coltrain generates complete jazz compositions with multiple coordinated instruments:
- 🎤 Melody (saxophone/trumpet) with swing rhythms
- 🎹 Piano (chord comping) with syncopation
- 🎸 Walking bass with quarter note patterns
- 🥁 Swing drums with varied patterns

**Key Features:**
- ✅ **Harmonic coordination** - All tracks follow the same chord progression
- ✅ **Learned rhythms** - 10,000+ rhythm patterns from your MIDI files
- ✅ **Chord-aware melody** - Notes fit the current chord
- ✅ **Walking bass** - Follows chord roots with chromatic approaches
- ✅ **Varied drum patterns** - Ride cymbal, hi-hat variations

---

## 🚀 Quick Start

```bash
cd /Users/dylangehl/coltrain
source venv/bin/activate

# Generate 6-minute jazz piece
python generate.py --measures 150 --tempo 120

# Open in your DAW
open coltrain_jazz.mid
```

---

## 📖 Usage

### Basic Command

```bash
python generate.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--measures, -m` | Number of measures | 64 (~3 min) |
| `--tempo, -t` | BPM | 120 |
| `--output, -o` | Output filename | `coltrain_jazz.mid` |
| `--midi-dir` | Training data directory | augmented_dataset |
| `--order` | Markov chain order | 2 |

### Examples

**Standard Jazz (6 minutes):**
```bash
python generate.py --measures 150 --tempo 120 --output standard.mid
```

**Fast Bebop:**
```bash
python generate.py --measures 150 --tempo 160 --output bebop.mid
```

**Slow Ballad:**
```bash
python generate.py --measures 150 --tempo 90 --output ballad.mid
```

**Use Original Dataset:**
```bash
python generate.py --midi-dir /Users/dylangehl/midi_dataset
```

---

## 🎛️ Tempo Guide

- **70-90 BPM:** Ballads, slow jazz
- **100-130 BPM:** Medium swing, standard
- **140-180 BPM:** Bebop, fast jazz

## 📏 Measures Guide

- **32 measures:** ~2 minutes
- **64 measures:** ~3 minutes
- **128 measures:** ~5 minutes
- **150 measures:** ~6 minutes

---

## 📊 What Coltrain Learns

From 100 jazz MIDI files, Coltrain learns:

- **95 chord progressions** - Real jazz sequences (ii-V-I, blues, etc.)
- **541 melody patterns** - Note sequences that fit chords
- **10,529 melody rhythms** - Swing eighths, syncopation, rests
- **4,523 piano rhythms** - Comp patterns and voicings
- **2,399 bass rhythms** - Walking patterns and phrasing

---

## 🎓 How It Works

```
1. ANALYZE your jazz MIDI files
   ↓
2. LEARN chord progressions
   ↓
3. LEARN melody patterns (constrained to scales)
   ↓
4. LEARN rhythm patterns (note durations + rests)
   ↓
5. GENERATE:
   • Chord progression (from learned sequences)
   • Melody (fits chords + uses learned rhythms)
   • Piano (comps on chords + varied rhythms)
   • Bass (walks through roots + learned patterns)
   • Drums (swing patterns + variations)
```

**Key Innovation:** Harmonic coordination + rhythm variation = realistic jazz

---

## 📂 Project Structure

```
coltrain/
├── generate.py                  ← Main generator (run this!)
├── examples/
│   ├── coltrain_6min.mid       ← Latest output (6 min)
│   └── example_output.mid      ← Reference output
├── docs/
│   └── FINAL_SUMMARY.md        ← Detailed documentation
├── venv/                       ← Python environment
└── README.md                   ← This file
```

---

## 🆚 What Makes Coltrain Different?

| Feature | Coltrain | Simple Generators |
|---------|----------|-------------------|
| **Harmony** | ✅ All tracks coordinated | ❌ Random dissonance |
| **Rhythm** | ✅ 10,000+ learned patterns | ❌ Fixed durations |
| **Melody** | ✅ Fits chord progressions | ❌ Random notes |
| **Bass** | ✅ Walks through chords | ❌ Random bass notes |
| **Drums** | ✅ Varied swing patterns | ❌ Repetitive |
| **Learning** | ✅ From YOUR jazz files | ❌ No learning |

---

## 🎨 Post-Processing

### In GarageBand/Logic/Ableton:

1. **Import MIDI** → Drag `coltrain_jazz.mid` into your DAW

2. **Assign Instruments:**
   - Track 1 (Melody) → Jazz Saxophone or Trumpet
   - Track 2 (Piano) → Jazz Piano (with slight reverb)
   - Track 3 (Bass) → Upright Bass or Electric Bass
   - Track 4 (Drums) → Jazz Kit (brushes optional)

3. **Add Effects:**
   - Reverb (room/hall for jazz club feel)
   - Compression (subtle, to glue tracks together)
   - EQ (carve space for each instrument)

4. **Humanize:**
   - Adjust velocities for dynamics
   - Slight timing variations (1-5ms)
   - Add crescendos/decrescendos

5. **Mix:**
   - Melody: -3dB (lead but not overpowering)
   - Piano: -6dB (supportive comping)
   - Bass: -4dB (present but not boomy)
   - Drums: -5dB (keep time, don't dominate)

---

## 🔧 Requirements

- **Python:** 3.10+ (tested with 3.12)
- **Platform:** macOS (M4 optimized), Linux, Windows
- **Dependencies:** mido (installed in venv)
- **Training Data:** MIDI files in `midi_dataset` or `augmented_dataset`

---

## 💡 Best Practices

1. **Generate 3-5 variations** and pick the best one
2. **Vary the tempo** to find the right feel
3. **Use augmented dataset** for more variety (2x patterns)
4. **Post-process in DAW** for professional sound
5. **Combine multiple outputs** for longer compositions
6. **Experiment with order** (--order 3 for closer to training data)

---

## 🚧 Current Limitations

Coltrain is practical but has constraints:
- Simplified chord detection (uses lowest note as root)
- No long-term form structure (no AABA, theme/variation)
- Basic swing quantization
- Limited dynamics (fixed velocities - adjust in DAW)

**For better results:**
- Generate multiple variations, cherry-pick best
- Use as starting point, refine in DAW
- Manually adjust dynamics and expression

---

## 🔬 Training Data

Coltrain works with two datasets:

### Original Dataset
- **Location:** `/Users/dylangehl/midi_dataset`
- **Files:** 104 jazz MIDI standards
- **Use when:** You want patterns closer to specific songs

### Augmented Dataset (Recommended)
- **Location:** `/Users/dylangehl/augmented_dataset`
- **Files:** 1,353 (104 × 13 transpositions)
- **Benefits:** 2x more patterns, all 12 keys, lower overfitting
- **Use when:** You want more variety and generalization

---

## 📚 Research Context

Jazz generation remains challenging in AI:
- **Solved:** Chord progressions, basic patterns
- **Unsolved:** Emotional depth, improvisational authenticity, creative risk-taking

Coltrain's approach:
1. Learn chord progressions from real jazz
2. Generate melody that fits chords
3. Use learned rhythm patterns (not fixed)
4. Coordinate all tracks harmonically

Result: Technically coherent jazz that maintains harmonic and rhythmic interest.

---

## 🎯 Example Session

```bash
cd /Users/dylangehl/coltrain
source venv/bin/activate

# Generate 3 variations
python generate.py -m 150 -t 120 -o jazz1.mid
python generate.py -m 150 -t 120 -o jazz2.mid
python generate.py -m 150 -t 120 -o jazz3.mid

# Listen to all three
open jazz1.mid jazz2.mid jazz3.mid

# Pick the best one, import to Logic/Ableton
# Add effects, adjust instruments, export!
```

---

## 🎷 Credits

**Inspired by:**
- John Coltrane (namesake & musical inspiration)
- Your 104 jazz MIDI training files
- Google Magenta research
- The quest for AI jazz generation

**Built with:**
- Python 3.12
- mido (MIDI processing)
- Markov chains + chord-aware generation
- Mac M4 / Metal Performance Shaders

---

## 📝 Version History

- **v2.0** (Current) - Single unified generator with learned rhythms
- **v1.0** - Multiple generators (merged into one)

---

## 🤝 Contributing

This is a personal project, but feel free to:
- Fork and modify for your needs
- Use the generated MIDI in your music
- Share improvements or findings

---

## 📄 License

Use freely for personal, educational, or commercial purposes.

---

**Generate some jazz! 🎺**

```bash
python generate.py --measures 150 --tempo 120
```

---

## 🆘 Troubleshooting

**No training data found:**
```bash
ls /Users/dylangehl/augmented_dataset/*.mid
# If empty, check /Users/dylangehl/midi_dataset
```

**Generated music sounds random:**
- Try using augmented dataset (more patterns)
- Generate multiple variations, pick best
- Increase --order to 3 for closer to training data

**Rhythm sounds repetitive:**
- Use augmented dataset (more rhythm patterns)
- The generator learned 10,000+ patterns, should vary

**Tracks don't sound coordinated:**
- This should not happen - all tracks follow same chords
- Try regenerating, might be unlucky chord progression

---

**Need help?** Check `docs/FINAL_SUMMARY.md` for detailed documentation.
