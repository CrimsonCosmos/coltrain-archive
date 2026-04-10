# 🎷 Coltrain Quick Start

One command to generate jazz with learned rhythms and harmonic coordination.

---

## ⚡️ Generate Jazz Now

```bash
cd /Users/dylangehl/coltrain
source venv/bin/activate
python generate.py --measures 150 --tempo 120
open coltrain_jazz.mid
```

Done! 🎺

---

## 🎛️ Common Commands

```bash
# 6-minute standard jazz
python generate.py -m 150 -t 120 -o standard.mid

# Fast bebop
python generate.py -m 150 -t 160 -o bebop.mid

# Slow ballad
python generate.py -m 150 -t 90 -o ballad.mid

# Generate 3 variations (pick the best)
python generate.py -m 150 -o jazz1.mid
python generate.py -m 150 -o jazz2.mid
python generate.py -m 150 -o jazz3.mid
```

---

## 🎹 What You Get

**4 tracks, all harmonically coordinated:**
1. 🎤 **Melody** - Sax/trumpet with swing rhythms and rests
2. 🎹 **Piano** - Chord comping with syncopation
3. 🎸 **Bass** - Walking bass (4 notes per measure)
4. 🥁 **Drums** - Swing pattern with ride/hi-hat variations

---

## 📊 What It Learned

From your 100 jazz MIDI files:
- **95 chord progressions** (real jazz sequences)
- **541 melody patterns** (that fit chords)
- **10,529 melody rhythms** ⭐ (swing, syncopation, rests)
- **4,523 piano rhythms** (comp patterns)
- **2,399 bass rhythms** (walking patterns)

---

## 🎯 Options

| Flag | What | Example |
|------|------|---------|
| `-m` | Measures (length) | `-m 150` (6 min) |
| `-t` | Tempo (speed) | `-t 160` (bebop) |
| `-o` | Output file | `-o my_jazz.mid` |

---

## 💡 Pro Tips

1. **Generate 3-5 variations** - Pick the best one
2. **Use augmented dataset** - More patterns (default)
3. **Import to DAW** - Add reverb, adjust instruments
4. **Vary tempo** - Try 90, 120, 160 BPM

---

## 🚀 What Changed (v2.0)

### Before (3 generators):
- ❌ Confusing (which one to use?)
- ❌ Fixed rhythms (all notes same length)
- ❌ Limited coordination

### Now (1 generator):
- ✅ One command, simple
- ✅ **10,000+ learned rhythm patterns**
- ✅ Full harmonic coordination
- ✅ Melody, piano, bass, drums with varied rhythms

---

**That's it! Generate some jazz** 🎺

```bash
python generate.py -m 150 -t 120
```
