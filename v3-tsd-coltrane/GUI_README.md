# 🎷 Coltrain GUI

## Quick Start

**Double-click `Coltrain.command` on your Desktop** to launch the GUI!

## What It Does

- Generate jazz MIDI files using either:
  - **Learned Generator**: Uses patterns from your MIDI training data
  - **Strict Rules Generator**: Uses hard-coded jazz theory (32-bar AABA)

- Automatically converts to audio:
  - MIDI → WAV → MP3

- Saves output to timestamped folders on your Desktop:
  - `Desktop/Coltrain_YYYYMMDD_HHMMSS/`
  - Contains: `jazz.mid`, `jazz.wav`, `jazz.mp3`

## GUI Options

### Generator
- **Learned**: Probabilistic generation from your training MIDIs (varied output)
- **Strict**: Rule-based 32-bar AABA bebop (consistent output)

### Tempo
- Slider: 60-200 BPM
- Presets: Ballad (80), Swing (120), Bebop (160)

### Length (Learned only)
- Slider: 32-300 measures (~1-12 minutes)

### Key (Strict only)
- Choose from: C, D, E, F, G, A, B

## Output

Each time you click "Generate Jazz", a new folder appears on your Desktop:

```
Desktop/
  Coltrain_20260327_143022/
    ├── jazz.mid  (MIDI file)
    ├── jazz.wav  (Audio - lossless)
    └── jazz.mp3  (Audio - compressed)
```

The folder automatically opens when generation completes!

## Dependencies

All dependencies are pre-installed:
- ✅ Python environment with mido, midi2audio, pydub
- ✅ FluidSynth (MIDI → WAV conversion)
- ✅ ffmpeg (WAV → MP3 conversion)
- ✅ Soundfont (VintageDreamsWaves-v2.sf2)

## Command Line (Alternative)

If you prefer the command line:

```bash
cd /Users/dylangehl/coltrain
source venv/bin/activate
python coltrain_gui.py
```

Or use the original generators directly:

```bash
# Learned generator (6-minute piece)
python generate.py --measures 150 --tempo 120 --output jazz.mid

# Strict rules generator (32-bar AABA)
python generate_strict.py --key C --tempo 140 --output strict.mid

# Live infinite generation (requires MIDI output port)
python generate_live.py --tempo 120
```

## Troubleshooting

**GUI won't launch from Desktop:**
- Open Terminal and run: `chmod +x ~/Desktop/Coltrain.command`
- Or drag the .command file to Terminal

**Audio conversion fails:**
- Check if FluidSynth is installed: `which fluidsynth`
- Check if ffmpeg is installed: `which ffmpeg`
- Reinstall if needed: `brew install fluid-synth ffmpeg`

**MIDI generation fails:**
- Ensure training data exists: `ls /Users/dylangehl/augmented_dataset`
- Check permissions on output directory

## Files

- `coltrain_gui.py` - GUI application
- `generate.py` - Learned generator (main)
- `generate_strict.py` - Rule-based generator
- `generate_live.py` - Real-time infinite generation
- `soundfonts/default.sf2` - Soundfont for audio synthesis
- `Desktop/Coltrain.command` - Desktop launcher

---

**Enjoy making jazz! 🎺**
