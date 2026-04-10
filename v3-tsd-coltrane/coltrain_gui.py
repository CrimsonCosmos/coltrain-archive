#!/usr/bin/env python3
"""
Coltrain GUI - Simple interface for jazz generation
Double-click to run, select options, generate WAV + MP3
"""
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
from datetime import datetime
import threading

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))


class ColtrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎷 Coltrain - AI Jazz Generator")
        self.root.geometry("500x650")
        self.root.resizable(False, False)

        # Style
        style = ttk.Style()
        style.theme_use('aqua')  # Mac native look

        self.create_widgets()

    def create_widgets(self):
        # Header
        header = tk.Label(
            self.root,
            text="🎷 Coltrain",
            font=("Helvetica", 32, "bold"),
            fg="#1a1a1a"
        )
        header.pack(pady=20)

        subtitle = tk.Label(
            self.root,
            text="AI Jazz Generation",
            font=("Helvetica", 14),
            fg="#666666"
        )
        subtitle.pack(pady=5)

        # Main frame
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        # Generator selection
        ttk.Label(frame, text="Generator:", font=("Helvetica", 12, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=10
        )

        self.generator_var = tk.StringVar(value="learned")
        generators = [
            ("Learned (from your MIDIs)", "learned"),
            ("Strict Rules (32-bar AABA)", "strict"),
        ]

        for i, (label, value) in enumerate(generators):
            ttk.Radiobutton(
                frame,
                text=label,
                variable=self.generator_var,
                value=value
            ).grid(row=i+1, column=0, sticky=tk.W, padx=20)

        # Tempo
        ttk.Label(frame, text="Tempo (BPM):", font=("Helvetica", 12, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=(20, 5)
        )

        self.tempo_var = tk.IntVar(value=120)
        tempo_frame = ttk.Frame(frame)
        tempo_frame.grid(row=4, column=0, sticky=tk.W, padx=20)

        ttk.Scale(
            tempo_frame,
            from_=60,
            to=200,
            variable=self.tempo_var,
            orient=tk.HORIZONTAL,
            length=300
        ).pack(side=tk.LEFT)

        self.tempo_label = ttk.Label(tempo_frame, text="120")
        self.tempo_label.pack(side=tk.LEFT, padx=10)

        self.tempo_var.trace_add('write', self.update_tempo_label)

        # Tempo presets
        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=5, column=0, sticky=tk.W, padx=20, pady=5)

        ttk.Button(preset_frame, text="Ballad (80)", command=lambda: self.tempo_var.set(80)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Swing (120)", command=lambda: self.tempo_var.set(120)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Bebop (160)", command=lambda: self.tempo_var.set(160)).pack(side=tk.LEFT, padx=2)

        # Length (only for learned generator)
        ttk.Label(frame, text="Length:", font=("Helvetica", 12, "bold")).grid(
            row=6, column=0, sticky=tk.W, pady=(20, 5)
        )

        self.measures_var = tk.IntVar(value=150)
        length_frame = ttk.Frame(frame)
        length_frame.grid(row=7, column=0, sticky=tk.W, padx=20)

        ttk.Scale(
            length_frame,
            from_=32,
            to=300,
            variable=self.measures_var,
            orient=tk.HORIZONTAL,
            length=300
        ).pack(side=tk.LEFT)

        self.measures_label = ttk.Label(length_frame, text="150 measures (~6 min)")
        self.measures_label.pack(side=tk.LEFT, padx=10)

        self.measures_var.trace_add('write', self.update_measures_label)

        # Key (only for strict generator)
        ttk.Label(frame, text="Key (strict only):", font=("Helvetica", 12, "bold")).grid(
            row=8, column=0, sticky=tk.W, pady=(20, 5)
        )

        self.key_var = tk.StringVar(value="C")
        key_frame = ttk.Frame(frame)
        key_frame.grid(row=9, column=0, sticky=tk.W, padx=20)

        keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        for key in keys:
            ttk.Radiobutton(
                key_frame,
                text=key,
                variable=self.key_var,
                value=key
            ).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(frame, mode='indeterminate', length=400)
        self.progress.grid(row=10, column=0, pady=20)

        # Status label
        self.status_label = ttk.Label(frame, text="Ready to generate jazz!", font=("Helvetica", 10))
        self.status_label.grid(row=11, column=0, pady=5)

        # Generate button
        self.generate_btn = ttk.Button(
            frame,
            text="🎺 Generate Jazz",
            command=self.generate,
            style="Accent.TButton"
        )
        self.generate_btn.grid(row=12, column=0, pady=20)

        # Add accent button style
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 14, "bold"))

    def update_tempo_label(self, *args):
        tempo = self.tempo_var.get()
        if tempo < 90:
            style = "Ballad"
        elif tempo < 140:
            style = "Swing"
        else:
            style = "Bebop"
        self.tempo_label.config(text=f"{tempo} ({style})")

    def update_measures_label(self, *args):
        measures = self.measures_var.get()
        minutes = measures * 2 / 60  # Rough estimate
        self.measures_label.config(text=f"{measures} measures (~{minutes:.1f} min)")

    def generate(self):
        # Disable button during generation
        self.generate_btn.config(state='disabled')
        self.progress.start(10)
        self.status_label.config(text="Generating jazz...")

        # Run generation in background thread
        thread = threading.Thread(target=self.run_generation)
        thread.daemon = True
        thread.start()

    def run_generation(self):
        try:
            # Get parameters
            generator = self.generator_var.get()
            tempo = self.tempo_var.get()
            measures = self.measures_var.get()
            key = self.key_var.get()

            # Create output folder on desktop
            desktop = Path.home() / "Desktop"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = desktop / f"Coltrain_{timestamp}"
            output_folder.mkdir(exist_ok=True)

            # File names
            midi_file = output_folder / "jazz.mid"
            wav_file = output_folder / "jazz.wav"
            mp3_file = output_folder / "jazz.mp3"

            # Update status
            self.root.after(0, lambda: self.status_label.config(text="Generating MIDI..."))

            # Generate MIDI
            coltrain_dir = Path(__file__).parent

            if generator == "learned":
                cmd = [
                    sys.executable,
                    str(coltrain_dir / "generate.py"),
                    "--measures", str(measures),
                    "--tempo", str(tempo),
                    "--output", str(midi_file)
                ]
            else:  # strict
                cmd = [
                    sys.executable,
                    str(coltrain_dir / "generate_strict.py"),
                    "--key", key,
                    "--tempo", str(tempo),
                    "--output", str(midi_file)
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"Generation failed: {result.stderr}")

            # Convert MIDI → WAV
            self.root.after(0, lambda: self.status_label.config(text="Converting to WAV..."))
            self.midi_to_wav(midi_file, wav_file)

            # Convert WAV → MP3
            self.root.after(0, lambda: self.status_label.config(text="Converting to MP3..."))
            self.wav_to_mp3(wav_file, mp3_file)

            # Success!
            self.root.after(0, lambda: self.show_success(output_folder))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))
        finally:
            self.root.after(0, self.reset_ui)

    def midi_to_wav(self, midi_file, wav_file):
        """Convert MIDI to WAV using FluidSynth or timidity."""
        # Try FluidSynth first
        soundfont_path = Path(__file__).parent / "soundfonts" / "default.sf2"
        try:
            subprocess.run(
                ["fluidsynth", "-ni", "-g", "1", "-F", str(wav_file),
                 str(soundfont_path), str(midi_file)],
                check=True,
                capture_output=True
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try timidity
        try:
            subprocess.run(
                ["timidity", str(midi_file), "-Ow", "-o", str(wav_file)],
                check=True,
                capture_output=True
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # If both fail, use Python library
        try:
            import midi2audio
            fs = midi2audio.FluidSynth()
            fs.midi_to_audio(str(midi_file), str(wav_file))
        except Exception as e:
            raise Exception(
                "Could not convert MIDI to WAV. Please install FluidSynth:\n"
                "brew install fluid-synth"
            )

    def wav_to_mp3(self, wav_file, mp3_file):
        """Convert WAV to MP3 using ffmpeg or pydub."""
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(wav_file), "-codec:a", "libmp3lame",
                 "-qscale:a", "2", str(mp3_file), "-y"],
                check=True,
                capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(str(wav_file))
                audio.export(str(mp3_file), format="mp3", bitrate="192k")
            except Exception as e:
                raise Exception(
                    "Could not convert WAV to MP3. Please install ffmpeg:\n"
                    "brew install ffmpeg"
                )

    def reset_ui(self):
        self.progress.stop()
        self.generate_btn.config(state='normal')
        self.status_label.config(text="Ready to generate jazz!")

    def show_success(self, output_folder):
        messagebox.showinfo(
            "Success! 🎷",
            f"Jazz generated successfully!\n\n"
            f"Folder: {output_folder.name}\n"
            f"Files: jazz.mid, jazz.wav, jazz.mp3\n\n"
            f"Opening folder..."
        )
        # Open folder in Finder
        subprocess.run(["open", str(output_folder)])

    def show_error(self, error):
        messagebox.showerror(
            "Error",
            f"Generation failed:\n\n{error}"
        )


def main():
    root = tk.Tk()
    app = ColtrainGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
