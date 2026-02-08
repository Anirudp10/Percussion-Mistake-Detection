"""
Demo: Pitch detection using librosa's YIN algorithm.
"""

import sys
import librosa
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python yin_pitch_demo.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

y, sr = librosa.load(audio_path)

# Use YIN to estimate pitch (Hz) for each frame
f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

# Convert Hz to MIDI note numbers (NaN for unvoiced frames)
midi_notes = librosa.hz_to_midi(f0)

# Print first 20 detected pitches (skip NaN)
print("First 20 detected pitches (MIDI numbers):")
count = 0
for i, midi_note in enumerate(midi_notes):
    if not np.isnan(midi_note):
        print(f"Frame {i}: MIDI {int(midi_note)}")
        count += 1
    if count >= 20:
        break
