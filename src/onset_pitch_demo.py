"""
Demo: Accurate note and onset detection using onset detection + YIN pitch estimation.
"""

import sys
import librosa
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python onset_pitch_demo.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

y, sr = librosa.load(audio_path)

# Detect onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

notes = []
window_size = int(0.1 * sr)  # 100ms window for pitch estimation
for onset_time in onset_times:
    onset_sample = int(onset_time * sr)
    start = max(0, onset_sample - window_size // 2)
    end = min(len(y), onset_sample + window_size // 2)
    segment = y[start:end]
    if len(segment) < window_size // 2:
        continue
    try:
        f0 = librosa.yin(segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        # Take median pitch in window
        pitch_hz = np.nanmedian(f0)
        if np.isnan(pitch_hz):
            continue
        midi_note = int(round(librosa.hz_to_midi(pitch_hz)))
        note_name = librosa.midi_to_note(midi_note)
        notes.append({'onset': onset_time, 'note': midi_note, 'name': note_name})
    except Exception as e:
        continue

print(f"Detected {len(notes)} notes:")
for i, note in enumerate(notes[:20], 1):
    print(f"{i}. {note['name']} (MIDI {note['note']}) at {note['onset']:.2f}s")
if len(notes) > 20:
    print(f"... and {len(notes) - 20} more notes")
