"""
Demo: Group detected pitches into notes with onset and duration using librosa's YIN.
"""

import sys
import librosa
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python yin_note_grouping_demo.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

y, sr = librosa.load(audio_path)

# Use YIN to estimate pitch (Hz) for each frame
f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
midi_notes = librosa.hz_to_midi(f0)
times = librosa.frames_to_time(np.arange(len(midi_notes)), sr=sr)

notes = []
current_note = None
for i, midi_note in enumerate(midi_notes):
    if np.isnan(midi_note):
        if current_note is not None:
            # End current note
            current_note['end'] = times[i]
            notes.append(current_note)
            current_note = None
        continue
    if current_note is None:
        # Start a new note
        current_note = {'note': int(round(midi_note)), 'start': times[i], 'end': None}
    elif abs(midi_note - current_note['note']) > 0.5:
        # Pitch changed, end previous note and start new
        current_note['end'] = times[i]
        notes.append(current_note)
        current_note = {'note': int(round(midi_note)), 'start': times[i], 'end': None}
# If a note is still open at the end
if current_note is not None:
    current_note['end'] = times[-1]
    notes.append(current_note)

print(f"Detected {len(notes)} notes:")
for i, note in enumerate(notes[:20], 1):
    duration = note['end'] - note['start']
    try:
        note_name = librosa.midi_to_note(note['note'])
    except:
        note_name = f"MIDI {note['note']}"
    print(f"{i}. {note_name} (MIDI {note['note']}) start={note['start']:.2f}s duration={duration:.2f}s")
if len(notes) > 20:
    print(f"... and {len(notes) - 20} more notes")
