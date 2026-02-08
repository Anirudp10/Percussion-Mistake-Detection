"""
Module for summarizing MIDI file content using pretty_midi.
"""

import pretty_midi

class MIDISummary:
    @staticmethod
    def print_summary(midi_path: str, max_notes: int = 10):
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            for inst in midi.instruments:
                for note in inst.notes:
                    notes.append({'note': note.pitch, 'onset': note.start})
            tempo = midi.get_tempo_changes()[1][0] if len(midi.get_tempo_changes()[1]) > 0 else None
            print(f"✓ Detected {len(notes)} notes")
            if tempo:
                print(f"✓ Estimated tempo: {tempo:.1f} BPM")
            if notes:
                print(f"\nFirst few notes detected:")
                for i, note in enumerate(notes[:max_notes], 1):
                    try:
                        import librosa
                        note_name = librosa.midi_to_note(note['note'])
                        print(f"  {i}. {note_name} (MIDI {note['note']}) at {note['onset']:.3f}s")
                    except:
                        print(f"  {i}. MIDI {note['note']} at {note['onset']:.3f}s")
                if len(notes) > max_notes:
                    print(f"  ... and {len(notes) - max_notes} more notes")
        except Exception as e:
            print(f"✗ Error reading MIDI: {str(e)}")
