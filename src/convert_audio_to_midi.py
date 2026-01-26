# ...existing code from your old convert_audio_to_midi.py...
"""
Standalone script to convert audio files to MIDI format.
Usage: python convert_audio_to_midi.py <audio_file> [output_midi_file] [tempo]
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import librosa
import pretty_midi


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_audio_to_midi.py <audio_file> [output_midi_file] [tempo]")
        print("\nExamples:")
        print("  python convert_audio_to_midi.py audio.mp3")
        print("  python convert_audio_to_midi.py audio.mp3 output.mid")
        print("  python convert_audio_to_midi.py audio.mp3 output.mid 120")
        print("\nThe output MIDI file will be auto-generated if not specified.")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    tempo = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    try:
        # Basic librosa + pretty_midi transcription
        y, sr = librosa.load(audio_path)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for onset in onset_times:
            frame = int(librosa.time_to_frames(onset, sr=sr))
            pitch = pitches[:, frame].argmax()
            note_number = pitch if pitch > 0 else 60
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=onset, end=onset+0.5)
            instrument.notes.append(note)
        midi.instruments.append(instrument)
        midi.write(output_path)
        print(f"✓ Created MIDI file: {output_path}\n")
        print(f"\n✓ Successfully converted to: {result}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()