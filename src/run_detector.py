# ...existing code from your old run_detector.py...
"""
Simple entry point script for running the music mistake detector.
This can be run directly: python run_detector.py <audio.mp3> <score.mid>
"""

import sys
from pathlib import Path

# Add parent directory to path to import music_detector
sys.path.insert(0, str(Path(__file__).parent))

from music_detector.detector import MusicMistakeDetector


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_detector.py <audio_file.mp3> [score_file.mid]")
        print("\nIf only MP3 is provided, it will be converted to MIDI first.")
        print("\nExamples:")
        print("  python run_detector.py performance.mp3 score.mid  # Compare MP3 to MIDI score")
        print("  python run_detector.py performance.mp3            # Convert MP3 to MIDI only")
        print("\nFor MP3-only analysis, you can also use:")
        print("  python analyze_mp3.py performance.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # If only MP3 provided, convert to MIDI
    if len(sys.argv) == 2:
        print("Only MP3 provided. Converting to MIDI first...")
        import librosa
        import pretty_midi
        audio_path_obj = Path(audio_path)
        midi_path = str(audio_path_obj.with_suffix('.mid'))
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
        midi.write(midi_path)
        print(f"✓ Created MIDI file: {midi_path}\n")
        print(f"Created MIDI: {midi_path}\n")
        print("Note: Comparing MP3 to its own transcription shows transcription accuracy,")
        print("      not performance mistakes. For mistake detection, provide a separate MIDI score.\n")
    else:
        midi_path = sys.argv[2]
    
    audio_path = sys.argv[1]
    midi_path = sys.argv[2]
    
    # Check if files exist
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    if not Path(midi_path).exists():
        print(f"Error: MIDI file not found: {midi_path}")
        sys.exit(1)
    
    # Create detector and analyze
    detector = MusicMistakeDetector(timing_tolerance=0.03, pitch_tolerance=0.5)
    
    try:
        mistakes = detector.analyze(audio_path, midi_path)
        # Print results
        detector.print_mistakes(mistakes)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
