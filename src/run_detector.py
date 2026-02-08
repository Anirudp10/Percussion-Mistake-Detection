"""
Percussion Mistake Detector - Main Entry Point
Converts audio to MIDI, then compares it to a reference score to detect mistakes.
Usage: python run_detector.py <audio.mp3> <score.mid> [output_report.json]
"""

import sys
from pathlib import Path

# Import our new modules
from performance_to_midi import PerformanceToMIDI
from midi_comparator import MIDIComparator


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_detector.py <audio_file> <score.mid> [output_report.json]")
        print("\nExamples:")
        print("  python run_detector.py performance.mp3 score.mid")
        print("  python run_detector.py performance.mp3 score.mid report.json")
        print("  python run_detector.py performance.mp3  # Convert to MIDI only")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    try:
        # STEP 1: Convert audio (MP3) to MIDI
        print("\n" + "="*60)
        print("STEP 1: Converting Audio to MIDI")
        print("="*60)
        converter = PerformanceToMIDI()
        performance_midi_path = converter.convert(audio_path)
        
        # If only audio provided, stop here
        if len(sys.argv) == 2:
            print("\n✓ Conversion complete!")
            print(f"Performance MIDI saved to: {performance_midi_path}")
            print("\nTo compare with a score, run:")
            print(f"  python run_detector.py {audio_path} <score.mid>")
            sys.exit(0)
        
        score_midi_path = sys.argv[2]
        output_json = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Check if score file exists
        if not Path(score_midi_path).exists():
            print(f"Error: Score MIDI file not found: {score_midi_path}")
            sys.exit(1)
        
        # STEP 2: Compare performance MIDI to score MIDI
        print("\n" + "="*60)
        print("STEP 2: Comparing Performance to Score")
        print("="*60)
        comparator = MIDIComparator(
            timing_tolerance_ms=30,
            pitch_tolerance_semitones=0.5
        )
        metrics = comparator.compare(performance_midi_path, score_midi_path, output_json)
        
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
