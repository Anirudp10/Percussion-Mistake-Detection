# ...existing code from your old validate_midi.py...
"""
Standalone script to validate MIDI files.
Usage: python validate_midi.py <midi_file>
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


import pretty_midi

class MIDIValidator:
    @staticmethod
    def print_validation_report(midi_path: str):
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            num_tracks = len(midi.instruments)
            num_notes = sum(len(inst.notes) for inst in midi.instruments)
            tempos = midi.get_tempo_changes()[1]
            print(f"MIDI Validation Report for: {midi_path}")
            print(f"  Number of tracks: {num_tracks}")
            print(f"  Total notes: {num_notes}")
            if len(tempos) > 0:
                print(f"  Tempos: {tempos}")
            else:
                print("  No tempo information found.")
            print("  Validation: Success! MIDI file loaded and parsed.")
        except Exception as e:
            print(f"  Validation failed: {str(e)}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_midi.py <midi_file>")
        print("\nExample:")
        print("  python validate_midi.py score.mid")
        sys.exit(1)
    
    midi_path = sys.argv[1]
    
    # Check if file exists
    if not Path(midi_path).exists():
        print(f"Error: MIDI file not found: {midi_path}")
        sys.exit(1)
    
    # Print validation report
    MIDIValidator.print_validation_report(midi_path)


if __name__ == "__main__":
    main()
