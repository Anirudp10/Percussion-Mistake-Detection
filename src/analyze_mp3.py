"""
Analyze MP3 file - converts to MIDI and shows transcription results.
Uses the new performance_to_midi.py module.
"""

import sys
from pathlib import Path

from performance_to_midi import PerformanceToMIDI
from validate_midi import MIDIValidator
from midi_summary import MIDISummary


def analyze_mp3_only(audio_path: str):
	"""
	Analyze an MP3 file when no MIDI score is provided.
	This will convert the MP3 to MIDI and show transcription results.
	"""
	print("=" * 60)
	print("MP3 Analysis Mode (No MIDI Score Provided)")
	print("=" * 60)
	print(f"\nInput: {audio_path}\n")
    
	# Step 1: Convert MP3 to MIDI using new module
	print("Step 1: Converting MP3 to MIDI...")
	print("-" * 60)
	try:
		converter = PerformanceToMIDI()
		midi_path = converter.convert(audio_path)
		print(f"✓ Created MIDI file: {midi_path}\n")
	except Exception as e:
		print(f"✗ Error converting to MIDI: {str(e)}")
		return
    
	# Step 2: Validate the MIDI
	print("Step 2: Validating MIDI file...")
	print("-" * 60)
	MIDIValidator.print_validation_report(midi_path)
    
	# Step 3: Show transcription summary
	print("\nStep 3: Transcription Summary")
	print("-" * 60)
	MIDISummary.print_summary(midi_path)
    
	print("\n" + "=" * 60)
	print("Analysis Complete!")
	print("=" * 60)
	print(f"\nGenerated MIDI file: {midi_path}")
	print("\nTo compare against a different MIDI score, use:")
	print(f"  python run_detector.py {audio_path} <your_score.mid>")


def main():
	"""Main entry point."""
	if len(sys.argv) < 2:
		print("Usage: python analyze_mp3.py <audio_file.mp3>")
		print("\nThis script will:")
		print("  1. Convert MP3 to MIDI")
		print("  2. Show transcription results")
		print("  3. Compare MP3 to its own MIDI (self-comparison)")
		print("\nExample:")
		print("  python analyze_mp3.py song.mp3")
		sys.exit(1)
    
	audio_path = sys.argv[1]
    
	# Check if file exists
	if not Path(audio_path).exists():
		print(f"Error: Audio file not found: {audio_path}")
		sys.exit(1)
    
	analyze_mp3_only(audio_path)


if __name__ == "__main__":
	main()
# ...existing code from your old analyze_mp3.py...