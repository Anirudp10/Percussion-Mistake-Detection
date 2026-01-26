
"""
Analyze MP3 file - handles cases where only MP3 is provided.
If only MP3 is given, it will:
1. Convert MP3 to MIDI (transcription)
2. Show what was detected
3. Optionally compare MP3 to the converted MIDI (self-comparison)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import librosa
import pretty_midi
from music_detector.detector import MusicMistakeDetector
from music_detector.midi_validator import MIDIValidator


def analyze_mp3_only(audio_path: str):
	"""
	Analyze an MP3 file when no MIDI score is provided.
	This will convert the MP3 to MIDI and show transcription results.
	"""
	print("=" * 60)
	print("MP3 Analysis Mode (No MIDI Score Provided)")
	print("=" * 60)
	print(f"\nInput: {audio_path}\n")
    
	# Step 1: Convert MP3 to MIDI
	print("Step 1: Converting MP3 to MIDI...")
	print("-" * 60)
	try:
		# Auto-generate MIDI filename
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
			# Find the strongest pitch at each onset
			frame = int(librosa.time_to_frames(onset, sr=sr))
			pitch = pitches[:, frame].argmax()
			note_number = pitch if pitch > 0 else 60  # Default to middle C if not found
			note = pretty_midi.Note(velocity=100, pitch=note_number, start=onset, end=onset+0.5)
			instrument.notes.append(note)
		midi.instruments.append(instrument)
		midi.write(midi_path)
		print(f"✓ Created MIDI file: {midi_path}\n")
		print(f"✓ Created MIDI file: {midi_path}\n")
	except Exception as e:
		print(f"✗ Error converting to MIDI: {str(e)}")
		return
    
	# Step 2: Validate the MIDI
	print("Step 2: Validating MIDI file...")
	print("-" * 60)
	MIDIValidator.print_validation_report(midi_path)
    
	# Step 3: Show transcription summary
	print("Step 3: Transcription Summary")
	print("-" * 60)
	try:
		from music_detector.midi_score import MIDIScore
		score = MIDIScore(midi_path)
		print(f"✓ Detected {len(score.notes)} notes")
		print(f"✓ Estimated tempo: {score.tempo:.1f} BPM")
        
		if len(score.notes) > 0:
			print(f"\nFirst few notes detected:")
			for i, note in enumerate(score.notes[:10], 1):
				import librosa
				try:
					note_name = librosa.midi_to_note(note['note'])
					print(f"  {i}. {note_name} (MIDI {note['note']}) at {note['onset']:.3f}s")
				except:
					print(f"  {i}. MIDI {note['note']} at {note['onset']:.3f}s")
            
			if len(score.notes) > 10:
				print(f"  ... and {len(score.notes) - 10} more notes")
	except Exception as e:
		print(f"✗ Error reading MIDI: {str(e)}")
    
	# Step 4: Optional self-comparison
	print("\n" + "=" * 60)
	print("Step 4: Self-Comparison (MP3 vs its own MIDI transcription)")
	print("=" * 60)
	print("\nNote: This compares the MP3 performance against its own transcription.")
	print("This shows transcription accuracy, not performance mistakes.\n")
    
	try:
		detector = MusicMistakeDetector(timing_tolerance=0.05, pitch_tolerance=0.5)
		mistakes = detector.analyze(audio_path, midi_path)
		detector.print_mistakes(mistakes)
        
		if len(mistakes) == 0:
			print("\n✓ Perfect transcription match! The MP3 was accurately transcribed to MIDI.")
		else:
			print(f"\n⚠ Found {len(mistakes)} differences between MP3 and its MIDI transcription.")
			print("These may indicate:")
			print("  - Transcription inaccuracies")
			print("  - Complex audio that's hard to transcribe")
			print("  - Timing variations in the performance")
	except Exception as e:
		print(f"✗ Error during comparison: {str(e)}")
		import traceback
		traceback.print_exc()
    
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