# ...existing code from your old music_mistake_detector.py...
"""
Music Performance Mistake Detection System
Compares audio recordings to MIDI scores to detect performance mistakes.
"""

import librosa
import numpy as np
import mido
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Mistake:
    """Represents a detected mistake in the performance."""
    timestamp: float  # Time in seconds
    error_type: str  # 'wrong_note', 'missing_note', 'extra_note', 'timing_error'
    expected_note: Optional[int] = None  # MIDI note number
    detected_note: Optional[int] = None  # MIDI note number
    expected_time: Optional[float] = None  # Expected onset time
    detected_time: Optional[float] = None  # Detected onset time
    confidence: Optional[float] = None  # Confidence score if available


class AudioProcessor:
    """Handles audio loading and feature extraction."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio array and sample rate."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(audio) == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {str(e)}")
    
    def detect_onsets(self, audio: np.ndarray) -> np.ndarray:
        """Detect note onsets in audio using librosa."""
        if len(audio) == 0:
            return np.array([])
        
        try:
            onsets = librosa.onset.onset_detect(
                y=audio,
                sr=self.sample_rate,
                units='time',
                backtrack=True
            )
            return onsets if len(onsets) > 0 else np.array([])
        except Exception as e:
            print(f"Warning: Onset detection failed: {str(e)}")
            return np.array([])
    
    def extract_pitch_yin(self, audio: np.ndarray, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract pitch using YIN algorithm via librosa.
        Returns pitch values in MIDI note numbers, times, and confidence scores.
        """
        # Use librosa's pyin (probabilistic YIN) for pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # Lower bound (C2)
            fmax=librosa.note_to_hz('C7'),  # Upper bound (C7)
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # Convert Hz to MIDI note numbers
        midi_notes = np.full_like(f0, np.nan, dtype=np.float64)
        valid_mask = ~np.isnan(f0)
        midi_notes[valid_mask] = librosa.hz_to_midi(f0[valid_mask])
        
        # Time array for each frame
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        return midi_notes, times, voiced_probs
    
    def extract_pitch_at_onsets(self, audio: np.ndarray, onsets: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Extract pitch values at detected onset times.
        Returns list of (onset_time, midi_note, confidence) tuples.
        """
        midi_notes, times, confidences = self.extract_pitch_yin(audio)
        
        onset_pitches = []
        for onset_time in onsets:
            # Clamp onset_time to valid range
            if onset_time < times[0]:
                idx = 0
            elif onset_time > times[-1]:
                idx = len(times) - 1
            else:
                # Find closest time index
                idx = np.argmin(np.abs(times - onset_time))
            
            # Check bounds
            if 0 <= idx < len(midi_notes) and not np.isnan(midi_notes[idx]):
                midi_note = float(midi_notes[idx])
                # Clamp MIDI note to valid range (0-127)
                midi_note = max(0, min(127, midi_note))
                
                confidence = 0.5  # Default confidence
                if confidences is not None and idx < len(confidences):
                    conf_val = confidences[idx]
                    if not np.isnan(conf_val):
                        confidence = float(conf_val)
                
                onset_pitches.append((
                    onset_time,
                    midi_note,
                    confidence
                ))
        
        return onset_pitches


class MIDIScore:
    """Handles MIDI score loading and parsing."""
    
    def __init__(self, midi_path: str):
        self.midi_path = midi_path
        self.notes = []
        self.tempo = 120  # Default tempo in BPM
        try:
            self._load_midi()
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file {midi_path}: {str(e)}")
    
    def _load_midi(self):
        """Load and parse MIDI file to extract note events."""
        try:
            mid = mido.MidiFile(self.midi_path)
        except Exception as e:
            raise ValueError(f"Failed to open MIDI file: {str(e)}")
        
        if mid.ticks_per_beat <= 0:
            raise ValueError("Invalid MIDI file: ticks_per_beat must be positive")
        
        # Calculate tempo (ticks per second)
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000  # Default tempo in microseconds per beat
        
        # Track tempo changes with timestamps
        tempo_changes = [(0.0, tempo)]  # (time, tempo)
        
        # Collect all messages from all tracks with absolute timing
        all_messages = []
        global_time = 0.0
        
        for track in mid.tracks:
            track_time = 0.0
            for msg in track:
                track_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
                
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    tempo_changes.append((track_time, tempo))
                
                # Store message with absolute time
                all_messages.append((track_time, msg))
        
        # Sort all messages by time (in case tracks overlap)
        all_messages.sort(key=lambda x: x[0])
        
        # Process messages in chronological order
        active_notes = {}  # Track active notes: key = (channel, note, instance_id)
        note_instance_counter = 0
        
        for msg_time, msg in all_messages:
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note start - use instance counter to handle polyphonic same notes
                # Validate MIDI note range
                if 0 <= msg.note <= 127:
                    instance_id = note_instance_counter
                    note_instance_counter += 1
                    key = (msg.channel, msg.note, instance_id)
                    active_notes[key] = {
                        'onset': msg_time,
                        'velocity': msg.velocity,
                        'channel': msg.channel,
                        'note': msg.note
                    }
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Note end - find matching note_on
                # Try to find the most recent unmatched note_on for this channel+note
                matching_key = None
                latest_time = -1
                
                for key, note_data in active_notes.items():
                    if key[0] == msg.channel and key[1] == msg.note:
                        if note_data['onset'] > latest_time:
                            latest_time = note_data['onset']
                            matching_key = key
                
                if matching_key:
                    note_data = active_notes[matching_key]
                    onset_time = note_data['onset']
                    duration = msg_time - onset_time
                    
                    # Use velocity from note_on, fallback to note_off if available
                    velocity = note_data['velocity']
                    if velocity == 0 and hasattr(msg, 'velocity'):
                        velocity = msg.velocity
                    if velocity == 0:
                        velocity = 64  # Default
                    
                    # Validate note number
                    note_num = msg.note
                    if 0 <= note_num <= 127:
                        self.notes.append({
                            'onset': onset_time,
                            'note': note_num,
                            'duration': max(duration, 0.01),  # Minimum duration
                            'velocity': velocity
                        })
                    del active_notes[matching_key]
        
        # Handle any remaining active notes (notes that never got note_off)
        final_time = all_messages[-1][0] if all_messages else 0.0
        for key, note_data in list(active_notes.items()):
            duration = final_time - note_data['onset']
            note_num = note_data['note']
            if 0 <= note_num <= 127:
                self.notes.append({
                    'onset': note_data['onset'],
                    'note': note_num,
                    'duration': max(duration, 0.01),
                    'velocity': note_data['velocity']
                })
        
        # Sort notes by onset time
        self.notes.sort(key=lambda x: x['onset'])
        
        # Calculate average tempo from tempo changes
        if tempo_changes:
            # Use the most common tempo or average
            total_tempo_time = 0.0
            weighted_tempo_sum = 0.0
            
            for i in range(len(tempo_changes)):
                start_time, start_tempo = tempo_changes[i]
                end_time = tempo_changes[i + 1][0] if i + 1 < len(tempo_changes) else final_time
                duration = end_time - start_time
                
                if duration > 0 and start_tempo > 0:
                    # Convert microseconds per beat to BPM
                    bpm = 60000000.0 / start_tempo
                    weighted_tempo_sum += bpm * duration
                    total_tempo_time += duration
            
            if total_tempo_time > 0:
                self.tempo = weighted_tempo_sum / total_tempo_time
            elif tempo_changes and tempo_changes[0][1] > 0:
                self.tempo = 60000000.0 / tempo_changes[0][1]
            else:
                self.tempo = 120  # Default
        else:
            self.tempo = 120  # Default
        
        # Ensure we have at least some notes
        if len(self.notes) == 0:
            print(f"Warning: No notes found in MIDI file {self.midi_path}")
    
    def get_notes_in_time_range(self, start_time: float, end_time: float) -> List[Dict]:
        """Get all notes that should be active in the given time range."""
        return [
            note for note in self.notes
            if note['onset'] <= end_time and (note['onset'] + note['duration']) >= start_time
        ]


class ScoreAlignment:
    """Handles alignment between score and performance."""
    
    def __init__(self, timing_tolerance: float = 0.03):  # 30ms default
        self.timing_tolerance = timing_tolerance
    
    def align_performance_to_score(
        self,
        score_notes: List[Dict],
        performance_onsets: List[Tuple[float, float, float]]  # (time, midi_note, confidence)
    ) -> List[Dict]:
        """
        Align performance onsets to score notes.
        Returns list of alignment results with matches and mismatches.
        """
        if not score_notes or not performance_onsets:
            # Handle empty cases
            alignments = []
            if score_notes:
                for score_note in score_notes:
                    alignments.append({
                        'score_note': score_note,
                        'performance': None,
                        'matched': False,
                        'time_error': None,
                        'pitch_error': None
                    })
            if performance_onsets:
                for perf_time, perf_note, conf in performance_onsets:
                    alignments.append({
                        'score_note': None,
                        'performance': (perf_time, perf_note, conf),
                        'matched': False,
                        'time_error': None,
                        'pitch_error': None,
                        'extra': True
                    })
            return alignments
        
        alignments = []
        used_performance_indices = set()
        
        # Calculate a global time offset if needed (for tempo differences)
        # Use first few notes to estimate offset
        time_offset = 0.0
        if len(score_notes) > 0 and len(performance_onsets) > 0:
            try:
                # Try to find initial alignment
                first_score_time = score_notes[0]['onset']
                # Find performance onset closest to first score note
                time_diffs = np.array([abs(perf_time - first_score_time) for perf_time, _, _ in performance_onsets])
                if len(time_diffs) > 0:
                    min_diff_idx = np.argmin(time_diffs)
                    if time_diffs[min_diff_idx] < 1.0:  # Within 1 second
                        time_offset = performance_onsets[min_diff_idx][0] - first_score_time
            except Exception:
                time_offset = 0.0
        
        for score_note in score_notes:
            best_match = None
            best_distance = float('inf')
            best_idx = None
            
            # Adjusted score time with offset
            adjusted_score_time = score_note['onset'] + time_offset
            
            # Find closest performance onset within tolerance
            for idx, (perf_time, perf_note, conf) in enumerate(performance_onsets):
                if idx in used_performance_indices:
                    continue
                
                time_diff = abs(perf_time - adjusted_score_time)
                note_diff = abs(perf_note - score_note['note'])
                
                # Combined distance metric (weighted)
                # Normalize note difference (semitone) and time difference
                distance = time_diff * 10 + note_diff  # Time weighted more heavily
                
                # Allow slightly larger tolerance for initial alignment
                tolerance = self.timing_tolerance * 2.0 if len(used_performance_indices) < 3 else self.timing_tolerance
                
                if time_diff <= tolerance and distance < best_distance:
                    best_match = (perf_time, perf_note, conf)
                    best_distance = distance
                    best_idx = idx
            
            if best_match:
                # Matched note
                alignments.append({
                    'score_note': score_note,
                    'performance': best_match,
                    'matched': True,
                    'time_error': best_match[0] - score_note['onset'],
                    'pitch_error': best_match[1] - score_note['note']
                })
                used_performance_indices.add(best_idx)
            else:
                # Missing note (in score but not in performance)
                alignments.append({
                    'score_note': score_note,
                    'performance': None,
                    'matched': False,
                    'time_error': None,
                    'pitch_error': None
                })
        
        # Find extra notes (in performance but not matched to score)
        for idx, (perf_time, perf_note, conf) in enumerate(performance_onsets):
            if idx not in used_performance_indices:
                alignments.append({
                    'score_note': None,
                    'performance': (perf_time, perf_note, conf),
                    'matched': False,
                    'time_error': None,
                    'pitch_error': None,
                    'extra': True
                })
        
        return alignments


class MistakeDetector:
    """Detects mistakes by comparing aligned performance to score."""
    
    def __init__(self, pitch_tolerance: float = 0.5, timing_tolerance: float = 0.03):
        self.pitch_tolerance = pitch_tolerance  # Semitones
        self.timing_tolerance = timing_tolerance  # Seconds
    
    def detect_mistakes(self, alignments: List[Dict]) -> List[Mistake]:
        """Analyze alignments and detect mistakes."""
        mistakes = []
        
        for alignment in alignments:
            if alignment.get('extra', False):
                # Extra note (played but not in score)
                perf_time, perf_note, conf = alignment['performance']
                # Clamp and round MIDI note
                midi_note = int(round(max(0, min(127, perf_note))))
                mistakes.append(Mistake(
                    timestamp=perf_time,
                    error_type='extra_note',
                    detected_note=midi_note,
                    confidence=conf
                ))
            
            elif not alignment['matched'] and alignment['score_note'] is not None:
                # Missing note (in score but not played)
                score_note = alignment['score_note']
                mistakes.append(Mistake(
                    timestamp=score_note['onset'],
                    error_type='missing_note',
                    expected_note=score_note['note'],
                    expected_time=score_note['onset']
                ))
            
            elif alignment['matched']:
                # Check for wrong note or timing error
                score_note = alignment['score_note']
                perf_time, perf_note, conf = alignment['performance']
                
                pitch_error = abs(alignment['pitch_error'])
                time_error = abs(alignment['time_error'])
                
                if pitch_error > self.pitch_tolerance:
                    # Wrong note
                    midi_note = int(round(max(0, min(127, perf_note))))
                    mistakes.append(Mistake(
                        timestamp=perf_time,
                        error_type='wrong_note',
                        expected_note=score_note['note'],
                        detected_note=midi_note,
                        expected_time=score_note['onset'],
                        detected_time=perf_time,
                        confidence=conf
                    ))
                
                if time_error > self.timing_tolerance:
                    # Timing error
                    midi_note = int(round(max(0, min(127, perf_note))))
                    mistakes.append(Mistake(
                        timestamp=perf_time,
                        error_type='timing_error',
                        expected_note=score_note['note'],
                        detected_note=midi_note,
                        expected_time=score_note['onset'],
                        detected_time=perf_time,
                        confidence=conf
                    ))
        
        # Sort mistakes by timestamp
        mistakes.sort(key=lambda x: x.timestamp)
        return mistakes


class MusicMistakeDetector:
    """Main class that orchestrates the entire mistake detection pipeline."""
    
    def __init__(self, timing_tolerance: float = 0.03, pitch_tolerance: float = 0.5):
        self.audio_processor = AudioProcessor()
        self.timing_tolerance = timing_tolerance
        self.pitch_tolerance = pitch_tolerance
    
    def analyze(self, audio_path: str, midi_path: str) -> List[Mistake]:
        """
        Main analysis function.
        
        Args:
            audio_path: Path to MP3 audio file
            midi_path: Path to MIDI score file
        
        Returns:
            List of detected mistakes
        """
        print(f"Loading audio: {audio_path}")
        audio, sr = self.audio_processor.load_audio(audio_path)
        
        print(f"Loading MIDI score: {midi_path}")
        score = MIDIScore(midi_path)
        print(f"Found {len(score.notes)} notes in score")
        
        print("Detecting onsets...")
        onsets = self.audio_processor.detect_onsets(audio)
        print(f"Detected {len(onsets)} onsets")
        
        print("Extracting pitch at onsets...")
        performance_onsets = self.audio_processor.extract_pitch_at_onsets(audio, onsets)
        print(f"Extracted pitch for {len(performance_onsets)} onsets")
        
        print("Aligning performance to score...")
        aligner = ScoreAlignment(timing_tolerance=self.timing_tolerance)
        alignments = aligner.align_performance_to_score(score.notes, performance_onsets)
        
        print("Detecting mistakes...")
        detector = MistakeDetector(
            pitch_tolerance=self.pitch_tolerance,
            timing_tolerance=self.timing_tolerance
        )
        mistakes = detector.detect_mistakes(alignments)
        
        print(f"Detected {len(mistakes)} mistakes")
        return mistakes
    
    def print_mistakes(self, mistakes: List[Mistake]):
        """Print mistakes in a readable format."""
        if not mistakes:
            print("\n✓ No mistakes detected!")
            return
        
        print(f"\n✗ Found {len(mistakes)} mistake(s):\n")
        for i, mistake in enumerate(mistakes, 1):
            print(f"{i}. [{mistake.timestamp:.3f}s] {mistake.error_type.upper()}")
            if mistake.expected_note is not None:
                try:
                    note_name = librosa.midi_to_note(mistake.expected_note)
                    print(f"   Expected: {note_name} (MIDI {int(mistake.expected_note)})")
                except (ValueError, TypeError):
                    print(f"   Expected: MIDI {int(mistake.expected_note)}")
            if mistake.detected_note is not None:
                try:
                    note_name = librosa.midi_to_note(mistake.detected_note)
                    print(f"   Detected: {note_name} (MIDI {int(mistake.detected_note)})")
                except (ValueError, TypeError):
                    print(f"   Detected: MIDI {int(mistake.detected_note)}")
            if mistake.expected_time is not None and mistake.detected_time is not None:
                time_diff = (mistake.detected_time - mistake.expected_time) * 1000
                print(f"   Timing: {time_diff:+.1f}ms")
            if mistake.confidence is not None:
                print(f"   Confidence: {mistake.confidence:.2f}")
            print()


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python music_mistake_detector.py <audio_file.mp3> <score_file.mid>")
        print("\nExample:")
        print("  python music_mistake_detector.py performance.mp3 score.mid")
        sys.exit(1)
    
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
    mistakes = detector.analyze(audio_path, midi_path)
    
    # Print results
    detector.print_mistakes(mistakes)


if __name__ == "__main__":
    main()
