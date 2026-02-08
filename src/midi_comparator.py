"""
MIDI Comparator
Compares two MIDI files (performance vs score) to detect mistakes.
Outputs metrics including pitch errors, timing errors, and performance score.
"""

import mido
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class MistakeReport:
    """Individual mistake detected."""
    timestamp: float
    error_type: str  # 'wrong_note', 'missing_note', 'extra_note', 'timing_error'
    expected_note: Optional[int] = None
    detected_note: Optional[int] = None
    expected_time: Optional[float] = None
    detected_time: Optional[float] = None
    timing_error_ms: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Overall performance metrics."""
    total_notes_in_score: int
    total_notes_detected: int
    correct_notes: int
    wrong_notes: int
    missing_notes: int
    extra_notes: int
    timing_errors: int
    mean_timing_error_ms: float
    timing_error_std_ms: float
    pitch_accuracy_percent: float
    timing_accuracy_percent: float
    overall_score: float
    mistakes: List[MistakeReport]


class MIDIComparator:
    """Compares two MIDI files and detects mistakes."""
    
    def __init__(self, timing_tolerance_ms: float = 30, pitch_tolerance_semitones: float = 0.5):
        """
        Args:
            timing_tolerance_ms: Timing tolerance in milliseconds
            pitch_tolerance_semitones: Pitch tolerance in semitones
        """
        self.timing_tolerance = timing_tolerance_ms / 1000.0  # Convert to seconds
        self.pitch_tolerance = pitch_tolerance_semitones
    
    def compare(self, performance_midi_path: str, score_midi_path: str, 
                output_json_path: Optional[str] = None) -> PerformanceMetrics:
        """
        Compare performance MIDI to score MIDI.
        
        Args:
            performance_midi_path: Path to performance MIDI file
            score_midi_path: Path to reference score MIDI file
            output_json_path: Optional path to save JSON report
        
        Returns:
            PerformanceMetrics object with all analysis results
        """
        print(f"Loading performance MIDI: {performance_midi_path}")
        performance_notes = self._load_midi_notes(performance_midi_path)
        print(f"Found {len(performance_notes)} notes in performance")
        
        print(f"Loading score MIDI: {score_midi_path}")
        score_notes = self._load_midi_notes(score_midi_path)
        print(f"Found {len(score_notes)} notes in score")
        
        print("Aligning notes...")
        alignments = self._align_notes(score_notes, performance_notes)
        
        print("Detecting mistakes...")
        mistakes = self._detect_mistakes(alignments)
        
        print("Calculating metrics...")
        metrics = self._calculate_metrics(score_notes, performance_notes, alignments, mistakes)
        
        # Print to console
        self._print_results(metrics)
        
        # Save to JSON if requested
        if output_json_path:
            self._save_json_report(metrics, output_json_path)
            print(f"\n✓ Saved detailed report to: {output_json_path}")
        
        return metrics
    
    def _load_midi_notes(self, midi_path: str) -> List[Dict]:
        """Load notes from MIDI file."""
        try:
            mid = mido.MidiFile(midi_path)
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file {midi_path}: {str(e)}")
        
        if mid.ticks_per_beat <= 0:
            raise ValueError("Invalid MIDI file: ticks_per_beat must be positive")
        
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000  # Default tempo
        
        notes = []
        active_notes = {}  # Track active notes
        
        # Process all tracks
        for track in mid.tracks:
            track_time = 0.0
            
            for msg in track:
                # Update time
                track_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
                
                # Handle tempo changes
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
                # Handle note events
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note start
                    if 0 <= msg.note <= 127:
                        key = (msg.channel, msg.note, len(active_notes))
                        active_notes[key] = {
                            'onset': track_time,
                            'note': msg.note,
                            'velocity': msg.velocity
                        }
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note end - find matching note_on
                    matching_key = None
                    latest_time = -1
                    
                    for key, note_data in active_notes.items():
                        if key[0] == msg.channel and key[1] == msg.note:
                            if note_data['onset'] > latest_time:
                                latest_time = note_data['onset']
                                matching_key = key
                    
                    if matching_key:
                        note_data = active_notes[matching_key]
                        duration = track_time - note_data['onset']
                        
                        notes.append({
                            'onset': note_data['onset'],
                            'note': note_data['note'],
                            'duration': max(duration, 0.01),
                            'velocity': note_data['velocity']
                        })
                        del active_notes[matching_key]
        
        # Sort notes by onset time
        notes.sort(key=lambda x: x['onset'])
        
        if len(notes) == 0:
            print(f"Warning: No notes found in MIDI file {midi_path}")
        
        return notes
    
    def _align_notes(self, score_notes: List[Dict], performance_notes: List[Dict]) -> List[Dict]:
        """
        Align performance notes to score notes.
        
        Returns list of alignments with match information.
        """
        if not score_notes or not performance_notes:
            # Handle empty cases
            alignments = []
            for note in score_notes:
                alignments.append({
                    'score_note': note,
                    'performance_note': None,
                    'matched': False
                })
            for note in performance_notes:
                alignments.append({
                    'score_note': None,
                    'performance_note': note,
                    'matched': False,
                    'extra': True
                })
            return alignments
        
        alignments = []
        used_performance_indices = set()
        
        # Calculate initial time offset (for tempo differences)
        time_offset = 0.0
        if len(score_notes) > 0 and len(performance_notes) > 0:
            first_score_time = score_notes[0]['onset']
            time_diffs = [abs(perf['onset'] - first_score_time) for perf in performance_notes]
            if time_diffs:
                min_idx = np.argmin(time_diffs)
                if time_diffs[min_idx] < 1.0:
                    time_offset = performance_notes[min_idx]['onset'] - first_score_time
        
        # Match each score note to closest performance note
        for score_note in score_notes:
            best_match = None
            best_idx = None
            best_distance = float('inf')
            
            adjusted_score_time = score_note['onset'] + time_offset
            
            for idx, perf_note in enumerate(performance_notes):
                if idx in used_performance_indices:
                    continue
                
                time_diff = abs(perf_note['onset'] - adjusted_score_time)
                note_diff = abs(perf_note['note'] - score_note['note'])
                
                # Combined distance (time weighted more)
                distance = time_diff * 10 + note_diff
                
                if time_diff <= self.timing_tolerance * 2.0 and distance < best_distance:
                    best_match = perf_note
                    best_idx = idx
                    best_distance = distance
            
            if best_match:
                alignments.append({
                    'score_note': score_note,
                    'performance_note': best_match,
                    'matched': True,
                    'time_error': best_match['onset'] - score_note['onset'],
                    'pitch_error': best_match['note'] - score_note['note']
                })
                used_performance_indices.add(best_idx)
            else:
                # Missing note
                alignments.append({
                    'score_note': score_note,
                    'performance_note': None,
                    'matched': False
                })
        
        # Find extra notes
        for idx, perf_note in enumerate(performance_notes):
            if idx not in used_performance_indices:
                alignments.append({
                    'score_note': None,
                    'performance_note': perf_note,
                    'matched': False,
                    'extra': True
                })
        
        return alignments
    
    def _detect_mistakes(self, alignments: List[Dict]) -> List[MistakeReport]:
        """Detect mistakes from alignments."""
        mistakes = []
        
        for alignment in alignments:
            if alignment.get('extra', False):
                # Extra note
                perf = alignment['performance_note']
                mistakes.append(MistakeReport(
                    timestamp=perf['onset'],
                    error_type='extra_note',
                    detected_note=perf['note']
                ))
            
            elif not alignment['matched'] and alignment['score_note'] is not None:
                # Missing note
                score = alignment['score_note']
                mistakes.append(MistakeReport(
                    timestamp=score['onset'],
                    error_type='missing_note',
                    expected_note=score['note'],
                    expected_time=score['onset']
                ))
            
            elif alignment['matched']:
                score = alignment['score_note']
                perf = alignment['performance_note']
                
                pitch_error = abs(alignment['pitch_error'])
                time_error = abs(alignment['time_error'])
                
                # Check for wrong note
                if pitch_error > self.pitch_tolerance:
                    mistakes.append(MistakeReport(
                        timestamp=perf['onset'],
                        error_type='wrong_note',
                        expected_note=score['note'],
                        detected_note=perf['note'],
                        expected_time=score['onset'],
                        detected_time=perf['onset'],
                        timing_error_ms=alignment['time_error'] * 1000
                    ))
                
                # Check for timing error
                if time_error > self.timing_tolerance:
                    mistakes.append(MistakeReport(
                        timestamp=perf['onset'],
                        error_type='timing_error',
                        expected_note=score['note'],
                        detected_note=perf['note'],
                        expected_time=score['onset'],
                        detected_time=perf['onset'],
                        timing_error_ms=alignment['time_error'] * 1000
                    ))
        
        mistakes.sort(key=lambda x: x.timestamp)
        return mistakes
    
    def _calculate_metrics(self, score_notes: List[Dict], performance_notes: List[Dict],
                          alignments: List[Dict], mistakes: List[MistakeReport]) -> PerformanceMetrics:
        """Calculate performance metrics."""
        total_score = len(score_notes)
        total_detected = len(performance_notes)
        
        # Count mistake types
        wrong_notes = sum(1 for m in mistakes if m.error_type == 'wrong_note')
        missing_notes = sum(1 for m in mistakes if m.error_type == 'missing_note')
        extra_notes = sum(1 for m in mistakes if m.error_type == 'extra_note')
        timing_errors = sum(1 for m in mistakes if m.error_type == 'timing_error')
        
        # Calculate correct notes (matched with acceptable pitch and timing)
        correct_notes = sum(1 for a in alignments 
                          if a['matched'] and 
                          abs(a.get('pitch_error', 0)) <= self.pitch_tolerance and
                          abs(a.get('time_error', 0)) <= self.timing_tolerance)
        
        # Calculate timing statistics
        timing_errors_list = [a['time_error'] * 1000 for a in alignments 
                             if a['matched'] and 'time_error' in a]
        
        if timing_errors_list:
            mean_timing_error = float(np.mean(np.abs(timing_errors_list)))
            timing_std = float(np.std(timing_errors_list))
        else:
            mean_timing_error = 0.0
            timing_std = 0.0
        
        # Calculate accuracy percentages
        if total_score > 0:
            pitch_accuracy = (correct_notes / total_score) * 100
            timing_accuracy = ((total_score - timing_errors) / total_score) * 100
        else:
            pitch_accuracy = 0.0
            timing_accuracy = 0.0
        
        # Calculate overall performance score (0-100)
        # Weighted: 50% pitch accuracy, 30% timing accuracy, 20% completeness
        completeness = ((total_score - missing_notes) / total_score * 100) if total_score > 0 else 0
        overall_score = (pitch_accuracy * 0.5 + timing_accuracy * 0.3 + completeness * 0.2)
        
        return PerformanceMetrics(
            total_notes_in_score=total_score,
            total_notes_detected=total_detected,
            correct_notes=correct_notes,
            wrong_notes=wrong_notes,
            missing_notes=missing_notes,
            extra_notes=extra_notes,
            timing_errors=timing_errors,
            mean_timing_error_ms=mean_timing_error,
            timing_error_std_ms=timing_std,
            pitch_accuracy_percent=pitch_accuracy,
            timing_accuracy_percent=timing_accuracy,
            overall_score=overall_score,
            mistakes=mistakes
        )
    
    def _print_results(self, metrics: PerformanceMetrics):
        """Print results to console."""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Notes in Score:        {metrics.total_notes_in_score}")
        print(f"  Notes Detected:        {metrics.total_notes_detected}")
        print(f"  Correct Notes:         {metrics.correct_notes}")
        
        print(f"\n❌ MISTAKES:")
        print(f"  Wrong Notes:           {metrics.wrong_notes}")
        print(f"  Missing Notes:         {metrics.missing_notes}")
        print(f"  Extra Notes:           {metrics.extra_notes}")
        print(f"  Timing Errors:         {metrics.timing_errors}")
        
        print(f"\n⏱️  TIMING:")
        print(f"  Mean Timing Error:     {metrics.mean_timing_error_ms:.2f} ms")
        print(f"  Timing Std Dev:        {metrics.timing_error_std_ms:.2f} ms")
        
        print(f"\n📈 ACCURACY:")
        print(f"  Pitch Accuracy:        {metrics.pitch_accuracy_percent:.1f}%")
        print(f"  Timing Accuracy:       {metrics.timing_accuracy_percent:.1f}%")
        
        print(f"\n⭐ OVERALL SCORE:        {metrics.overall_score:.1f}/100")
        
        # Print detailed mistakes
        if metrics.mistakes:
            print(f"\n📝 DETAILED MISTAKES ({len(metrics.mistakes)}):")
            print("-"*60)
            for i, mistake in enumerate(metrics.mistakes, 1):
                print(f"\n{i}. [{mistake.timestamp:.3f}s] {mistake.error_type.upper().replace('_', ' ')}")
                
                if mistake.expected_note is not None:
                    try:
                        import librosa
                        note_name = librosa.midi_to_note(mistake.expected_note)
                        print(f"   Expected: {note_name} (MIDI {mistake.expected_note})")
                    except:
                        print(f"   Expected: MIDI {mistake.expected_note}")
                
                if mistake.detected_note is not None:
                    try:
                        import librosa
                        note_name = librosa.midi_to_note(mistake.detected_note)
                        print(f"   Detected: {note_name} (MIDI {mistake.detected_note})")
                    except:
                        print(f"   Detected: MIDI {mistake.detected_note}")
                
                if mistake.timing_error_ms is not None:
                    print(f"   Timing Error: {mistake.timing_error_ms:+.1f} ms")
        else:
            print("\n✅ No mistakes detected!")
        
        print("\n" + "="*60)
    
    def _save_json_report(self, metrics: PerformanceMetrics, output_path: str):
        """Save detailed report to JSON file."""
        # Convert dataclasses to dictionaries
        report = {
            'summary': {
                'total_notes_in_score': metrics.total_notes_in_score,
                'total_notes_detected': metrics.total_notes_detected,
                'correct_notes': metrics.correct_notes,
                'wrong_notes': metrics.wrong_notes,
                'missing_notes': metrics.missing_notes,
                'extra_notes': metrics.extra_notes,
                'timing_errors': metrics.timing_errors,
                'mean_timing_error_ms': round(metrics.mean_timing_error_ms, 2),
                'timing_error_std_ms': round(metrics.timing_error_std_ms, 2),
                'pitch_accuracy_percent': round(metrics.pitch_accuracy_percent, 1),
                'timing_accuracy_percent': round(metrics.timing_accuracy_percent, 1),
                'overall_score': round(metrics.overall_score, 1)
            },
            'mistakes': [
                {
                    'timestamp': round(m.timestamp, 3),
                    'error_type': m.error_type,
                    'expected_note': m.expected_note,
                    'detected_note': m.detected_note,
                    'expected_time': round(m.expected_time, 3) if m.expected_time else None,
                    'detected_time': round(m.detected_time, 3) if m.detected_time else None,
                    'timing_error_ms': round(m.timing_error_ms, 1) if m.timing_error_ms else None
                }
                for m in metrics.mistakes
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python midi_comparator.py <performance.mid> <score.mid> [output_report.json]")
        print("\nExample:")
        print("  python midi_comparator.py performance.mid score.mid report.json")
        sys.exit(1)
    
    performance_path = sys.argv[1]
    score_path = sys.argv[2]
    output_json = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Check files exist
    if not Path(performance_path).exists():
        print(f"Error: Performance MIDI file not found: {performance_path}")
        sys.exit(1)
    
    if not Path(score_path).exists():
        print(f"Error: Score MIDI file not found: {score_path}")
        sys.exit(1)
    
    try:
        comparator = MIDIComparator(timing_tolerance_ms=30, pitch_tolerance_semitones=0.5)
        comparator.compare(performance_path, score_path, output_json)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
