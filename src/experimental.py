"""
Percussion Practice Tool - Timing Accuracy Analyzer
For mridangam and other hand drums
Detects drum hits and compares timing to a reference pattern
"""

import sys
import librosa
import numpy as np
import mido
from typing import List, Tuple, Dict
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION PARAMETERS - Tune these for different recording conditions
# ============================================================================

# Onset detection sensitivity (lower = more sensitive, picks up quieter hits)
# Range: 0.1 (very sensitive) to 0.5 (only loud hits)
# Recommended: 0.2 for clean recordings, 0.3 for noisy environments
ONSET_THRESHOLD = 0.2

# Minimum time between detected onsets in seconds (prevents double-detection)
# Mridangam hits are typically > 0.05s apart even in fast passages
MIN_ONSET_INTERVAL = 0.05

# Timing tolerance for "on-time" classification (in seconds)
# Within this window = "on time", outside = "early" or "late"
TIMING_TOLERANCE = 0.2  # 200ms is more forgiving for loose timing

# Energy threshold - ignore very quiet hits (0.0 to 1.0)
# Helps filter out background noise or weak finger taps
MIN_HIT_ENERGY = 0.1


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect percussion onsets (drum hits) in audio signal.
    
    Optimizations for hand drums:
    - backtrack=True: Refines onset times to actual attack point
    - pre_max/post_max: Finds local maximum energy around onset
    - delta: Controls sensitivity (lower = more sensitive)
    
    Args:
        y: Audio time series (numpy array)
        sr: Sample rate
        
    Returns:
        onset_times: Array of onset times in seconds
    """
    # Use librosa's onset detection with percussion-optimized parameters
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        backtrack=True,      # Refine to true attack point
        pre_max=3,           # Look 3 frames before for local max
        post_max=3,          # Look 3 frames after for local max
        pre_avg=3,           # Average over 3 frames before
        post_avg=5,          # Average over 5 frames after
        delta=ONSET_THRESHOLD,  # Sensitivity threshold
        wait=int(MIN_ONSET_INTERVAL * sr / 512),  # Min frames between onsets
        units='frames'
    )
    
    # Convert frame indices to time in seconds
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    return onset_times


def filter_weak_hits(y: np.ndarray, sr: int, onset_times: np.ndarray) -> np.ndarray:
    """
    Filter out very quiet hits that might be noise or accidental taps.
    
    For each onset, measure the RMS energy in a small window around it.
    Remove onsets with energy below MIN_HIT_ENERGY threshold.
    
    Args:
        y: Audio time series
        sr: Sample rate
        onset_times: Detected onset times in seconds
        
    Returns:
        filtered_onset_times: Onsets with weak hits removed
    """
    window_size = int(0.05 * sr)  # 50ms window around each hit
    energies = []
    
    for onset_time in onset_times:
        onset_sample = int(onset_time * sr)
        start = max(0, onset_sample - window_size // 2)
        end = min(len(y), onset_sample + window_size // 2)
        segment = y[start:end]
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(segment**2))
        energies.append(rms)
    
    # Normalize energies to 0-1 range
    energies = np.array(energies)
    if len(energies) > 0 and np.max(energies) > 0:
        energies = energies / np.max(energies)
    
    # Filter onsets by energy threshold
    filtered_onsets = onset_times[energies >= MIN_HIT_ENERGY]
    
    return filtered_onsets


def compare_to_reference(detected_times: np.ndarray, 
                         reference_times: np.ndarray,
                         tolerance: float = TIMING_TOLERANCE) -> List[Dict]:
    """
    Compare detected hits to reference pattern and calculate timing errors.
    
    For each reference beat, find the nearest detected hit and calculate error.
    Classify as early, late, or on-time based on tolerance.
    
    Args:
        detected_times: Array of detected onset times
        reference_times: Array of expected beat times
        tolerance: Time window (seconds) for "on time" classification
        
    Returns:
        List of dicts with timing analysis for each reference beat
    """
    results = []
    
    for ref_time in reference_times:
        # Find nearest detected hit to this reference beat
        if len(detected_times) == 0:
            results.append({
                'reference_time': ref_time,
                'detected_time': None,
                'error': None,
                'status': 'MISSED'
            })
            continue
            
        # Calculate distance to all detected hits
        distances = np.abs(detected_times - ref_time)
        nearest_idx = np.argmin(distances)
        nearest_time = detected_times[nearest_idx]
        error = nearest_time - ref_time  # Positive = late, negative = early
        
        # Classify timing accuracy
        if abs(error) <= tolerance:
            status = 'ON TIME'
        elif error > 0:
            status = 'LATE'
        else:
            status = 'EARLY'
        
        results.append({
            'reference_time': ref_time,
            'detected_time': nearest_time,
            'error': error,
            'error_ms': error * 1000,  # Convert to milliseconds for readability
            'status': status
        })
    
    return results


def generate_metronome_pattern(bpm: float, n_beats: int, start_time: float = 0.0) -> np.ndarray:
    """
    Generate a reference timing pattern at given BPM.
    
    Args:
        bpm: Beats per minute
        n_beats: Number of beats to generate
        start_time: Start time in seconds
        
    Returns:
        Array of reference beat times
    """
    beat_interval = 60.0 / bpm
    reference_times = np.arange(n_beats) * beat_interval + start_time
    return reference_times


# ============================================================================
# PHASE 3 HELPER FUNCTIONS - Vectorized MIDI Analysis
# ============================================================================

@dataclass
class MIDINotes:
    """Vectorized note representation from MIDI."""
    onsets: np.ndarray      # Shape: (n_notes,)
    durations: np.ndarray   # Shape: (n_notes,)
    pitches: np.ndarray     # Shape: (n_notes,)
    velocities: np.ndarray  # Shape: (n_notes,)


def _extract_midi_notes_vectorized(midi_file: mido.MidiFile) -> MIDINotes:
    """Extract notes from MIDI file into vectorized format."""
    if midi_file.ticks_per_beat <= 0:
        raise ValueError("Invalid MIDI file: ticks_per_beat must be positive")
    
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500000  # Default tempo in microseconds per beat
    
    notes_data = {'onsets': [], 'durations': [], 'pitches': [], 'velocities': []}
    active_notes = {}
    
    # Process all tracks
    for track in midi_file.tracks:
        track_time = 0.0
        
        for msg in track:
            track_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            
            elif msg.type == 'note_on' and msg.velocity > 0:
                if 0 <= msg.note <= 127:
                    key = (msg.channel, msg.note, len(active_notes))
                    active_notes[key] = {'onset': track_time, 'velocity': msg.velocity, 'note': msg.note}
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Find matching note_on
                matching_key = None
                latest_time = -1
                
                for key, note_data in active_notes.items():
                    if key[0] == msg.channel and key[1] == msg.note and note_data['onset'] > latest_time:
                        latest_time = note_data['onset']
                        matching_key = key
                
                if matching_key:
                    note_data = active_notes[matching_key]
                    onset = note_data['onset']
                    duration = max(track_time - onset, 0.01)
                    
                    notes_data['onsets'].append(onset)
                    notes_data['durations'].append(duration)
                    notes_data['pitches'].append(note_data['note'])
                    notes_data['velocities'].append(note_data['velocity'])
                    
                    del active_notes[matching_key]
    
    # Convert to numpy arrays and sort by onset
    for key in notes_data:
        notes_data[key] = np.array(notes_data[key])
    
    if len(notes_data['onsets']) > 0:
        sort_idx = np.argsort(notes_data['onsets'])
        for key in notes_data:
            notes_data[key] = notes_data[key][sort_idx]
    
    return MIDINotes(**notes_data)


def _get_tempo_bpm(midi_file: mido.MidiFile) -> float:
    """Extract tempo in BPM from MIDI file."""
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return 60000000.0 / msg.tempo  # Convert microseconds per beat to BPM
    return 120.0  # Default


def _detect_subdivision(interval: float, beat_duration: float) -> str:
    """Classify interval into subdivision type using pattern matching."""
    ratio = interval / beat_duration
    match ratio:
        case r if r >= 2.0:      return "whole_or_half"
        case r if r >= 1.0:      return "quarter"
        case r if r >= 0.5:      return "eighth"
        case r if r >= 0.25:     return "sixteenth"
        case r if r >= 0.125:    return "thirty_second"
        case _:                  return "very_fast_run"


def _detect_sectional_tempo_drift(onset_times: np.ndarray, reference_bpm: float, window_size: int = 4) -> dict:
    """Detect rushing/dragging using a fixed metronome grid (no tempo estimation)."""
    if len(onset_times) < window_size + 1:
        return {
            'variance_ms': 0.0,
            'overall_trend': 'stable',
            'dragging_sections': np.array([]),
            'rushing_sections': np.array([])
        }

    beat_duration = 60.0 / reference_bpm
    reference_times = onset_times[0] + np.arange(len(onset_times)) * beat_duration

    # Error relative to metronome grid (seconds)
    errors = onset_times - reference_times

    # Sliding window mean error (seconds)
    window_mean_errors = np.array([
        np.mean(errors[i:i + window_size])
        for i in range(len(errors) - window_size + 1)
    ])

    # Trend of error: positive trend => increasingly late (dragging)
    error_trend = np.gradient(window_mean_errors)

    # Threshold in seconds per window step
    drift_threshold = 0.02  # 20 ms
    dragging_sections = np.where(error_trend > drift_threshold)[0]
    rushing_sections = np.where(error_trend < -drift_threshold)[0]

    return {
        'variance_ms': np.std(window_mean_errors) * 1000.0,
        'overall_trend': 'dragging' if np.mean(error_trend) > 0 else ('rushing' if np.mean(error_trend) < 0 else 'stable'),
        'dragging_sections': dragging_sections,
        'rushing_sections': rushing_sections
    }


def _analyze_rhythm_vectorized(score: MIDINotes, 
                               performance: MIDINotes,
                               beat_duration: float,
                               tolerance: float = 0.2) -> dict:
    """Fully vectorized rhythm analysis using NumPy broadcasting.
    
    Only analyzes performance notes - doesn't penalize for missing notes.
    """
    
    # Align notes: for each performance note, find nearest score note
    # Shape: (n_perf, n_score)
    time_diffs = np.abs(performance.onsets[:, None] - score.onsets[None, :])
    # For each perf note, find index of closest score note
    matches = np.argmin(time_diffs, axis=1)  # Shape: (n_perf,)
    
    # Get matched score onsets and durations
    matched_score_onsets = score.onsets[matches]  # Shape: (n_perf,)
    matched_score_durations = score.durations[matches]  # Shape: (n_perf,)
    
    # Match distances for each performance note
    match_distances = time_diffs[np.arange(len(performance.onsets)), matches]
    valid_mask = match_distances < 2.0  # Shape: (n_perf,)
    
    # Filter to valid matches only
    perf_onsets_valid = performance.onsets[valid_mask]
    perf_durations_valid = performance.durations[valid_mask]
    score_onsets_valid = matched_score_onsets[valid_mask]
    score_durations_valid = matched_score_durations[valid_mask]
    
    if len(perf_onsets_valid) == 0:
        return {
            'timing_errors': [],
            'tie_errors': [],
            'subdivision_errors': [],
            'mean_timing_error_ms': 0.0,
            'total_mistakes': 0,
            'accuracy_pct': 0.0
        }
    
    # 1. Timing errors (vectorized)
    timing_errors = perf_onsets_valid - score_onsets_valid
    timing_mask = np.abs(timing_errors) > tolerance
    
    # 2. Duration errors (ties) - vectorized
    duration_errors = perf_durations_valid - score_durations_valid
    tie_mask = np.abs(duration_errors) > (beat_duration * 0.25)  # Quarter beat tolerance
    
    # 3. Subdivision comparison - vectorized
    perf_intervals = np.diff(perf_onsets_valid) if len(perf_onsets_valid) > 1 else np.array([])
    score_intervals = np.diff(score_onsets_valid) if len(score_onsets_valid) > 1 else np.array([])
    
    subdivision_mask = np.zeros(len(perf_intervals), dtype=bool)
    if len(perf_intervals) > 0:
        perf_ratios = perf_intervals / beat_duration
        score_ratios = score_intervals / beat_duration
        
        # Quantize to bins using vectorized digitize
        bins = np.array([0, 0.125, 0.25, 0.5, 1.0, 2.0, np.inf])
        perf_subdivs = np.digitize(perf_ratios, bins)
        score_subdivs = np.digitize(score_ratios, bins)
        
        subdivision_mask = perf_subdivs != score_subdivs
    
    # Combine error masks with logical operations
    timing_errors_list = np.where(timing_mask)[0].tolist()
    tie_errors_list = np.where(tie_mask)[0].tolist()
    subdivision_errors_list = np.where(subdivision_mask)[0].tolist() if len(subdivision_mask) > 0 else []
    
    # Calculate total unique errors
    all_error_indices = set(timing_errors_list) | set(tie_errors_list) | set(subdivision_errors_list)
    total_errors = len(all_error_indices)
    
    # Calculate mean timing error: only for notes with actual timing errors
    mean_timing_error_all = np.mean(np.abs(timing_errors)) * 1000  # All notes
    if len(timing_errors_list) > 0:
        mean_timing_error_problematic = np.mean(np.abs(timing_errors[timing_errors_list])) * 1000  # Only problematic
    else:
        mean_timing_error_problematic = 0.0
    
    # Create per-note error details for table display
    note_details = []
    for i in range(len(perf_onsets_valid)):
        errors = []
        error_values = []
        
        if i in timing_errors_list:
            errors.append('Timing')
            error_values.append(f"{timing_errors[i]*1000:+.1f}ms")
        
        if i in tie_errors_list:
            errors.append('Duration')
            error_values.append(f"{duration_errors[i]*1000:+.1f}ms")
        
        if i in subdivision_errors_list:
            errors.append('Rhythm')
            error_values.append('')
        
        note_details.append({
            'note_num': i + 1,
            'time': perf_onsets_valid[i],
            'errors': errors,
            'error_values': error_values
        })
    
    return {
        'timing_errors': timing_errors_list,
        'tie_errors': tie_errors_list,
        'subdivision_errors': subdivision_errors_list,
        'mean_timing_error_ms': mean_timing_error_all,
        'mean_timing_error_problematic_ms': mean_timing_error_problematic,
        'total_mistakes': int(total_errors),
        'accuracy_pct': (1 - total_errors / len(perf_onsets_valid)) * 100 if len(perf_onsets_valid) > 0 else 0,
        'note_details': note_details
    }


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_percussion_timing(audio_path: str, 
                              reference_bpm: float,
                              n_reference_beats: int = None) -> Dict:
    """
    Complete analysis pipeline for percussion timing accuracy.
    
    Args:
        audio_path: Path to audio file
        reference_bpm: Expected tempo (required)
        n_reference_beats: Number of expected beats (if None, uses detected onsets)
        
    Returns:
        Dictionary containing full analysis results
    """
    # Load audio file
    print(f"Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves original sample rate
    except Exception as e:
        print(f"ERROR: Could not load audio file: {e}")
        sys.exit(1)
    
    audio_duration = len(y) / sr
    print(f"Duration: {audio_duration:.2f}s, Sample rate: {sr} Hz")
    
    # Detect onsets
    print("\nDetecting drum hits...")
    onset_times = detect_onsets(y, sr)
    print(f"Initial detection: {len(onset_times)} hits")
    
    # Filter weak hits
    # onset_times = filter_weak_hits(y, sr, onset_times)
    # print(f"After filtering weak hits: {len(onset_times)} hits")
    
    if len(onset_times) == 0:
        print("WARNING: No drum hits detected!")
        print("Possible issues:")
        print("  - Audio file might be too quiet")
        print(f"  - Try reducing ONSET_THRESHOLD (currently {ONSET_THRESHOLD})")
        print(f"  - Try reducing MIN_HIT_ENERGY (currently {MIN_HIT_ENERGY})")
        return {}
    
    if reference_bpm is None:
        raise ValueError("reference_bpm is required. Provide a fixed BPM for metronome comparison.")
    
    if n_reference_beats is None:
        n_reference_beats = len(onset_times)
    
    reference_times = generate_metronome_pattern(reference_bpm, n_reference_beats, 
                                                  start_time=onset_times[0] if len(onset_times) > 0 else 0)
    
    # Compare to reference
    print(f"\nComparing to reference pattern ({reference_bpm:.1f} BPM)...")
    timing_results = compare_to_reference(onset_times, reference_times)
    
    # Calculate statistics
    errors = [r['error_ms'] for r in timing_results if r['error_ms'] is not None]
    if errors:
        mean_error = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        
        n_on_time = sum(1 for r in timing_results if r['status'] == 'ON TIME')
        n_early = sum(1 for r in timing_results if r['status'] == 'EARLY')
        n_late = sum(1 for r in timing_results if r['status'] == 'LATE')
        n_missed = sum(1 for r in timing_results if r['status'] == 'MISSED')
        
        accuracy_pct = (n_on_time / len(timing_results)) * 100
    else:
        mean_error = max_error = accuracy_pct = 0
        n_on_time = n_early = n_late = n_missed = 0
    
    return {
        'audio_duration': audio_duration,
        'detected_onsets': onset_times,
        'reference_pattern': reference_times,
        'reference_bpm': reference_bpm,
        'timing_results': timing_results,
        'statistics': {
            'mean_error_ms': mean_error,
            'max_error_ms': max_error,
            'accuracy_pct': accuracy_pct,
            'n_on_time': n_on_time,
            'n_early': n_early,
            'n_late': n_late,
            'n_missed': n_missed
        }
    }


def print_detailed_results(analysis: Dict):
    """Print human-readable analysis results."""
    if not analysis:
        return
    
    print("\n" + "="*70)
    print("TIMING ANALYSIS RESULTS")
    print("="*70)
    
    stats = analysis['statistics']
    print(f"\nOverall Accuracy: {stats['accuracy_pct']:.1f}%")
    print(f"Average timing error: {stats['mean_error_ms']:.1f} ms")
    print(f"Maximum timing error: {stats['max_error_ms']:.1f} ms")
    
    print(f"\nBreakdown:")
    print(f"  ✓ On time: {stats['n_on_time']}")
    print(f"  ↑ Early:   {stats['n_early']}")
    print(f"  ↓ Late:    {stats['n_late']}")
    print(f"  ✗ Missed:  {stats['n_missed']}")
    
    print(f"\nBeat-by-beat analysis:")
    print("-" * 70)
    print(f"{'Beat':<6} {'Expected':<12} {'Actual':<12} {'Error':<12} {'Status':<10}")
    print("-" * 70)
    
    for i, result in enumerate(analysis['timing_results'][:20], 1):  # Show first 20
        ref_time = result['reference_time']
        det_time = result['detected_time']
        error_ms = result.get('error_ms', 0)
        status = result['status']
        
        # Format strings
        ref_str = f"{ref_time:.3f}s"
        det_str = f"{det_time:.3f}s" if det_time is not None else "---"
        err_str = f"{error_ms:+.1f} ms" if error_ms is not None else "---"
        
        # Add emoji/symbol for visual feedback
        if status == 'ON TIME':
            symbol = '✓'
        elif status == 'EARLY':
            symbol = '↑'
        elif status == 'LATE':
            symbol = '↓'
        else:
            symbol = '✗'
        
        print(f"{i:<6} {ref_str:<12} {det_str:<12} {err_str:<12} {symbol} {status}")
    
    if len(analysis['timing_results']) > 20:
        print(f"... and {len(analysis['timing_results']) - 20} more beats")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python experimental.py <audio_file> <bpm> [score.mid]")
        print("\nExamples:")
        print("  python experimental.py performance.m4a 120")
        print("    → Runs PHASE 1 (timing analysis only)")
        print("\n  python experimental.py performance.m4a 120 score.mid")
        print("    → Runs PHASE 1 (timing analysis)")
        print("    → Runs PHASE 2 (audio to MIDI conversion)")
        print("    → Runs PHASE 3 (comprehensive mistake detection)")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    try:
        reference_bpm = float(sys.argv[2])
    except ValueError:
        print("ERROR: BPM must be a number (e.g., 120)")
        sys.exit(1)
    score_midi_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # ========================================================================
    # PHASE 1: PERCUSSION TIMING ANALYSIS (Independent - Always Runs)
    # ========================================================================
    # Run silently for internal use
    analysis = analyze_percussion_timing(audio_path, reference_bpm=reference_bpm)
    
    # ========================================================================
    # PHASE 2: AUDIO TO MIDI CONVERSION (Prerequisite for Phase 3)
    # ========================================================================
    if score_midi_path:
        print("\n" + "="*70)
        print("PHASE 2: AUDIO TO MIDI CONVERSION")
        print("="*70)
        
        from performance_to_midi import PerformanceToMIDI
        from pathlib import Path
        
        print(f"Converting audio to MIDI: {audio_path}")
        converter = PerformanceToMIDI()
        performance_midi_path = converter.convert(audio_path)
        print(f"✓ MIDI conversion complete: {performance_midi_path}")
        
        # ====================================================================
        # PHASE 3: COMPREHENSIVE MISTAKE DETECTION (Vectorized Analysis)
        # ====================================================================
        print("\n" + "="*70)
        print("PHASE 3: COMPREHENSIVE MISTAKE DETECTION")
        print("="*70)
        
        import mido
        
        # Load MIDI files
        print(f"Loading performance MIDI: {performance_midi_path}")
        perf_midi = mido.MidiFile(performance_midi_path)
        perf_notes = _extract_midi_notes_vectorized(perf_midi)
        
        print(f"Loading score MIDI: {score_midi_path}")
        score_midi = mido.MidiFile(score_midi_path)
        score_notes = _extract_midi_notes_vectorized(score_midi)
        
        # Get tempo info
        perf_tempo = _get_tempo_bpm(perf_midi)
        score_tempo = _get_tempo_bpm(score_midi)
        beat_duration = 60.0 / score_tempo
        
        print(f"Performance tempo: {perf_tempo:.1f} BPM")
        print(f"Score tempo: {score_tempo:.1f} BPM")
        
        # Perform vectorized analysis
        print("\nAnalyzing rhythm and timing...")
        metrics = _analyze_rhythm_vectorized(score_notes, perf_notes, beat_duration)
        
        print("\n" + "="*70)
        print("📊 MISTAKE DETECTION RESULTS")
        print("="*70)
        print(f"\nTiming Analysis:")
        print(f"  Mean Timing Error (all notes): {metrics['mean_timing_error_ms']:.1f} ms")
        if len(metrics['timing_errors']) > 0:
            print(f"  Mean Timing Error (problematic notes only): {metrics['mean_timing_error_problematic_ms']:.1f} ms")
        print(f"  Timing Errors Found: {len(metrics['timing_errors'])}")
        
        print(f"\nRhythm Analysis:")
        print(f"  Subdivision Errors: {len(metrics['subdivision_errors'])}")
        print(f"  Tie/Duration Errors: {len(metrics['tie_errors'])}")
        
        print(f"\nOverall Performance:")
        print(f"  Total Mistakes: {metrics['total_mistakes']}")
        print(f"  Accuracy: {metrics['accuracy_pct']:.1f}%")
        
        # Per-note error table
        print(f"\n{'Note Details:':-<70}")
        print(f"{'Note':<6} {'Time (s)':<10} {'Errors':<40}")
        print("-" * 70)
        
        for detail in metrics['note_details']:
            note_num = detail['note_num']
            time = detail['time']
            
            if len(detail['errors']) > 0:
                # Build error string
                error_parts = []
                for i, err_type in enumerate(detail['errors']):
                    if detail['error_values'][i]:
                        error_parts.append(f"{err_type} ({detail['error_values'][i]})")
                    else:
                        error_parts.append(err_type)
                error_str = ", ".join(error_parts)
                print(f"{note_num:<6} {time:<10.2f} {error_str}")
        
        # Tempo drift analysis
        print(f"\nTempo Stability:")
        drift_analysis = _detect_sectional_tempo_drift(perf_notes.onsets, reference_bpm)
        print(f"  Sectional Variance: {drift_analysis['variance_ms']:.2f} ms")
        print(f"  Overall Trend: {drift_analysis['overall_trend']}")
        if len(drift_analysis['dragging_sections']) > 0:
            print(f"  Dragging in sections: {drift_analysis['dragging_sections'].tolist()}")
        if len(drift_analysis['rushing_sections']) > 0:
            print(f"  Rushing in sections: {drift_analysis['rushing_sections'].tolist()}")
        
        print("\n" + "="*70)
    
    else:
        print("\n" + "="*70)
        print("No MIDI score provided - Phases 2 & 3 skipped")
        print("="*70)
        print("\nTo run comprehensive mistake detection, provide a MIDI score:")
        print(f"  python experimental.py {audio_path} <your_score.mid>")