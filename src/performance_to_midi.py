"""
Performance to MIDI Converter
Converts audio (MP3/WAV) to MIDI by extracting pitch and timing data.
"""

import librosa
import numpy as np
import mido
from pathlib import Path
from typing import List, Tuple, Optional


class PerformanceToMIDI:
    """Converts audio performance to MIDI file."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def convert(self, audio_path: str, output_midi_path: Optional[str] = None) -> str:
        """
        Convert audio file to MIDI.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            output_midi_path: Path for output MIDI file (auto-generated if None)
        
        Returns:
            Path to created MIDI file
        """
        # Auto-generate output path if not provided
        if output_midi_path is None:
            audio_path_obj = Path(audio_path)
            output_midi_path = str(audio_path_obj.with_suffix('.mid'))
        
        print(f"Loading audio: {audio_path}")
        audio, sr = self._load_audio(audio_path)
        
        print("Detecting onsets...")
        onsets = self._detect_onsets(audio)
        print(f"Detected {len(onsets)} onsets")
        
        print("Extracting pitch at onsets...")
        notes = self._extract_notes(audio, onsets)
        print(f"Extracted {len(notes)} notes")
        
        print(f"Creating MIDI file: {output_midi_path}")
        self._create_midi_file(notes, output_midi_path)
        
        print(f"✓ Successfully converted to MIDI: {output_midi_path}")
        return output_midi_path
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(audio) == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {str(e)}")
    
    def _detect_onsets(self, audio: np.ndarray) -> np.ndarray:
        """Detect note onsets in audio with refined timing."""
        if len(audio) == 0:
            return np.array([])
        
        try:
            hop_length = 512
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=self.sample_rate,
                hop_length=hop_length,
                aggregate=np.median
            )
            
            # Stricter onset detection to avoid double-triggering
            min_separation = int(0.12 * self.sample_rate / hop_length)  # ~120ms
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=hop_length,
                backtrack=True,
                pre_max=20,
                post_max=20,
                pre_avg=100,
                post_avg=100,
                delta=0.25,
                wait=min_separation,
                units='frames'
            )
            
            if len(onset_frames) == 0:
                return np.array([])
            
            # Refine timing to the nearest local peak in the onset envelope
            refine_window = 2  # frames
            refined_frames = []
            for frame in onset_frames:
                start = max(0, frame - refine_window)
                end = min(len(onset_env) - 1, frame + refine_window)
                local_max = start + int(np.argmax(onset_env[start:end + 1]))
                refined_frames.append(local_max)
            
            refined_frames = np.array(refined_frames, dtype=int)
            refined_times = librosa.frames_to_time(
                refined_frames,
                sr=self.sample_rate,
                hop_length=hop_length
            )
            
            return refined_times if len(refined_times) > 0 else np.array([])
        except Exception as e:
            print(f"Warning: Onset detection failed: {str(e)}")
            return np.array([])
    
    def _extract_pitch_yin(self, audio: np.ndarray, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using YIN algorithm.
        Returns (midi_notes, times).
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # Convert Hz to MIDI note numbers
        midi_notes = np.full_like(f0, np.nan, dtype=np.float64)
        valid_mask = ~np.isnan(f0)
        midi_notes[valid_mask] = librosa.hz_to_midi(f0[valid_mask])
        
        # Time array
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        return midi_notes, times
    
    def _extract_notes(self, audio: np.ndarray, onsets: np.ndarray) -> List[dict]:
        """
        Extract note information at onset times.
        Returns list of {onset, note, duration, velocity} dicts.
        """
        midi_notes, times = self._extract_pitch_yin(audio)
        
        notes = []
        for i, onset_time in enumerate(onsets):
            # Find closest time index
            if onset_time < times[0]:
                idx = 0
            elif onset_time > times[-1]:
                idx = len(times) - 1
            else:
                idx = np.argmin(np.abs(times - onset_time))
            
            # Get pitch at this onset
            if 0 <= idx < len(midi_notes) and not np.isnan(midi_notes[idx]):
                midi_note = float(midi_notes[idx])
                midi_note = max(0, min(127, midi_note))  # Clamp to valid range
                midi_note = int(round(midi_note))  # Round to nearest integer
                
                # Estimate duration (time to next onset, or default)
                if i + 1 < len(onsets):
                    duration = onsets[i + 1] - onset_time
                else:
                    duration = 0.5  # Default duration for last note
                
                duration = max(0.1, min(2.0, duration))  # Clamp duration
                
                notes.append({
                    'onset': float(onset_time),
                    'note': midi_note,
                    'duration': float(duration),
                    'velocity': 100  # Default velocity
                })
        
        return notes
    
    def _create_midi_file(self, notes: List[dict], output_path: str):
        """Create MIDI file from note list using mido."""
        # Create MIDI file
        mid = mido.MidiFile(type=0)  # Type 0: single track
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo (120 BPM)
        tempo = mido.bpm2tempo(120)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        
        # Sort notes by onset time
        notes_sorted = sorted(notes, key=lambda x: x['onset'])
        
        # Convert notes to MIDI messages
        ticks_per_beat = mid.ticks_per_beat
        current_time = 0.0
        
        for note in notes_sorted:
            onset = note['onset']
            duration = note['duration']
            pitch = note['note']
            velocity = note['velocity']
            
            # Calculate delta time in ticks
            delta_time = onset - current_time
            delta_ticks = int(mido.second2tick(delta_time, ticks_per_beat, tempo))
            delta_ticks = max(0, delta_ticks)
            
            # Note on
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta_ticks))
            
            # Note off
            duration_ticks = int(mido.second2tick(duration, ticks_per_beat, tempo))
            duration_ticks = max(1, duration_ticks)
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration_ticks))
            
            current_time = onset + duration
        
        # Save MIDI file
        mid.save(output_path)


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python performance_to_midi.py <audio_file> [output_midi_file]")
        print("\nExamples:")
        print("  python performance_to_midi.py performance.mp3")
        print("  python performance_to_midi.py performance.mp3 performance.mid")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    try:
        converter = PerformanceToMIDI()
        converter.convert(audio_path, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
