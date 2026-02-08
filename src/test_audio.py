# ...existing code from your old test_audio.py...
"""
Quick test script to check if audio file can be loaded.
Run this to diagnose audio loading issues.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_audio_file(audio_path: str):
    """Test if an audio file can be loaded."""
    print(f"Testing audio file: {audio_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(audio_path).exists():
        print(f"✗ File does not exist: {audio_path}")
        return False
    
    file_size = Path(audio_path).stat().st_size
    print(f"✓ File exists ({file_size:,} bytes)")
    
    # Try to load with librosa
    try:
        import librosa
        print("\n[1/2] Testing librosa.load()...")
        audio, sr = librosa.load(audio_path, sr=22050)
        print(f"✓ Successfully loaded!")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - Duration: {len(audio) / sr:.2f} seconds")
        print(f"  - Audio shape: {audio.shape}")
        
        # Try onset detection
        print("\n[2/2] Testing onset detection...")
        onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', backtrack=True)
        print(f"✓ Onset detection successful!")
        print(f"  - Detected {len(onsets)} onsets")
        if len(onsets) > 0:
            print(f"  - First onset: {onsets[0]:.3f}s")
            print(f"  - Last onset: {onsets[-1]:.3f}s")
        
        return True
        
    except ImportError as e:
        print(f"✗ librosa not installed: {str(e)}")
        print("\nInstall with: pip install librosa")
        return False
    except Exception as e:
        print(f"✗ Error loading audio: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the file in input folder
        default_file = Path(__file__).parent / "input" / "New Recording 3.m4a"
        if default_file.exists():
            audio_path = str(default_file)
            print(f"No file specified, using: {audio_path}\n")
        else:
            print("Usage: python test_audio.py <audio_file>")
            print("\nOr place an audio file in the 'input' folder and run without arguments")
            sys.exit(1)
    else:
        audio_path = sys.argv[1]
    
    success = test_audio_file(audio_path)
    sys.exit(0 if success else 1)
