# VS Code Setup
# Moving Project to VS Code

## Quick Setup

### Option 1: Open Folder in VS Code
1. Open VS Code
2. Go to `File` → `Open Folder...` (or `Ctrl + K, Ctrl + O`)
3. Navigate to: `c:\Users\aniru_50kr0a\.cursor`
4. Click "Select Folder"

### Option 2: From Command Line
1. Open PowerShell or Command Prompt
2. Navigate to the project:
   ```powershell
   cd c:\Users\aniru_50kr0a\.cursor
   ```
3. Open in VS Code:
   ```powershell
   code .
   ```

## Project Structure

Once opened, you'll see:
```
.cursor/
├── music_detector/          # Main package
│   ├── __init__.py
│   ├── models.py
│   ├── audio_processor.py
│   ├── midi_score.py
│   ├── alignment.py
│   ├── mistake_detector.py
│   ├── detector.py
│   ├── audio_to_midi.py
│   ├── midi_validator.py
│   └── main.py
├── input/                   # Put audio files here
├── output/                  # Results go here
├── analyze_project_audio.py # Main analysis script
├── run_detector.py
├── convert_audio_to_midi.py
├── validate_midi.py
├── test_audio.py
└── requirements.txt
```

## Using the Terminal in VS Code

1. Open terminal: `Terminal` → `New Terminal` (or `` Ctrl + ` ``)
2. Make sure you're in the project directory
3. Run scripts:
   ```powershell
   python analyze_project_audio.py
   ```

## Install Dependencies

In VS Code terminal:
```powershell
pip install -r requirements.txt
```

## Recommended VS Code Extensions

- **Python** (by Microsoft) - Python language support
- **Pylance** - Python language server
- **Python Debugger** - Debugging support

## Running Scripts

All scripts work the same in VS Code:

1. **Analyze audio files:**
   ```powershell
   python analyze_project_audio.py
   ```

2. **Convert audio to MIDI:**
   ```powershell
   python convert_audio_to_midi.py input/your_file.mp3
   ```

3. **Test audio file:**
   ```powershell
   python test_audio.py
   ```

4. **Run mistake detector:**
   ```powershell
   python run_detector.py audio.mp3 score.mid
   ```

## Notes

- All code is already in `c:\Users\aniru_50kr0a\.cursor`
- Just open this folder in VS Code
- No need to copy/move files - everything is already there!
- The terminal in VS Code works exactly like PowerShell/CMD

Open the `music-audio-analysis` folder in VS Code for best results.