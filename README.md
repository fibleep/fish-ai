# Gemini Fish Voice Assistant

This project implements a voice-controlled robotic fish assistant using Google's Gemini Live API. The fish can listen to voice commands, process them with Gemini, and respond with both voice and physical movements.

## Features

- Real-time voice interaction using Gemini Live API
- Live mouth movement synchronized with audio output
- Physical movements (head and tail) through tool calling
- Wake word detection based on sound energy
- Ambient sound effects during processing

## Requirements

- Python 3.8+
- Google API key with access to Gemini models
- ElevenLabs API key (for the action processor)
- Arduino-based fish hardware connected via serial

## Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ```

4. Make sure the fish hardware is connected via USB

## Usage

To start the fish assistant:

```
python gemini_fish.py
```

The fish will wait for a "wake word" (loud noise) and then:
1. Play a beep sound and start listening
2. Process your spoken command using Gemini
3. Respond with voice output and physical movements
4. Return to listening mode

## Project Structure

- `gemini_fish.py` - Main script with Gemini Live API integration
- `action_processor.py` - Handles audio processing and fish movements
- `tooling.py` - Controls the physical hardware movements
- Sound files (beep.wav, microwave_ambient.wav) - Audio cues

## How it Works

The system uses:
1. Gemini's Live API for real-time voice conversation
2. Function calling to trigger physical movements during speech
3. Energy-based audio analysis for mouth movement synchronization
4. Simple energy threshold for wake word detection

## Extending the Project

You can:
- Add more movement tools by extending the tool declarations
- Improve wake word detection with a more sophisticated model
- Customize the system instruction to change the fish's personality
