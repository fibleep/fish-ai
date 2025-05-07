#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Gemini Live test with audio-only interaction - with strict turn management
"""

import asyncio
import os
import sys
import pyaudio
import numpy as np
from dotenv import load_dotenv
from google import genai
import time
import wave
import tempfile
import sounddevice as sd
import warnings
import subprocess
from collections import deque
import threading

# For Python versions less than 3.11
if sys.version_info < (3, 11, 0):
    import exceptiongroup
    import taskgroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Load environment variables
load_dotenv()

# Audio constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000  # Standard rate for speech recognition
RECEIVE_SAMPLE_RATE = 24000  # Gemini returns audio at 24kHz
CHUNK_SIZE = 512  # Same as entrypoint.py
MAX_RECORD_SECONDS = 10
SILENCE_THRESHOLD = 0.015  # Lower for more sensitivity
SILENCE_DURATION = 1.5     # Wait this many seconds of silence before ending turn
PRE_PADDING_MS = 500       # Add 500ms of silence before each response
POST_PADDING_MS = 800      # Add 800ms of silence after each response

# Generate silence padding
def generate_silence(duration_ms, sample_rate=RECEIVE_SAMPLE_RATE):
    """Generate a buffer of silence at the specified sample rate"""
    num_samples = int(duration_ms * sample_rate / 1000)
    return np.zeros(num_samples, dtype=np.int16).tobytes()

# Pre-generate padding
PRE_PADDING = generate_silence(PRE_PADDING_MS)
POST_PADDING = generate_silence(POST_PADDING_MS)

class RobustAudioPlayer:
    """A robust audio player that can handle stream errors"""
    def __init__(self, device_sample_rate):
        self.device_sample_rate = device_sample_rate
        self.audio_buffer = deque()
        self.is_playing = False
        self.temp_dir = tempfile.mkdtemp()
        self.chunk_counter = 0
        self.player_thread = None
        
        # For direct ALSA playback
        self.use_direct_playback = False
        if sys.platform == 'linux':
            # Check if aplay is available
            try:
                subprocess.run(['aplay', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.use_direct_playback = True
                print("Will use direct ALSA playback for reliability")
            except:
                print("aplay not found, falling back to sounddevice")
    
    def start(self):
        """Start the audio player"""
        if self.is_playing:
            self.stop()
            
        self.is_playing = True
        self.audio_buffer.clear()
        self.chunk_counter = 0
        
        # Add pre-padding silence
        self.add_audio_chunk(PRE_PADDING)
        
        if self.use_direct_playback:
            # For direct playback, we'll start the player thread
            self.player_thread = threading.Thread(target=self._direct_player_thread)
            self.player_thread.daemon = True
            self.player_thread.start()
        
        print("Audio player started with pre-padding")
    
    def stop(self):
        """Stop the audio player, adding post-padding first"""
        # Add post-padding silence
        if self.is_playing:
            self.add_audio_chunk(POST_PADDING)
            # Give a moment for the post-padding to be processed
            time.sleep(0.1)
        
        self.is_playing = False
        
        if self.player_thread and self.player_thread.is_alive():
            self.player_thread.join(timeout=1.0)
            self.player_thread = None
            
        print("Audio player stopped with post-padding")
    
    def add_audio_chunk(self, audio_data):
        """Add a chunk of audio data to be played"""
        if not self.is_playing:
            return
            
        # Add to buffer
        self.audio_buffer.append(audio_data)
        
        # If not using direct playback, play immediately with sounddevice
        if not self.use_direct_playback:
            self._play_chunk_with_sounddevice(audio_data)
    
    def _play_chunk_with_sounddevice(self, audio_data):
        """Play a single chunk using sounddevice (immediate playback)"""
        try:
            # Convert to float32
            audio = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio.astype(np.float32) / 32768.0
            
            # Simple linear resampling if needed
            if RECEIVE_SAMPLE_RATE != self.device_sample_rate:
                ratio = self.device_sample_rate / RECEIVE_SAMPLE_RATE
                target_len = int(len(audio_float) * ratio)
                resampled = np.zeros(target_len, dtype=np.float32)
                
                for i in range(target_len):
                    src_idx = i / ratio
                    src_idx_floor = int(src_idx)
                    src_idx_ceil = min(src_idx_floor + 1, len(audio_float) - 1)
                    fraction = src_idx - src_idx_floor
                    
                    resampled[i] = (1 - fraction) * audio_float[src_idx_floor] + \
                                   fraction * audio_float[src_idx_ceil]
                
                audio_float = resampled
            
            # Amplify
            audio_float = audio_float * 1.5
            
            # Clip to prevent distortion
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            # Play non-blocking
            sd.play(audio_float, self.device_sample_rate, blocking=False)
            
        except Exception as e:
            print(f"Error playing chunk with sounddevice: {e}")
    
    def _direct_player_thread(self):
        """Thread for direct ALSA playback via aplay"""
        # Wait for some initial data
        while self.is_playing and len(self.audio_buffer) < 3:
            time.sleep(0.1)
            
        if not self.is_playing:
            return
            
        try:
            # Start aplay process that reads from stdin
            aplay_cmd = ['aplay', '-r', str(RECEIVE_SAMPLE_RATE), '-f', 'S16_LE', '-c', '1', '-t', 'raw']
            
            # Removed specific device check - use default ALSA device
            # # Try with direct hardware access first
            # if os.path.exists('/proc/asound/cards'):
            #     with open('/proc/asound/cards', 'r') as f:
            #         cards_content = f.read()
            #         
            #     for line in cards_content.splitlines():
            #         if 'UACDemoV1.0' in line:
            #             # Found the UACDemo device
            #             for i, part in enumerate(line.split()):
            #                 if part.isdigit():
            #                     card_num = part
            #                     aplay_cmd.extend(['-D', f'plughw:{card_num},0'])
            #                     print(f"Using direct hardware device: plughw:{card_num},0")
            #                     break
            
            print(f"Starting direct playback with default ALSA device: {' '.join(aplay_cmd)}")
            process = subprocess.Popen(
                aplay_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0  # Unbuffered
            )
            
            # Process audio chunks as they arrive
            while self.is_playing or self.audio_buffer:
                if self.audio_buffer:
                    chunk = self.audio_buffer.popleft()
                    try:
                        process.stdin.write(chunk)
                        process.stdin.flush()
                    except Exception as e:
                        print(f"Error writing to aplay: {e}")
                        break
                else:
                    # No data yet, small sleep
                    time.sleep(0.01)
            
            # Clean up
            try:
                process.stdin.close()
                process.wait(timeout=1.0)
            except:
                process.kill()
                
        except Exception as e:
            print(f"Error in direct player thread: {e}")

class SimpleGeminiTest:
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options={"api_version": "v1alpha"}
        )
        
        # Using Gemini 2.0 Flash for stability
        self.model = "models/gemini-2.0-flash-exp"
        
        # Configure for audio response
        self.config = {
            "response_modalities": ["AUDIO"]  # Audio-only response
        }
        
        self.pyaudio = pyaudio.PyAudio()
        self.session = None
        self.need_new_session = True
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Find and configure audio devices
        self.find_and_set_devices()
        
        # Create temp directory for audio files
        self.temp_dir = tempfile.mkdtemp()
        print(f"Using temp directory for audio files: {self.temp_dir}")
        self.response_count = 0
        
        # Conversation state
        self.conversation_state = "IDLE"  # IDLE, LISTENING, PROCESSING, RESPONDING
        
        # Buffers
        self.recorded_audio = []
        
        # Track metrics
        self.recording_started = False
        self.min_recording_time = 1.0
        self.speech_detected = False
        self.silence_start = None
        self.start_time = None
        
        # Session management
        self.session_lock = asyncio.Lock()
        
        # Turn management
        self.current_turn_id = 0
        self.current_processing_turn = None
        self.pending_turns = asyncio.Queue()
        self.response_event = asyncio.Event()
        
        # Create robust audio player
        self.audio_player = RobustAudioPlayer(self.device_sample_rate)
    
    def find_and_set_devices(self):
        """Find and set audio devices"""
        devices = sd.query_devices()
        print("\nAvailable Audio Devices:")
        
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append('INPUT')
            if device['max_output_channels'] > 0:
                device_type.append('OUTPUT')
            
            print(f"{i}: {device['name']} "
                  f"({', '.join(device_type)}) "
                  f"- Default SR: {device['default_samplerate']}")

        # Look for UACDemo device first
        self.output_device = None
        for i, device in enumerate(devices):
            if "UACDemoV1.0" in device['name'] and device['max_output_channels'] > 0:
                self.output_device = i
                print(f"\nFound UACDemo device at index {i}")
                break

        if self.output_device is None:
            print("UACDemo device not found, using default output")
            self.output_device = sd.default.device[1]
        else:
            # Set environment variables as in entrypoint.py
            os.environ['SDL_AUDIODRIVER'] = 'alsa'
            os.environ['SDL_AUDIODEV'] = 'sysdefault:CARD=UACDemoV10'
            
        device_info = sd.query_devices(self.output_device)
        self.device_sample_rate = int(device_info['default_samplerate'])
        
        # Configure sounddevice defaults
        sd.default.device = (None, self.output_device)
        sd.default.samplerate = self.device_sample_rate
        sd.default.channels = (None, 1)
        sd.default.dtype = ('float32', 'float32')
        
        print(f"Using output device: {device_info['name']}")
        print(f"Sample rate: {self.device_sample_rate} Hz")
        
        # Find input device
        try:
            self.input_device = self.pyaudio.get_default_input_device_info()["index"]
            print(f"Using input device: {self.pyaudio.get_device_info_by_index(self.input_device)['name']}")
        except:
            # Fallback to a default if no device info available
            self.input_device = 0
            print(f"Using default input device (index 0)")
    
    async def send_text(self):
        """Allow user to type messages to Gemini."""
        while True:
            text = await asyncio.to_thread(input, "\nmessage > ")
            if text.lower() == "q":
                break
            
            if not text.strip():
                continue
                
            # Create a turn and queue it
            self.current_turn_id += 1
            turn_id = self.current_turn_id
            
            # Queue the text turn for processing
            turn_data = {
                'type': 'text',
                'id': turn_id,
                'content': text,
                'time': time.time()
            }
            
            await self.pending_turns.put(turn_data)
            print(f"Queued text input (turn {turn_id}): '{text}'")
            
            # Notify the processor that a new turn is available
            self.response_event.set()
    
    async def audio_input_stream(self):
        """Handle audio input in a structured way like entrypoint.py"""
        try:
            # Open microphone stream
            self.audio_stream = self.pyaudio.open(
            format=FORMAT,
            channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=self.input_device,
                frames_per_buffer=CHUNK_SIZE,
            )
            
            print("Microphone ready. Speak now...")
            
            while True:
                # Read audio from microphone
                data = self.audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Calculate energy
                audio_int16 = np.frombuffer(data, np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                energy = np.mean(np.abs(audio_float32))
                
                # Draw energy meter
                bars = int(energy * 50)
                energy_display = f"\rAudio: [{'#' * bars}{' ' * (50-bars)}] - {energy:.4f}"
                
                # State machine handling
                if self.conversation_state == "IDLE":
                    # Check for wake word (any loud sound)
                    if energy > SILENCE_THRESHOLD * 3:
                        print("\nStarting to listen...")
                        self.conversation_state = "LISTENING"
                        self.start_time = time.time()
                        self.silence_start = None
                        self.speech_detected = False
                        self.recorded_audio = []
                        self.recording_started = False
                
                elif self.conversation_state == "LISTENING":
                    print(energy_display, end="")
                    
                    # Add to buffer
                    self.recorded_audio.append(data)
                    
                    # Check if we've detected speech
                    if not self.speech_detected and energy > SILENCE_THRESHOLD * 1.5:
                        self.speech_detected = True
                        self.recording_started = True
                        print("\nSpeech detected!")
                    
                    # Check for end conditions
                    current_time = time.time()
                    
                    # End if we've recorded too long
                    if current_time - self.start_time > MAX_RECORD_SECONDS:
                        print("\nReached maximum recording time")
                        await self.finish_listening()
                    
                    # Check for silence after we've recorded for at least the minimum time
                    if (self.speech_detected and 
                        current_time - self.start_time > self.min_recording_time):
                        
                        if energy < SILENCE_THRESHOLD:
                            if self.silence_start is None:
                                self.silence_start = current_time
                            elif current_time - self.silence_start > SILENCE_DURATION:
                                print("\nSilence detected, finishing recording")
                                await self.finish_listening()
                        else:
                            self.silence_start = None
                
                # Add a small delay to prevent CPU overload
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"Error in audio stream: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(self, 'audio_stream') and self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
                self.audio_stream.close()
    
    async def ensure_session(self):
        """Ensure we have a valid session, creating a new one if needed"""
        if self.need_new_session or self.session is None:
            if self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except:
                    pass
            
            print("Creating new Gemini session...")
            session_ctx = self.client.aio.live.connect(model=self.model, config=self.config)
            self.session = await session_ctx.__aenter__()
            self.session_ctx = session_ctx
            self.need_new_session = False
            print("New session created successfully")
    
    async def finish_listening(self):
        """Process the recorded audio and send to Gemini"""
        if not self.recorded_audio or not self.speech_detected:
            print("No speech detected, canceling")
            self.conversation_state = "IDLE"
            return
        
        # Reset state immediately to allow new input
        self.conversation_state = "IDLE"
            
        # Create a turn and queue it
        self.current_turn_id += 1
        turn_id = self.current_turn_id
        
        # Save audio for debugging
        audio_file = os.path.join(self.temp_dir, f"input_{turn_id}.wav")
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SEND_SAMPLE_RATE)
            wf.writeframes(b''.join(self.recorded_audio))
        
        print(f"Saved input to {audio_file}")
        
        # Queue the audio turn for processing
        turn_data = {
            'type': 'audio',
            'id': turn_id,
            'content': self.recorded_audio.copy(),
            'time': time.time(),
            'file': audio_file
        }
        
        await self.pending_turns.put(turn_data)
        print(f"Queued audio input (turn {turn_id})")
        
        # Notify the processor that a new turn is available
        self.response_event.set()
    
    async def save_audio_for_reference(self, audio_data, turn_id):
        """Save audio for reference/debugging"""
        self.response_count += 1
        wav_path = os.path.join(self.temp_dir, f"response_{turn_id}_{self.response_count}.wav")
        
        # Save raw PCM as WAV
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(RECEIVE_SAMPLE_RATE)  # Gemini audio sample rate
            wf.writeframes(audio_data)
        
        print(f"Saved raw response to {wav_path} for reference")
        return wav_path
    
    async def process_turns(self):
        """Process turns in order, handling new priorities"""
        while True:
            try:
                # Wait for a turn to be available
                if self.pending_turns.empty():
                    # Clear the event since we're caught up
                    self.response_event.clear()
                    
                    # Wait for next turn notification
                    await self.response_event.wait()
                
                # Process all pending turns, always taking the newest one
                newest_turn = None
                while not self.pending_turns.empty():
                    turn = await self.pending_turns.get()
                    if newest_turn is None or turn['time'] > newest_turn['time']:
                        newest_turn = turn
                    
                    # Mark as done
                    self.pending_turns.task_done()
                
                if newest_turn:
                    # Process the newest turn
                    print(f"Processing turn {newest_turn['id']} (type: {newest_turn['type']})")
                    self.current_processing_turn = newest_turn
                    await self.process_turn(newest_turn)
                    self.current_processing_turn = None
                
            except Exception as e:
                print(f"Error in turn processor: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.5)
    
    async def process_turn(self, turn):
        """Process a single turn"""
        # Ensure we have a valid session
        async with self.session_lock:
            await self.ensure_session()
            
            try:
                # Send the appropriate content
                if turn['type'] == 'text':
                    print(f"Sending text (turn {turn['id']}): '{turn['content']}'")
                    await self.session.send(input=turn['content'], end_of_turn=True)
                elif turn['type'] == 'audio':
                    print(f"Sending audio (turn {turn['id']})")
                    combined_audio = b''.join(turn['content'])
                    await self.session.send(
                        input={"data": combined_audio, "mime_type": "audio/pcm"},
                        end_of_turn=True
                    )
                
                # Now receive the response
                print(f"\nWaiting for Gemini response (turn {turn['id']})...")
                turn_response = self.session.receive()
                
                # Stop any current playback
                self.audio_player.stop()
                
                # Start the audio player (which adds pre-padding)
                self.audio_player.start()
                
                full_audio_data = bytearray()
                has_text = False
                
                try:
                    # Process each response in the turn
                    async for response in turn_response:
                        # Check if a newer turn came in while we were processing
                        if not self.pending_turns.empty():
                            print(f"New turn available, stopping processing of turn {turn['id']}")
                            break
                        
                        if hasattr(response, 'data') and response.data:
                            # Stream audio data immediately
                            audio_data = response.data
                            full_audio_data.extend(audio_data)
                            
                            # Add to player
                            self.audio_player.add_audio_chunk(audio_data)
                            
                            print(f"Received and streaming audio chunk: {len(audio_data)} bytes")
                        
                        if hasattr(response, 'text') and response.text:
                            has_text = True
                            print(f"\nGemini (turn {turn['id']}): {response.text}")
                    
                    # Save the complete audio for reference
                    if full_audio_data:
                        print(f"\nReceived complete audio response ({len(full_audio_data)} bytes)")
                        
                        # Save with padding for better audio experience
                        padded_audio = bytearray()
                        padded_audio.extend(PRE_PADDING)
                        padded_audio.extend(full_audio_data)
                        padded_audio.extend(POST_PADDING)
                        
                        await self.save_audio_for_reference(bytes(padded_audio), turn['id'])
                        
                        # Also allow playback of the full file as fallback
                        if sys.platform == 'linux':
                            wav_path = os.path.join(self.temp_dir, f"response_{turn['id']}_{self.response_count}.wav")
                            print(f"You can also play the response manually: aplay {wav_path}")
                    
                    if not full_audio_data and not has_text:
                        print("No audio or text in response")
                    
                    # Let the audio finish playing
                    if not self.pending_turns.empty():
                        # New turn - stop current playback
                        self.audio_player.stop()
                    else:
                        # Stop the player (which adds post-padding)
                        self.audio_player.stop()
                        
                        # Give a moment for audio to finish playing
                        await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error in response processing: {e}")
                    self.need_new_session = True
                    self.audio_player.stop()
                
                # Only print this if there are no pending turns
                if self.pending_turns.empty():
                    print(f"\nTurn {turn['id']} complete, ready for your input")
                
            except Exception as e:
                print(f"Error processing turn {turn['id']}: {e}")
                self.need_new_session = True
                import traceback
                traceback.print_exc()
    
    async def run(self):
        print("Starting Gemini Live with real-time audio streaming...")
        
        try:
            # Start with a valid session
            async with self.session_lock:
                await self.ensure_session()
            
            print("\nConnected to Gemini. Speak or type to begin.")
            
            # Start with a greeting
            self.current_turn_id += 1
            greeting_turn = {
                'type': 'text',
                'id': self.current_turn_id,
                'content': "Hello Gemini, let's have a conversation.",
                'time': time.time()
            }
            await self.pending_turns.put(greeting_turn)
            self.response_event.set()
            
            # Start all tasks concurrently
            turn_processor = asyncio.create_task(self.process_turns())
            input_task = asyncio.create_task(self.audio_input_stream())
            text_task = asyncio.create_task(self.send_text())
            
            # Wait for text input to finish (when user types 'q')
            await text_task
            
            # Cancel other tasks
            turn_processor.cancel()
            input_task.cancel()
            
            print("Session ended by user.")
                
        except Exception as e:
            print(f"Error in session: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            if hasattr(self, 'audio_stream') and self.audio_stream and self.audio_stream.is_active():
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass
            
            # Stop audio player
            self.audio_player.stop()
            
            if self.session:
                try:
                    await self.session_ctx.__aexit__(None, None, None)
                except:
                    pass
            
            self.pyaudio.terminate()
            print("Session ended.")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        sys.exit(1)
        
    # Run the application
    test = SimpleGeminiTest()
    
    try:
        asyncio.run(test.run())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")