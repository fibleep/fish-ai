import pyaudio
import numpy as np
import time
import wave
import subprocess
import os
from threading import Thread
from queue import Queue
from brain import FishBrain
from action_processor import ActionProcessor
import asyncio

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
MAX_RECORD_SECONDS = 10
SILENCE_THRESHOLD = 0.15
SILENCE_DURATION = 4.5
FOLLOW_UP_LISTEN_TIMEOUT = 5.0 # Seconds to wait for user to start follow-up
FOLLOW_UP_SILENCE_DURATION = 2.0 # Seconds of silence to end follow-up recording
INITIAL_LISTEN_TIMEOUT = 7.0 # Seconds to wait for user to speak after initial beep

audio_queue = Queue()
is_recording = False
ap = None  # Will be initialized in main()
brain = None  # Will be initialized in main()
ambient_process = None
p = None
stream = None

# Get speaker device from environment variable, fallback to CONEXANT USB AUDIO (card 3)
SPEAKER_DEVICE = os.getenv('SPEAKER_DEVICE', 'plughw:3,0')
MIC_DEVICE_NAME = os.getenv('MIC_DEVICE_NAME', 'CONEXANT USB AUDIO')
MIC_DEVICE_INDEX = None

# Force TTS output to use device 3 (CONEXANT USB AUDIO)
os.environ['OUTPUT_DEVICE_INDEX'] = '3'

def get_mic_index(p_instance):
    """Find audio device index by name."""
    print("Available audio input devices:")
    for i in range(p_instance.get_device_count()):
        dev = p_instance.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"  {i}: {dev['name']}")
            if MIC_DEVICE_NAME in dev['name']:
                print(f"Found matching device: {dev['name']} at index {i}")
                return i
    print(f"Warning: Could not find audio device '{MIC_DEVICE_NAME}'. Using default.")
    return None

def detect_wake_word(audio_data, threshold=0.6):
    audio_int16 = np.frombuffer(audio_data, np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    energy = np.mean(np.abs(audio_float32))
    if energy > threshold:
        print(f"Wake word energy detected: {energy:.4f}")
        return True
    return False

def set_volume_max():
    """Set audio output volume to maximum (100%)"""
    try:
        # Set volume for the specific device
        print(f"🔊 Setting volume to 100% for device {SPEAKER_DEVICE}...")
        subprocess.run(["amixer", "-D", SPEAKER_DEVICE, "sset", "PCM", "100%"], 
                      check=False, capture_output=True)
        
        # Also try setting Master volume if it exists
        subprocess.run(["amixer", "-D", SPEAKER_DEVICE, "sset", "Master", "100%"], 
                      check=False, capture_output=True)
        
        # Try setting volume for the card number (extract from plughw:X,Y)
        if "plughw:" in SPEAKER_DEVICE:
            card_num = SPEAKER_DEVICE.split(":")[1].split(",")[0]
            subprocess.run(["amixer", "-c", card_num, "sset", "PCM", "100%"], 
                          check=False, capture_output=True)
            subprocess.run(["amixer", "-c", card_num, "sset", "Master", "100%"], 
                          check=False, capture_output=True)
        
        print("🔊 Volume set to maximum")
    except Exception as e:
        print(f"⚠️ Could not set volume: {e}")

def init_sound():
    beep = "beep.wav"
    microwave_beep = "microwave_beep.wav"
    microwave_ambient = "microwave_ambient.wav"
    
    # Set volume to maximum
    set_volume_max()
    
    return beep, microwave_beep, microwave_ambient

def play_sound(sound_file):
    try:
        subprocess.run(["aplay", "-q", "-D", SPEAKER_DEVICE, sound_file], check=True)
        print(f"Played sound: {sound_file} on {SPEAKER_DEVICE}")
    except Exception as e:
        print(f"Error playing sound on {SPEAKER_DEVICE}: {e}")
        try:
            subprocess.run(["aplay", "-q", sound_file], check=True)
            print(f"Played sound on default device: {sound_file}")
        except Exception as e:
            print(f"Error playing sound on default device: {e}")

def play_ambient_sound(sound_file):
    global ambient_process
    try:
        stop_ambient_sound()
        print(f"Starting ambient sound: {sound_file} on {SPEAKER_DEVICE}")
        ambient_process = subprocess.Popen(
            ["aplay", "-q", "-D", SPEAKER_DEVICE, sound_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Error playing ambient sound on {SPEAKER_DEVICE}: {e}")
        try:
            print(f"Attempting ambient sound on default device")
            ambient_process = subprocess.Popen(
                ["aplay", "-q", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Started ambient sound on default device")
        except Exception as e:
            print(f"Error playing ambient sound on default device: {e}")

def stop_ambient_sound():
    global ambient_process
    if ambient_process is not None:
        try:
            ambient_process.terminate()
            ambient_process.wait(timeout=1)
            ambient_process = None
            print("Stopped ambient sound")
        except Exception as e:
            print(f"Error stopping ambient sound: {e}")
            try:
                ambient_process.kill()
            except:
                pass
            ambient_process = None

def save_audio(frames, filename="command.wav"):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def open_wakeword_stream():
    global p, stream, MIC_DEVICE_INDEX
    if stream is not None and stream.is_active():
        print("Wake word stream already open and active.")
        return
    if p is None:
        p = pyaudio.PyAudio()
        MIC_DEVICE_INDEX = get_mic_index(p)

    print("Opening wake word stream...")
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=MIC_DEVICE_INDEX
    )
    print(f"Wake word stream opened on device index: {MIC_DEVICE_INDEX}")

def close_wakeword_stream():
    global p, stream
    if stream is not None:
        print("Closing wake word stream...")
        if stream.is_active():
             stream.stop_stream()
        stream.close()
        stream = None
        print("Wake word stream closed.")
    # Keep p instance alive until final cleanup

def wakeword_process(brain):
    global stream, is_recording
    threshold = 0.3
    open_wakeword_stream()

    while True:
        try:
            if is_recording:
                if stream is not None and stream.is_active():
                    close_wakeword_stream()
                time.sleep(0.1)
                continue

            if stream is None or not stream.is_active():
                 open_wakeword_stream()
                 time.sleep(0.1) # Give stream time to open

            if stream is not None and stream.is_active():
                 data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                 if detect_wake_word(data, threshold):
                     print("Wake word detected with energy detector")
                     # Set is_recording immediately to prevent duplicate triggers
                     is_recording = True
                     close_wakeword_stream() # Close before recording starts
                     # Use threading to handle the async call
                     import threading
                     def run_async_command():
                         loop = asyncio.new_event_loop()
                         asyncio.set_event_loop(loop)
                         try:
                             loop.run_until_complete(record_command(brain))
                         finally:
                             loop.close()
                     
                     thread = threading.Thread(target=run_async_command)
                     thread.start()
                 else:
                    time.sleep(0.01) # Short sleep only if not detected/recording
            else:
                print("Wake word stream is not active, waiting...")
                time.sleep(0.5)


        except IOError as e:
             if e.errno == -9988: # Input overflowed
                 print(f"Wake word stream overflow, restarting stream. {e}")
                 close_wakeword_stream()
                 time.sleep(0.1)
                 open_wakeword_stream()
             else:
                print(f"Error reading from wake word stream: {e}")
                close_wakeword_stream()
                time.sleep(1)


        except Exception as e:
            print(f"Error in wake word detection loop: {e}")
            close_wakeword_stream() # Close stream on other errors
            time.sleep(1)


def listen_and_record(listen_timeout, silence_duration_end, min_record_sec=1.0):
    frames = []
    global p, MIC_DEVICE_INDEX
    rec_stream = None

    if p is None:
        print("Error: PyAudio instance not initialized.")
        return None

    try:
        rec_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=MIC_DEVICE_INDEX
        )
    except Exception as e:
        print(f"Failed to open recording stream: {e}")
        return None

    print(f"Listening... (timeout in {listen_timeout:.1f}s) on device index: {MIC_DEVICE_INDEX}")
    initial_sound_detected = False
    listen_start_time = time.time()

    # 1. Listen for initial sound within timeout
    while time.time() - listen_start_time < listen_timeout:
        try:
            data = rec_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data) # Start buffering immediately
            
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            energy = np.mean(np.abs(audio_float32))

            if energy > SILENCE_THRESHOLD * 1.5:
                print("Initial speech detected.")
                initial_sound_detected = True
                break
            time.sleep(0.01)
        except IOError as e:
            print(f"Error reading during initial listen: {e}")
            time.sleep(0.1)

    if not initial_sound_detected:
        print("No sound detected within timeout.")
        rec_stream.stop_stream()
        rec_stream.close()
        return None

    # 2. Record until silence or max duration
    print("Recording...")
    start_time = time.time()
    silence_start = None

    # Trim leading silence buffered before speech detection
    # Heuristic: keep last 0.5 seconds before detected sound
    keep_chunks = int(0.5 * SAMPLE_RATE / CHUNK_SIZE)
    frames = frames[-keep_chunks:]


    while time.time() - start_time < MAX_RECORD_SECONDS:
        try:
            data = rec_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            energy = np.mean(np.abs(audio_float32))

            if time.time() - start_time > min_record_sec:
                if energy < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration_end:
                        print(f"Silence detected for {silence_duration_end:.1f}s, stopping recording.")
                        break
                else:
                    silence_start = None
            time.sleep(0.01)
        except IOError as e:
             print(f"Error reading during recording: {e}")
             time.sleep(0.1)

    print("Recording stopped.")
    rec_stream.stop_stream()
    rec_stream.close()

    if not frames:
        return None

    return frames

async def record_command(brain):
    global is_recording, ap
    # is_recording is already set to True in wake word detection
    turn_count = 0  # Initialize turn_count at the start

    try:
        # Move tail to indicate listening
        print("🐟 Moving tail to indicate listening...")
        ap.tooling.run_tool("MoveTail")

        print("🔔 Playing beep sound...")
        play_sound(beep_sound)

        # --- Initial Command Recording ---
        print("🎤 Starting initial command recording...")
        current_frames = listen_and_record(
            listen_timeout=INITIAL_LISTEN_TIMEOUT,
            silence_duration_end=SILENCE_DURATION,
            min_record_sec=1.5
        )

        if current_frames is None:
            print("❌ Initial command recording failed or timed out.")
            return # Exit if initial recording failed

        # --- Conversation Loop ---
        while current_frames is not None:
            turn_count += 1
            print(f"\n🔄 === CONVERSATION TURN {turn_count} ===")
            print("🎵 Playing processing ambient sound...")
            play_ambient_sound(microwave_ambient_sound)
            
            print("💾 Saving recorded audio...")
            audio_file = save_audio(current_frames, filename="user_command.wav")
            print(f"📁 Audio saved as: {audio_file}")
            
            # This is the main async call to the brain
            print("🧠 Sending audio to brain for processing...")
            response = await brain.process_audio_input(audio_file)
            print(f"🧠 Brain response received: '{response}'")
            
            print("🔄 Preprocessing response for TTS...")
            original_text, cleaned_text, arduino_tool_positions, mcp_tool_positions, audio = ap.preprocess(response)
            print(f"🧹 Cleaned text for TTS: '{cleaned_text}'")
            
            print("🔇 Stopping ambient sound...")
            stop_ambient_sound()
            
            print("🔔 Playing completion beep...")
            play_sound(microwave_beep_sound)
            time.sleep(0.2)
            
            # This part is now async and handles speaking + MCP tool execution
            print("🗣️ Starting speech synthesis and playback...")
            await ap.process_and_speak(original_text, cleaned_text, arduino_tool_positions, mcp_tool_positions, audio)
            
            print(f"🐟 Fish responded: '{response}'")

            # --- Listen for Follow-up ---
            print("👂 Listening for follow-up command...")
            current_frames = listen_and_record(
                listen_timeout=FOLLOW_UP_LISTEN_TIMEOUT,
                silence_duration_end=FOLLOW_UP_SILENCE_DURATION,
                min_record_sec=0.5
            )

            if current_frames is None:
                print("⏰ No follow-up detected within timeout.")

    finally:
        # --- End of Conversation ---
        print(f"🏁 Conversation ended after {turn_count} turns.")
        is_recording = False # Allow wake word process to restart listening
        print("👂 Re-enabling wake word detection...")


async def main():
    global beep_sound, microwave_beep_sound, microwave_ambient_sound, p, stream, ap, brain
    beep_sound, microwave_beep_sound, microwave_ambient_sound = init_sound()
    
    # Initialize ActionProcessor
    ap = ActionProcessor()
    
    # Pass the initialized processor to the brain
    brain = FishBrain(action_processor=ap)

    # Initialize PyAudio instance here, but don't open stream yet
    p = pyaudio.PyAudio()

    wakeword_thread = Thread(target=wakeword_process, args=(brain,))
    wakeword_thread.daemon = True
    wakeword_thread.start()
    
    print("Fish Voice Assistant is ready! Make a loud noise to activate.")
    play_sound(microwave_beep_sound)
    
    # Initialize MCP client in the background after a short delay
    async def delayed_mcp_init():
        await asyncio.sleep(2)  # Give the system time to fully start
        await initialize_mcp_background()
    
    asyncio.create_task(delayed_mcp_init())
    
    try:
        # Keep the main async loop running
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_ambient_sound()
        close_wakeword_stream() # Close the stream
        if p is not None:
            p.terminate() # Terminate the main PyAudio instance
            print("PyAudio terminated.")
        await ap.cleanup()
        brain.cleanup()
        print("Cleanup complete.")

async def initialize_mcp_background():
    """Initialize MCP client in the background"""
    global ap, brain
    if ap:
        try:
            print("🔌 Attempting to connect to Home Assistant in background...")
            await ap.initialize_mcp_client()
            if ap.mcp_wrapper:
                print("✅ Home Assistant connection successful!")
                # Update the brain with the new tools
                if brain:
                    brain.update_tools()
            else:
                print("⚠️  Home Assistant connection failed, continuing without it")
        except Exception as e:
            print(f"⚠️  Home Assistant background connection error: {e}")
            print("🐠 Fish will continue to work without Home Assistant features")


if __name__ == "__main__":
    asyncio.run(main())
