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
ap = ActionProcessor()
ambient_process = None
p = None
stream = None

# Get speaker device from environment variable, fallback to default
SPEAKER_DEVICE = os.getenv('SPEAKER_DEVICE', 'default')

def detect_wake_word(audio_data, threshold=0.6):
    audio_int16 = np.frombuffer(audio_data, np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    energy = np.mean(np.abs(audio_float32))
    if energy > threshold:
        print(f"Wake word energy detected: {energy:.4f}")
        return True
    return False

def init_sound():
    beep = "beep.wav"
    microwave_beep = "microwave_beep.wav"
    microwave_ambient = "microwave_ambient.wav"
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
    global p, stream
    if stream is not None and stream.is_active():
        print("Wake word stream already open and active.")
        return
    if p is None:
        p = pyaudio.PyAudio()
    print("Opening wake word stream...")
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    print("Wake word stream opened.")

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
                     close_wakeword_stream() # Close before recording starts
                     record_command(brain)
                     # Stream will be reopened automatically at the start of the next loop iteration if needed
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
    rec_p = pyaudio.PyAudio()
    rec_stream = None

    try:
        rec_stream = rec_p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"Failed to open recording stream: {e}")
        rec_p.terminate()
        return None # Indicate failure to open stream

    print(f"Listening... (timeout in {listen_timeout:.1f}s)")
    initial_sound_detected = False
    recording_started = False
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
                initial_sound_detected = True
                recording_started = True
                print("Initial speech detected.")
                break # Sound detected, move to recording phase
            time.sleep(0.01)
        except IOError as e:
            print(f"Error reading during initial listen: {e}")
            time.sleep(0.1) # Avoid busy-looping on persistent errors


    if not initial_sound_detected:
        print("No sound detected within timeout.")
        rec_stream.stop_stream()
        rec_stream.close()
        rec_p.terminate()
        return None # Indicate timeout

    # 2. Record until silence or max duration
    print("Recording...")
    start_time = time.time()
    silence_start = None
    max_record_duration = MAX_RECORD_SECONDS # Use the global max for safety

    # Trim leading silence buffered before speech detection
    # Heuristic: keep last 0.5 seconds before detected sound
    keep_chunks = int(0.5 * SAMPLE_RATE / CHUNK_SIZE)
    frames = frames[-keep_chunks:]

    while time.time() - start_time < max_record_duration:
        try:
            data = rec_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            energy = np.mean(np.abs(audio_float32))

            # Silence detection logic (only after minimum recording time)
            if time.time() - start_time > min_record_sec:
                if energy < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration_end:
                        print(f"Silence detected for {silence_duration_end:.1f}s, stopping recording.")
                        break
                else:
                    silence_start = None # Reset silence timer
            time.sleep(0.01)
        except IOError as e:
             print(f"Error reading during recording: {e}")
             time.sleep(0.1)


    print("Recording stopped.")
    rec_stream.stop_stream()
    rec_stream.close()
    rec_p.terminate()

    if not frames:
        return None

    return frames

def record_command(brain):
    global is_recording
    is_recording = True # Signal wake word loop to stop listening

    # Ensure wake word stream is closed before playing sound
    print("Wake word detected, preparing to record...")
    time.sleep(0.2) # Give wakeword_process time to close stream

    # Move head outwards to indicate listening
    print("Moving head outwards...")
    ap.tooling.run_tool("MoveHead&&Outward")

    play_sound(beep_sound)
    # No sleep needed here, listen_and_record handles opening its own stream

    # --- Initial Command Recording ---
    print("Recording your initial command...")
    # Use INITIAL_LISTEN_TIMEOUT for the first command listen timeout
    current_frames = listen_and_record(
        listen_timeout=INITIAL_LISTEN_TIMEOUT, # Increased leniency
        silence_duration_end=SILENCE_DURATION,
        min_record_sec=1.5 # Existing minimum for initial command
    )

    if current_frames is None:
        print("Initial command recording failed or timed out.")
        is_recording = False # Allow wake word process to restart
        return # Exit if initial recording failed

    # --- Conversation Loop ---
    while current_frames is not None:
        print("Processing command...")
        play_ambient_sound(microwave_ambient_sound)
        audio_file = save_audio(current_frames, filename="command.wav") # Overwrite previous
        response = brain.process_audio_input(audio_file)
        cleaned_text, tool_positions, audio = ap.preprocess(response)
        stop_ambient_sound()
        play_sound(microwave_beep_sound) # Beep before speaking response
        time.sleep(0.2)
        ap.process_and_speak(cleaned_text, tool_positions, audio)
        print(f"Fish: {response}") # Print after speaking

        # --- Listen for Follow-up ---
        print("Listening for follow-up command...")
        current_frames = listen_and_record(
            listen_timeout=FOLLOW_UP_LISTEN_TIMEOUT,
            silence_duration_end=FOLLOW_UP_SILENCE_DURATION,
            min_record_sec=0.5 # Shorter min recording for follow-ups
        )

        if current_frames is None:
            print("No follow-up detected within timeout.")
            # Loop condition (current_frames is None) will cause exit

    # --- End of Conversation ---
    print("Conversation turn ended.")
    is_recording = False # Allow wake word process to restart listening

def main():
    global beep_sound, microwave_beep_sound, microwave_ambient_sound, p, stream
    beep_sound, microwave_beep_sound, microwave_ambient_sound = init_sound()
    brain = FishBrain()

    # Initialize PyAudio instance here, but don't open stream yet
    p = pyaudio.PyAudio()

    wakeword_thread = Thread(target=wakeword_process, args=(brain,))
    wakeword_thread.daemon = True
    wakeword_thread.start()
    print("Fish Voice Assistant is ready! Make a loud noise to activate.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_ambient_sound()
        close_wakeword_stream() # Close the stream
        if p is not None:
            p.terminate() # Terminate the main PyAudio instance
            print("PyAudio terminated.")
        brain.cleanup()
        ap.cleanup()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
