import pyaudio
import numpy as np
import time
import wave
import os
import subprocess
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

audio_queue = Queue()
is_recording = False
ap = ActionProcessor()
ambient_process = None

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
        subprocess.run(["aplay", "-q", "-D", "sysdefault:CARD=UACDemoV10", sound_file], check=True)
        print(f"Played sound: {sound_file}")
    except Exception as e:
        print(f"Error playing sound: {e}")
        try:
            subprocess.run(["aplay", "-q", sound_file], check=True)
            print(f"Played sound on default device: {sound_file}")
        except Exception as e:
            print(f"Error playing sound on default device: {e}")

def play_ambient_sound(sound_file):
    global ambient_process
    try:
        stop_ambient_sound()
        print(f"Starting ambient sound: {sound_file}")
        ambient_process = subprocess.Popen(
            ["aplay", "-q", "-D", "sysdefault:CARD=UACDemoV10", sound_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Error playing ambient sound: {e}")
        try:
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

def wakeword_process(brain):
    print("Using simple energy-based wake word detection...")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    threshold = 0.3
    
    while True:
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            if detect_wake_word(data, threshold) and not is_recording:
                print("Wake word detected with energy detector")
                record_command(brain)
            
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            time.sleep(1)

def record_command(brain):
    global is_recording
    
    is_recording = True
    command_frames = []
    
    print("Recording your command... (speak now)")
    play_sound(beep_sound)
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    start_time = time.time()
    silence_start = None
    recording_started = False
    min_recording_time = 1.5
    
    while time.time() - start_time < MAX_RECORD_SECONDS:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        command_frames.append(data)
        
        audio_int16 = np.frombuffer(data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        energy = np.mean(np.abs(audio_float32))
        
        if not recording_started and energy > SILENCE_THRESHOLD * 1.5:
            recording_started = True
            print("Speech detected")
        
        if recording_started and time.time() - start_time > min_recording_time:
            if energy < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Long silence detected, stopping recording")
                    break
            else:
                silence_start = None
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    is_recording = False
    print("Recording stopped.")
    
    if command_frames:
        # Start ambient sound right after recording stops
        play_ambient_sound(microwave_ambient_sound)
        
        audio_file = save_audio(command_frames)
        
        print("Processing your command...")
        response = brain.process_audio_input(audio_file)
        
        # Stop ambient sound and play beep before speaking
        cleaned_text, tool_positions, audio = ap.preprocess(response)
        stop_ambient_sound()
        play_sound(microwave_beep_sound)
        
        # Give a moment for the beep to finish before speaking
        time.sleep(0.2)
        ap.process_and_speak(cleaned_text, tool_positions, audio)
        
        print(f"Fish: {response}")

def main():
    global beep_sound, microwave_beep_sound, microwave_ambient_sound
    
    beep_sound, microwave_beep_sound, microwave_ambient_sound = init_sound()
    
    brain = FishBrain()
    
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
        brain.cleanup()
        ap.cleanup()
        print("Stopped recording")

if __name__ == "__main__":
    main()