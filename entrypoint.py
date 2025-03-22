import torch
import pyaudio
import numpy as np
import time
import wave
import os
import pygame
import struct
from threading import Thread
from queue import Queue
from pvporcupine import create as create_porcupine
from pvrecorder import PvRecorder
from utils.llm import call, transcribe_audio
from fish_brain import FishBrain
from action_processor import ActionProcessor

SILERO_THRESHOLD = 0.5
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_RECORD_SECONDS = 5
SILENCE_THRESHOLD = 0.2
SILENCE_DURATION = 3

audio_queue = Queue()
is_recording = False

def init_sound():
    pygame.mixer.init()
    ding_sound_path = "ding.wav"
    
    if not os.path.exists(ding_sound_path):
        create_ding_sound(ding_sound_path)
    
    return ding_sound_path

def create_ding_sound(filename, duration=0.5):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * 880 * t)
    
    fade_samples = int(sample_rate * 0.05)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    tone[:fade_samples] *= fade_in
    tone[-fade_samples:] *= fade_out
    
    audio = (tone * 32767).astype(np.int16)
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(audio.tobytes())
    wf.close()

def play_ding(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def load_silero_model():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    return model

def save_audio(frames, filename="command.wav"):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def vad_process():
    silero_model = load_silero_model()
    print("Silero VAD model loaded. Listening for speech...")
    
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            
            audio_int16 = np.frombuffer(audio_data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            tensor = torch.from_numpy(audio_float32)
            
            with torch.no_grad():
                speech_prob = silero_model(tensor, SAMPLE_RATE).item()
            
            if speech_prob > SILERO_THRESHOLD:
                print(f"Speech detected! Probability: {speech_prob:.4f}")

def porcupine_process(brain):
    access_key = os.getenv("PORCUPINE_ACCESS_KEY")
    
    keywords = ["data/fish.ppn"]
    
    porcupine = create_porcupine(
        access_key=access_key,
        keyword_paths=keywords,
    )
    
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    recorder.start()
    
    print("Porcupine initialized. Listening for keyword...")
    
    while True:
        pcm = recorder.read()
        
        result = porcupine.process(pcm)
        
        if result >= 0 and not is_recording:
            print(f"Keyword detected: {keywords[result]}")
            play_ding(ding_sound_path)
            
            record_command(brain)
    
    recorder.delete()
    porcupine.delete()

def record_command(brain):
    global is_recording
    
    is_recording = True
    command_frames = []
    
    silero_model = load_silero_model()
    
    print("Recording your command... (speak now)")
    
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
    
    while time.time() - start_time < MAX_RECORD_SECONDS:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        command_frames.append(data)
        
        audio_int16 = np.frombuffer(data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)
        
        with torch.no_grad():
            speech_prob = silero_model(tensor, SAMPLE_RATE).item()
        
        if speech_prob < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_DURATION:
                print(f"Silence detected for {SILENCE_DURATION} seconds, stopping recording.")
                break
        else:
            silence_start = None
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    is_recording = False
    print("Recording stopped.")
    
    if command_frames:
        audio_file = save_audio(command_frames)
        
        print("Processing your command...")
        
        response = brain.process_audio_input(audio_file)
        ap = ActionProcessor()
        ap.process_action(response)
        
        print(f"Fish: {response}")

def main():
    global ding_sound_path
    
    ding_sound_path = init_sound()
    
    brain = FishBrain()
    
    porcupine_thread = Thread(target=porcupine_process, args=(brain,))
    porcupine_thread.daemon = True
    porcupine_thread.start()
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=audio_callback
    )
    
    stream.start_stream()
    
    vad_thread = Thread(target=vad_process)
    vad_thread.daemon = True
    vad_thread.start()
    
    print("Fish Voice Assistant is ready! Say the wake word to start.")
    
    while True:
        time.sleep(0.1)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    brain.cleanup()
    print("Stopped recording")

if __name__ == "__main__":
    main()
