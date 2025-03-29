import os
import time
import numpy as np
import sounddevice as sd
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from tooling import Tooling
from rich.console import Console
import io
from pydub import AudioSegment
import warnings
import librosa
import hashlib
import json
import pathlib
from typing import Optional, Dict, List, Tuple
from collections import deque
from dotenv import load_dotenv

load_dotenv()

class ActionProcessor:
    def __init__(self, cache_dir="tts_cache"):
        self.console = Console()
        self.console.print("[cyan]Initializing AudioProcessor...[/cyan]")
        
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        warnings.filterwarnings('ignore', category=UserWarning)
        
        self.find_and_set_devices()
        
        self.console.print("\n[cyan]Initializing Arduino tooling...[/cyan]")
        self.tooling = Tooling()
        
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY")
        )
        
        self.chunk_size = int(self.device_sample_rate * 0.02)
        self.energy_threshold = 0.1
        self.min_mouth_duration = 0.02
        
        # Track last tool executed for persistence
        self.last_tool = None
        self.conversation_active = True
        
        self.console.print("[green]AudioProcessor initialization complete![/green]")
        
        self.current_ambient_playback = None

    def find_and_set_devices(self):
        devices = sd.query_devices()
        self.console.print("\n[yellow]Available Audio Devices:[/yellow]")
        
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append('INPUT')
            if device['max_output_channels'] > 0:
                device_type.append('OUTPUT')
            
            self.console.print(f"[cyan]{i}:[/cyan] {device['name']} "
                            f"({', '.join(device_type)}) "
                            f"- Default SR: {device['default_samplerate']}")

        self.output_device = None
        for i, device in enumerate(devices):
            if "UACDemoV1.0" in device['name'] and device['max_output_channels'] > 0:
                self.output_device = i
                self.console.print(f"\n[green]Found UACDemo device at index {i}[/green]")
                break

        if self.output_device is None:
            self.console.print("[yellow]UACDemo device not found, using default output[/yellow]")
            self.output_device = sd.default.device[1]
        else:
            os.environ['SDL_AUDIODRIVER'] = 'alsa'
            os.environ['SDL_AUDIODEV'] = 'sysdefault:CARD=UACDemoV10'
            
        device_info = sd.query_devices(self.output_device)
        self.device_sample_rate = int(device_info['default_samplerate'])
        
        sd.default.device = (None, self.output_device)
        sd.default.samplerate = self.device_sample_rate
        sd.default.channels = (None, 1)
        sd.default.dtype = ('float32', 'float32')
        
        self.console.print(f"[green]Using output device: {device_info['name']}[/green]")
        self.console.print(f"[green]Sample rate: {self.device_sample_rate} Hz[/green]")

    def _generate_cache_key(self, text: str, voice_id: str, model_id: str, voice_settings: dict) -> str:
        params = {
            'text': text,
            'voice_id': voice_id,
            'model_id': model_id,
            'voice_settings': voice_settings
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        if cache_file.exists():
            self.console.print("[green]Found cached audio![/green]")
            return cache_file.read_bytes()
        return None

    def _save_to_cache(self, cache_key: str, audio_data: bytes):
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        cache_file.write_bytes(audio_data)
        self.console.print("[green]Saved audio to cache[/green]")

    def analyze_audio_energy(self, audio: np.ndarray) -> list:
        chunks = np.array_split(audio, len(audio) // self.chunk_size)
        mouth_states = []
        
        window_size = 3
        rolling_energy = []
        
        for chunk in chunks:
            energy = np.sqrt(np.mean(chunk**2))
            rolling_energy.append(energy)
            
            if len(rolling_energy) > window_size:
                rolling_energy.pop(0)
            
            avg_energy = np.mean(rolling_energy)
            mouth_states.append(avg_energy > self.energy_threshold)
        
        return mouth_states

    def process_audio(self, audio_data):
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        samples = np.array(audio_segment.get_array_of_samples())
        audio = samples.astype(np.float32)
        
        audio = audio / (np.abs(audio).max() + 1e-6) * 0.9
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if audio_segment.frame_rate != self.device_sample_rate:
            self.console.print(f"[cyan]Resampling from {audio_segment.frame_rate} Hz to {self.device_sample_rate} Hz...[/cyan]")
            audio = librosa.resample(
                audio,
                orig_sr=audio_segment.frame_rate,
                target_sr=self.device_sample_rate
            )
        
        # Add half a second buffer of silence at the end
        buffer_samples = int(self.device_sample_rate * 0.5)
        silent_buffer = np.zeros(buffer_samples, dtype=np.float32)
        audio_with_buffer = np.concatenate([audio, silent_buffer])
        
        return audio_with_buffer.astype(np.float32)

    def _generate_audio(self, text: str, voice_settings: dict) -> bytes:
        self.console.print("\n[cyan]Generating speech with ElevenLabs...[/cyan]")
        audio_data = b''
        response = self.client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(**voice_settings)
        )
        
        for chunk in response:
            audio_data += chunk
        
        cache_key = self._generate_cache_key(
            text, "pNInz6obpgDQGcFmaJgB", "eleven_turbo_v2_5", voice_settings
        )
        self._save_to_cache(cache_key, audio_data)
        
        return audio_data

    def play_sound(self, sound_file):
        try:
            with open(sound_file, 'rb') as f:
                audio_data = f.read()
            
            audio = self.process_audio(audio_data)
            sd.play(audio, self.device_sample_rate)
            sd.wait()
            print(f"Played sound: {sound_file}")
        except Exception as e:
            print(f"Error playing sound: {e}")

    def play_ambient_sound(self, sound_file):
        try:
            self.stop_ambient_sound()
            
            with open(sound_file, 'rb') as f:
                audio_data = f.read()
            
            audio = self.process_audio(audio_data)
            
            def _callback(outdata, frames, time, status):
                if len(audio) > 0:
                    if len(audio) >= frames:
                        outdata[:] = audio[:frames].reshape(-1, 1)
                        audio = np.roll(audio, -frames)
                    else:
                        outdata[:len(audio)] = audio.reshape(-1, 1)
                        outdata[len(audio):] = 0
            
            self.current_ambient_playback = sd.OutputStream(
                samplerate=self.device_sample_rate,
                device=self.output_device,
                channels=1,
                callback=_callback,
                blocksize=self.chunk_size
            )
            self.current_ambient_playback.start()
            print(f"Started ambient sound: {sound_file}")
            
        except Exception as e:
            print(f"Error playing ambient sound: {e}")

    def stop_ambient_sound(self):
        if self.current_ambient_playback is not None:
            self.current_ambient_playback.stop()
            self.current_ambient_playback.close()
            self.current_ambient_playback = None
            print("Stopped ambient sound")

    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_active = False
        
        if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
            self.tooling.move_mouth(False)
            self.tooling.run_tool("MoveHead&&Inward")
            self.last_tool = None
        
        self.conversation_active = True
        
    def preprocess(self, text: str):
        cleaned_text, tool_positions = self.tooling.extract_tools(text)
        voice_settings_dict = {
            'stability': 0.0,
            'similarity_boost': 1.0,
            'style': 0.0,
            'use_speaker_boost': True
        }
        cache_key = self._generate_cache_key(
            cleaned_text,
            "pNInz6obpgDQGcFmaJgB",
            "eleven_turbo_v2_5",
            voice_settings_dict
        )
        
        audio_data = self._get_cached_audio(cache_key) or self._generate_audio(
            cleaned_text, voice_settings_dict
        )
        
        audio = self.process_audio(audio_data)
        return cleaned_text, tool_positions, audio
        

    def process_and_speak(self, cleaned_text, tool_positions, audio):
        mouth_states = self.analyze_audio_energy(audio)
        
        total_duration = len(audio) / self.device_sample_rate
        char_to_time_ratio = total_duration / max(1, len(cleaned_text))
        
        actions = deque([
            (pos * char_to_time_ratio, tool)
            for tool, pos in tool_positions.items()
            if not tool.startswith("Mouth")
        ])
        
        actions = deque(sorted(actions, key=lambda x: x[0]))
        
        self.console.print("\n[green]Starting playback...[/green]")
        
        playback_start = time.time()
        sd.play(audio, self.device_sample_rate)
        
        chunk_duration = self.chunk_size / self.device_sample_rate
        last_mouth_state = False
        chunk_index = 0
        last_mouth_change = time.time()
        last_action = None
        last_action_time = None
        
        while sd.get_stream().active or actions:
            current_time = time.time() - playback_start
            
            if chunk_index < len(mouth_states):
                should_open = mouth_states[chunk_index]
                if should_open != last_mouth_state and (time.time() - last_mouth_change) >= self.min_mouth_duration:
                    if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
                        self.tooling.move_mouth(should_open)
                        last_mouth_state = should_open
                        last_mouth_change = time.time()
                chunk_index = int(current_time / chunk_duration)
            
            while actions and current_time >= actions[0][0]:
                action_time, tool = actions.popleft()
                self.console.print(f"\nExecuting {tool} at {current_time:.2f}s")
                if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
                    self.tooling.run_tool(tool)
                    last_action = tool
                    last_action_time = action_time
                    self.last_tool = tool  # Store the last tool for persistence
                else:
                    self.console.print("[red]Serial connection lost, attempting to reconnect...[/red]")
                    self.tooling._init_connection()
            
            time.sleep(0.005)
        
        sd.wait()
        
        if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
            self.tooling.move_mouth(False)
            
            # Check if the last action is near the end of the audio
            # If it's in the last 20% of the audio, consider it an "end state" to persist
            if last_action and last_action_time:
                time_from_end = total_duration - last_action_time
                if time_from_end < 0.2 * total_duration:
                    self.console.print(f"[yellow]Maintaining end state: {last_action}[/yellow]")
                else:
                    self.console.print(f"[cyan]Last action not at end, resetting state[/cyan]")
                    self.tooling.reset_state()
                    self.last_tool = None
            else:
                self.tooling.reset_state()
                self.last_tool = None
            
        self.console.print("\n[green]Playback complete![/green]")
        
        return audio

    def cleanup(self):
        self.stop_ambient_sound()
        self.conversation_active = False
        if hasattr(self, 'tooling') and self.tooling.serial_conn and self.tooling.serial_conn.is_open:
            self.tooling.cleanup()

    def clear_cache(self):
        for cache_file in self.cache_dir.glob("*.mp3"):
            cache_file.unlink()
        self.console.print("[green]Cache cleared successfully![/green]")

