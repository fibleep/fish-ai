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
from typing import Optional
from collections import deque
from dotenv import load_dotenv

load_dotenv()

class ElevenLabsTTSService:
    def __init__(self, cache_dir="tts_cache"):
        self.console = Console()
        self.console.print("[cyan]Initializing ElevenLabs TTS Service...[/cyan]")
        
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.console.print(f"[cyan]Using cache directory: {self.cache_dir}[/cyan]")
        
        warnings.filterwarnings('ignore', category=UserWarning)
        self.find_and_set_devices()
        
        try:
            self.console.print("\n[cyan]Initializing Arduino tooling...[/cyan]")
            self.tooling = Tooling()
        except Exception as e:
            self.console.print(f"[red]Error initializing tooling: {e}[/red]")
            raise
        
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY")
        )
        
        # Audio analysis parameters
        self.chunk_size = int(self.device_sample_rate * 0.02)  # 20ms chunks
        self.energy_threshold = 0.1  # Adjustable threshold for mouth movement
        self.min_mouth_duration = 0.02  # Minimum time for mouth to stay in one position
        
        self.console.print("[green]Initialization complete![/green]")

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
            
        device_info = sd.query_devices(self.output_device)
        self.device_sample_rate = int(device_info['default_samplerate'])
        
        sd.default.device = (None, self.output_device)
        sd.default.samplerate = self.device_sample_rate
        sd.default.channels = (None, 1)
        sd.default.dtype = ('float32', 'float32')
        
        self.console.print(f"[green]Using output device: {device_info['name']}[/green]")
        self.console.print(f"[green]Sample rate: {self.device_sample_rate} Hz[/green]")

    def _generate_cache_key(self, text: str, voice_id: str, model_id: str, voice_settings: dict) -> str:
        """Generate a unique cache key based on input parameters"""
        params = {
            'text': text,
            'voice_id': voice_id,
            'model_id': model_id,
            'voice_settings': voice_settings
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Try to get audio data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        if cache_file.exists():
            self.console.print("[green]Found cached audio![/green]")
            return cache_file.read_bytes()
        return None

    def _save_to_cache(self, cache_key: str, audio_data: bytes):
        """Save audio data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        cache_file.write_bytes(audio_data)
        self.console.print("[green]Saved audio to cache[/green]")

    def analyze_audio_energy(self, audio: np.ndarray) -> list:
        """Analyze audio energy to determine mouth movements"""
        chunks = np.array_split(audio, len(audio) // self.chunk_size)
        mouth_states = []
        
        # Use rolling average for smoother transitions
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
        """Process audio data for playback and analysis"""
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        samples = np.array(audio_segment.get_array_of_samples())
        audio = samples.astype(np.float32)
        
        # Normalize audio
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
        
        return audio.astype(np.float32)

    def _generate_audio(self, text: str, voice_settings: dict) -> bytes:
        """Generate audio from ElevenLabs"""
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
        
        self._save_to_cache(self._generate_cache_key(
            text, "pNInz6obpgDQGcFmaJgB", "eleven_turbo_v2_5", voice_settings
        ), audio_data)
        
        return audio_data

    def process_and_speak(self, text: str):
        """Main method to process text and generate speech with actions"""
        try:
            # Extract tools and clean text
            cleaned_text, tool_positions = self.tooling.extract_tools(text)
            
            # Voice settings
            voice_settings_dict = {
                'stability': 0.0,
                'similarity_boost': 1.0,
                'style': 0.0,
                'use_speaker_boost': True
            }
            
            # Get or generate audio
            cache_key = self._generate_cache_key(
                cleaned_text,
                "pNInz6obpgDQGcFmaJgB",
                "eleven_turbo_v2_5",
                voice_settings_dict
            )
            
            audio_data = self._get_cached_audio(cache_key) or self._generate_audio(
                cleaned_text, voice_settings_dict)
            
            # Process audio and analyze for mouth movements
            audio = self.process_audio(audio_data)
            mouth_states = self.analyze_audio_energy(audio)
            
            # Calculate timings based on text position
            total_duration = len(audio) / self.device_sample_rate
            char_to_time_ratio = total_duration / len(cleaned_text)
            
            # Queue actions (excluding mouth movements) with relative timings
            actions = deque([
                (pos * char_to_time_ratio, tool)
                for tool, pos in tool_positions.items()
                if not tool.startswith("Mouth")
            ])
            
            # Sort actions by timing
            actions = deque(sorted(actions, key=lambda x: x[0]))
            
            self.console.print("\n[green]Starting playback...[/green]")
            
            # Start playback and get the exact start time
            playback_start = time.time()
            sd.play(audio, self.device_sample_rate)
            
            # Playback loop
            chunk_duration = self.chunk_size / self.device_sample_rate
            last_mouth_state = False
            chunk_index = 0
            last_mouth_change = time.time()
            
            while sd.get_stream().active or actions:
                current_time = time.time() - playback_start  # Relative time from start
                
                # Handle mouth movement
                if chunk_index < len(mouth_states):
                    should_open = mouth_states[chunk_index]
                    if should_open != last_mouth_state and (time.time() - last_mouth_change) >= self.min_mouth_duration:
                        if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
                            self.tooling.move_mouth(should_open)
                            last_mouth_state = should_open
                            last_mouth_change = time.time()
                    chunk_index = int(current_time / chunk_duration)
                
                # Execute queued actions
                while actions and current_time >= actions[0][0]:
                    action_time, tool = actions.popleft()
                    self.console.print(f"\nExecuting {tool} at {current_time:.2f}s")
                    if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
                        self.tooling.run_tool(tool)
                    else:
                        self.console.print("[red]Serial connection lost, attempting to reconnect...[/red]")
                        self.tooling._init_connection()
                
                time.sleep(0.005)
            
            sd.wait()
            if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
                self.tooling.reset_state()
            self.console.print("\n[green]Playback complete![/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error in process_and_speak: {e}[/red]")
            if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
                self.tooling.reset_state()
            raise

    def cleanup(self):
        """Clean up resources"""
        try:
            # Only cleanup if we haven't already
            if hasattr(self, 'tooling') and self.tooling.serial_conn and self.tooling.serial_conn.is_open:
                self.tooling.cleanup()
        except Exception as e:
            self.console.print(f"[red]Error during cleanup: {e}[/red]")

    def clear_cache(self):
        """Clear all cached audio files"""
        try:
            for cache_file in self.cache_dir.glob("*.mp3"):
                cache_file.unlink()
            self.console.print("[green]Cache cleared successfully![/green]")
        except Exception as e:
            self.console.print(f"[red]Error clearing cache: {e}[/red]")

if __name__ == "__main__":
    try:
        tts = ElevenLabsTTSService()
        
        test_text = """
        Hello! This is a test of our integrated system.
        Watch as I <<MoveHead&&Outward>>move my head outward<<MoveHead&&Inward>>. 
        Now, <<TailFlop>>watch my tail flop!
        """
        
        tts.process_and_speak(test_text)
        
    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        raise
    finally:
        if 'tts' in locals():
            tts.cleanup()