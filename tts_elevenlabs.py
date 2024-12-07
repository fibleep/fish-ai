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

class ElevenLabsTTSService:
    def __init__(self, cache_dir="tts_cache"):
        self.console = Console()
        self.console.print("[cyan]Initializing ElevenLabs TTS Service...[/cyan]")
        
        # Initialize cache directory
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
        
        self.console.print("[green]Initialization complete![/green]")

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

    def process_audio(self, audio_data):
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        samples = np.array(audio_segment.get_array_of_samples())
        audio = samples.astype(np.float32)
        
        # Normalize audio
        audio = audio / (np.abs(audio).max() + 1e-6) * 0.9
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to match device sample rate
        if audio_segment.frame_rate != self.device_sample_rate:
            self.console.print(f"[cyan]Resampling from {audio_segment.frame_rate} Hz to {self.device_sample_rate} Hz...[/cyan]")
            audio = librosa.resample(
                audio,
                orig_sr=audio_segment.frame_rate,
                target_sr=self.device_sample_rate
            )
        
        return audio.astype(np.float32)

    def process_and_speak(self, text: str):
        try:
            # Extract tools and clean text
            cleaned_text, tool_positions = self.tooling.extract_tools(text)
            
            # Create list of actions with their timing positions
            actions = [(pos, tool) for tool, pos in tool_positions.items()]
            actions.sort()
            
            self.console.print(f"\nFound {len(actions)} actions in text")
            for pos, tool in actions:
                self.console.print(f"Position {pos}: {tool}")
            
            # Generate or retrieve cached audio
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
            
            audio_data = self._get_cached_audio(cache_key)
            
            if audio_data is None:
                self.console.print("\n[cyan]Generating speech with ElevenLabs...[/cyan]")
                audio_data = b''
                response = self.client.text_to_speech.convert(
                    voice_id="pNInz6obpgDQGcFmaJgB",
                    output_format="mp3_22050_32",
                    text=cleaned_text,
                    model_id="eleven_turbo_v2_5",
                    voice_settings=VoiceSettings(
                        stability=0.0,
                        similarity_boost=1.0,
                        style=0.0,
                        use_speaker_boost=True,
                    )
                )
                
                for chunk in response:
                    audio_data += chunk
                
                # Save to cache
                self._save_to_cache(cache_key, audio_data)
            
            # Process audio data
            audio = self.process_audio(audio_data)
            
            # Calculate timing information
            total_duration = len(audio) / self.device_sample_rate
            char_to_time_ratio = total_duration / len(cleaned_text)
            
            # Calculate action timings
            action_timings = []
            for pos, tool in actions:
                action_time = min(pos * char_to_time_ratio, total_duration - 0.1)
                action_timings.append((action_time, tool))
                self.console.print(f"Scheduled {tool} at {action_time:.2f} seconds")
            
            self.console.print(f"\n[cyan]Audio info:[/cyan]")
            self.console.print(f"Duration: {total_duration:.2f} seconds")
            self.console.print(f"Sample rate: {self.device_sample_rate} Hz")
            self.console.print(f"Shape: {audio.shape}")
            
            # Start playback
            self.console.print("\n[green]Starting playback...[/green]")
            
            # Start playing audio in a non-blocking way
            start_time = time.time()
            sd.play(audio, self.device_sample_rate)
            
            # Monitor and execute actions while audio is playing
            action_idx = 0
            while sd.get_stream().active or action_idx < len(action_timings):
                current_time = time.time() - start_time
                
                # Execute any actions that should happen now
                while action_idx < len(action_timings):
                    action_time, tool = action_timings[action_idx]
                    if current_time >= action_time:
                        self.console.print(f"\nExecuting {tool} at {current_time:.2f}s")
                        self.tooling.run_tool(tool)
                        action_idx += 1
                    else:
                        break
                        
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
            sd.wait()  # Wait for playback to finish
            
            # Cleanup
            self.tooling.reset_state()
            self.console.print("\n[green]Playback complete![/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error in process_and_speak: {e}[/red]")
            self.tooling.reset_state()
            raise

    def cleanup(self):
        try:
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
        Hello! <<MouthOpen>>This is a test of our integrated system.<<MouthClose>> 
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