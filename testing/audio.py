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

class ElevenLabsTTSService:
    def __init__(self):
        self.console = Console()
        self.console.print("[cyan]Initializing ElevenLabs TTS Service...[/cyan]")
        
        # Filter warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Set sample rate for output device
        self.device_sample_rate = 48000
        
        # Configure audio devices
        self.find_and_set_devices()
        
        # Initialize Tooling
        try:
            self.console.print("\n[cyan]Initializing Arduino tooling...[/cyan]")
            self.tooling = Tooling()
        except Exception as e:
            self.console.print(f"[red]Error initializing tooling: {e}[/red]")
            raise
        
        # Initialize ElevenLabs client
        self.client = ElevenLabs(
            api_key="sk_9fc2b48d11d6e4af926264d2b604f57fdf7e00a5cad16f61"
        )
        
        self.console.print("[green]Initialization complete![/green]")

    def find_and_set_devices(self):
        """Automatically find and configure the correct audio devices"""
        devices = sd.query_devices()
        self.console.print("\n[yellow]Available Audio Devices:[/yellow]")
        
        # Print all devices
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append('INPUT')
            if device['max_output_channels'] > 0:
                device_type.append('OUTPUT')
            
            self.console.print(f"[cyan]{i}:[/cyan] {device['name']} "
                             f"({', '.join(device_type)}) "
                             f"- Default SR: {device['default_samplerate']}")

        # Find UACDemo device automatically
        self.output_device = None
        for i, device in enumerate(devices):
            if "UACDemoV1.0" in device['name'] and device['max_output_channels'] > 0:
                self.output_device = i
                self.console.print(f"\n[green]Found UACDemo device at index {i}[/green]")
                break

        if self.output_device is None:
            self.console.print("[yellow]UACDemo device not found, using default output[/yellow]")
            self.output_device = sd.default.device[1]
            
        # Configure audio settings
        device_info = sd.query_devices(self.output_device)
        sd.default.device = (None, self.output_device)  # Only set output device
        sd.default.samplerate = self.device_sample_rate
        sd.default.channels = (1, 2)  # Mono input, stereo output
        sd.default.dtype = ('float32', 'float32')
        
        self.console.print(f"[green]Using output device: {device_info['name']}[/green]")
        self.console.print(f"[green]Sample rate: {self.device_sample_rate} Hz[/green]")

    def process_and_speak(self, text: str):
        """Process text with tools and synthesize speech using ElevenLabs"""
        try:
            # Extract tools and clean text
            cleaned_text, tool_positions = self.tooling.extract_tools(text)
            
            # Create list of actions with their timing positions
            actions = [(pos, tool) for tool, pos in tool_positions.items()]
            actions.sort()
            
            self.console.print(f"\nFound {len(actions)} actions in text")
            for pos, tool in actions:
                self.console.print(f"Position {pos}: {tool}")
            
            # Generate audio using ElevenLabs
            self.console.print("\n[cyan]Generating speech with ElevenLabs...[/cyan]")
            
            response = self.client.text_to_speech.convert(
                voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
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
            
            # Convert response chunks to audio data
            audio_data = b''
            for chunk in response:
                audio_data += chunk
            
            # Convert MP3 to wav array
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize audio
            audio = audio / (np.abs(audio).max() + 1e-6)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Calculate timing information
            total_duration = len(audio) / audio_segment.frame_rate
            char_to_time_ratio = total_duration / len(cleaned_text)
            
            # Calculate action timings
            action_timings = []
            for pos, tool in actions:
                action_time = min(pos * char_to_time_ratio, total_duration - 0.5)
                action_timings.append((action_time, tool))
                self.console.print(f"Scheduled {tool} at {action_time:.2f} seconds")
            
            self.console.print(f"\n[cyan]Audio info:[/cyan]")
            self.console.print(f"Duration: {total_duration:.2f} seconds")
            self.console.print(f"Sample rate: {audio_segment.frame_rate} Hz")
            self.console.print(f"Shape: {audio.shape}")
            
            # Start playback
            self.console.print("\n[green]Starting playback...[/green]")
            sd.play(audio, audio_segment.frame_rate, device=self.output_device)
            start_time = time.time()
            
            action_idx = 0
            
            # Monitor playback and execute actions
            while sd.get_stream().active or action_idx < len(action_timings):
                current_time = time.time() - start_time
                
                # Handle scheduled actions
                while action_idx < len(action_timings):
                    action_time, tool = action_timings[action_idx]
                    
                    if current_time >= action_time:
                        self.console.print(f"\nExecuting {tool} at {current_time:.2f}s")
                        self.tooling.run_tool(tool)
                        action_idx += 1
                        continue
                    break
                
                time.sleep(0.01)
            
            # Ensure cleanup
            self.tooling.reset_state()
            self.console.print("\n[green]Playback complete![/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error in process_and_speak: {e}[/red]")
            self.tooling.reset_state()
            raise

    def cleanup(self):
        """Clean up resources"""
        try:
            self.tooling.cleanup()
        except Exception as e:
            self.console.print(f"[red]Error during cleanup: {e}[/red]")

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