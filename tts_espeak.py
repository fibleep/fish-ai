import os
import time
import numpy as np
import sounddevice as sd
from tooling import Tooling
from rich.console import Console
import subprocess
import tempfile
import wave
import warnings
import librosa

class ESpeakTTSService:
    def __init__(self):
        self.console = Console()
        self.console.print("[cyan]Initializing ESpeak TTS Service...[/cyan]")
        
        # Filter warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Configure audio devices first to get the correct sample rate
        self.find_and_set_devices()
        
        # Initialize Tooling
        try:
            self.console.print("\n[cyan]Initializing Arduino tooling...[/cyan]")
            self.tooling = Tooling()
        except Exception as e:
            self.console.print(f"[red]Error initializing tooling: {e}[/red]")
            raise
        
        # Check if espeak is installed
        try:
            subprocess.run(['espeak', '--version'], capture_output=True)
            self.console.print("[green]ESpeak found on system[/green]")
        except FileNotFoundError:
            self.console.print("[red]Error: espeak not found. Please install it with:\nsudo apt-get install espeak[/red]")
            raise
        
        # Action timing settings
        self.action_delay = 0.5  # Seconds to pause for action execution
        
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
            
        # Get device info and supported sample rate
        device_info = sd.query_devices(self.output_device)
        self.device_sample_rate = int(device_info['default_samplerate'])
        
        # Configure audio settings
        sd.default.device = (None, self.output_device)
        sd.default.samplerate = self.device_sample_rate
        sd.default.channels = (None, 1)  # Mono output
        sd.default.dtype = ('float32', 'float32')
        
        self.console.print(f"[green]Using output device: {device_info['name']}[/green]")
        self.console.print(f"[green]Sample rate: {self.device_sample_rate} Hz[/green]")

    def text_to_wav(self, text: str, output_file: str):
        """Convert text to speech using espeak and save as WAV"""
        cmd = [
            'espeak',
            '-v', 'en-us',  # Voice
            '-s', '150',    # Speed
            '-p', '50',     # Pitch
            '-a', '200',    # Amplitude
            '-w', output_file,
            text
        ]
        subprocess.run(cmd, check=True)

    def _execute_action(self, tool: str, current_time: float):
        """Execute a single action with proper timing"""
        self.console.print(f"\nExecuting {tool} at {current_time:.2f}s")
        
        # Execute the action
        self.tooling.run_tool(tool)
        
        # Allow time for the action to complete
        time.sleep(self.action_delay)

    def process_and_speak(self, text: str):
        """Process text with tools and synthesize speech using espeak"""
        try:
            # Extract tools and clean text
            cleaned_text, tool_positions = self.tooling.extract_tools(text)
            
            # Create list of actions with their timing positions
            actions = [(pos, tool) for tool, pos in tool_positions.items()]
            actions.sort()
            
            self.console.print(f"\nFound {len(actions)} actions in text")
            for pos, tool in actions:
                self.console.print(f"Position {pos}: {tool}")
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Generate speech using espeak
            self.console.print("\n[cyan]Generating speech with espeak...[/cyan]")
            self.text_to_wav(cleaned_text, temp_wav_path)
            
            # Read the WAV file
            with wave.open(temp_wav_path, 'rb') as wav_file:
                wav_sample_rate = wav_file.getframerate()
                wav_data = wav_file.readframes(wav_file.getnframes())
            
            # Convert to numpy array
            audio = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize to [-1, 1]
            
            # Resample audio to match device sample rate
            self.console.print(f"[cyan]Resampling from {wav_sample_rate} Hz to {self.device_sample_rate} Hz...[/cyan]")
            audio = librosa.resample(audio, orig_sr=wav_sample_rate, target_sr=self.device_sample_rate)
            
            # Calculate timing information
            total_duration = len(audio) / self.device_sample_rate
            char_to_time_ratio = total_duration / len(cleaned_text)
            
            # Calculate action timings and add buffer time for each action
            action_timings = []
            current_offset = 0
            for pos, tool in actions:
                # Calculate base timing
                action_time = min(pos * char_to_time_ratio, total_duration - 0.5)
                # Add any accumulated offset from previous actions
                adjusted_time = action_time + current_offset
                action_timings.append((adjusted_time, tool))
                # Add delay for the next action
                current_offset += self.action_delay
                self.console.print(f"Scheduled {tool} at {adjusted_time:.2f}s")
            
            # Extend audio duration to accommodate action delays
            if current_offset > 0:
                silence_samples = int(current_offset * self.device_sample_rate)
                audio = np.pad(audio, (0, silence_samples))
                total_duration = len(audio) / self.device_sample_rate
            
            self.console.print(f"\n[cyan]Audio info:[/cyan]")
            self.console.print(f"Duration: {total_duration:.2f} seconds")
            self.console.print(f"Sample rate: {self.device_sample_rate} Hz")
            self.console.print(f"Shape: {audio.shape}")
            
            # Calculate chunk size for ~100ms chunks (increased from 50ms)
            chunk_size = int(self.device_sample_rate * 0.1)  # 100ms chunks
            
            # Create a stream for continuous playback
            stream = sd.OutputStream(
                samplerate=self.device_sample_rate,
                channels=1,
                device=self.output_device,
                dtype=np.float32,
                blocksize=chunk_size
            )
            
            # Start playback
            self.console.print("\n[green]Starting playback...[/green]")
            
            with stream:
                start_time = time.time()
                action_idx = 0
                
                # Process audio in chunks
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:min(i + chunk_size, len(audio))]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    
                    current_time = time.time() - start_time
                    
                    # Execute any actions that should occur during this chunk
                    while action_idx < len(action_timings):
                        action_time, tool = action_timings[action_idx]
                        
                        if current_time >= action_time:
                            # Pause the audio stream while executing action
                            stream.stop()
                            self._execute_action(tool, current_time)
                            stream.start()
                            action_idx += 1
                        else:
                            break
                    
                    # Play the chunk
                    stream.write(chunk)
            
            # Cleanup
            os.unlink(temp_wav_path)
            self.tooling.reset_state()
            self.console.print("\n[green]Playback complete![/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error in process_and_speak: {e}[/red]")
            self.tooling.reset_state()
            if 'temp_wav_path' in locals():
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
            raise

    def cleanup(self):
        """Clean up resources"""
        try:
            self.tooling.cleanup()
        except Exception as e:
            self.console.print(f"[red]Error during cleanup: {e}[/red]")

if __name__ == "__main__":
    try:
        tts = ESpeakTTSService()
        
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