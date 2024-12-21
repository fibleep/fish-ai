import os
import threading
import time
import numpy as np
import sounddevice as sd
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import librosa
from tooling import Tooling
from rich.console import Console

class TTSService:
    def __init__(self):
        self.console = Console()
        self.console.print("[cyan]Initializing TTS Service...[/cyan]")
        
        # Set sample rates first
        self.model_sample_rate = 16000
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
        
        # Initialize TTS components
        self.console.print("\n[cyan]Loading TTS models...[/cyan]")
        self.device = "cpu"
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir=".model_cache")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir=".model_cache").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=".model_cache").to(self.device)
        
        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir=".model_cache")
        self.speaker_embeddings = torch.tensor(dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        
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
        sd.default.dtype = ('int16', 'float32')
        
        self.console.print(f"[green]Using output device: {device_info['name']}[/green]")
        self.console.print(f"[green]Sample rate: {self.device_sample_rate} Hz[/green]")

    def process_and_speak(self, text: str):
        """Process text with tools and synthesize speech"""
        try:
            # Extract tools and clean text
            cleaned_text, tool_positions = self.tooling.extract_tools(text)
            
            # Create list of actions with their timing positions
            actions = [(pos, tool) for tool, pos in tool_positions.items()]
            actions.sort()
            
            self.console.print(f"\nFound {len(actions)} actions in text")
            for pos, tool in actions:
                self.console.print(f"Position {pos}: {tool}")
            
            # Generate audio first
            self.console.print("\n[cyan]Generating speech...[/cyan]")
            sentences = sent_tokenize(cleaned_text)
            audio_chunks = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                inputs = self.processor(text=sentence, return_tensors="pt")
                speech = self.model.generate_speech(
                    inputs["input_ids"].to(self.device), 
                    self.speaker_embeddings,
                    vocoder=self.vocoder
                )
                audio_chunk = speech.cpu().numpy()
                audio_chunks.append(audio_chunk)
            
            audio = np.concatenate(audio_chunks)
            audio = audio / (np.abs(audio).max() + 1e-6)
            
            # Resample to match device
            self.console.print("[cyan]Resampling audio...[/cyan]")
            audio = librosa.resample(audio, orig_sr=self.model_sample_rate, target_sr=self.device_sample_rate)
            
            # Ensure audio is in the correct format and volume
            audio = audio.astype(np.float32)
            audio = audio * 2  # Adjust volume
            
            total_duration = len(audio) / self.device_sample_rate
            char_to_time_ratio = total_duration / len(cleaned_text)
            
            # Calculate action timings
            action_timings = []
            for pos, tool in actions:
                action_time = min(pos * char_to_time_ratio, total_duration - 0.5)
                action_timings.append((action_time, tool))
                self.console.print(f"Scheduled {tool} at {action_time:.2f} seconds")
            
            self.console.print(f"\n[cyan]Audio info:[/cyan]")
            self.console.print(f"Duration: {total_duration:.2f} seconds")
            self.console.print(f"Sample rate: {self.device_sample_rate} Hz")
            self.console.print(f"Shape: {audio.shape}")
            self.console.print(f"Data type: {audio.dtype}")
            self.console.print(f"Value range: {audio.min():.3f} to {audio.max():.3f}")
            
            # Start playback
            self.console.print("\n[green]Starting playback...[/green]")
            sd.play(audio, self.device_sample_rate, device=self.output_device)
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
            
            return self.device_sample_rate, audio
            
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
        tts = TTSService()
        
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