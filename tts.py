import os
import threading
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
import sounddevice as sd
import torch
from rich.console import Console
from rich.live import Live
from rich.text import Text
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import serial
import subprocess
import json
import re
from queue import Queue
from tooling import Tooling


class TTSService:
    def __init__(self, device: str = None):
        self.console = Console(stderr=True)
        self.console.print("[yellow]Initializing Integrated TTS engine...")
        
        # Initialize Tooling
        self.tooling = Tooling()
        
        # Initialize TTS components
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir=".model_cache")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, cache_dir=".model_cache").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, cache_dir=".model_cache").to(self.device)
        
        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir=".model_cache")
        self.speaker_embeddings = torch.tensor(dataset[7306]["xvector"], dtype=torch.float16 if self.device == "cuda" else torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        self.vocoder.eval()
        
        self.sample_rate = 16000
        self._stop_event = threading.Event()
        self._audio_cache = {}
        
        self.console.print("[green]Integrated TTS Service initialized successfully!")

    def process_and_speak(self, text: str):
        """Process text with tools and synthesize speech"""
        # Extract tools and clean text
        cleaned_text, tool_positions = self.tooling.extract_tools(text)
        
        # Create list of actions with their timing positions
        actions = [(pos, tool) for tool, pos in tool_positions.items()]
        actions.sort()
        
        print(f"\nFound {len(actions)} actions in text")
        for pos, tool in actions:
            print(f"Position {pos}: {tool}")
        
        # Generate audio first
        with torch.inference_mode():
            audio = self._text_to_audio(cleaned_text)
        
        total_duration = len(audio) / self.sample_rate
        char_to_time_ratio = total_duration / len(cleaned_text)
        
        # Calculate action timings and ensure end actions have proper timing
        action_timings = []
        for pos, tool in actions:
            # If action is near the end, make sure it still executes
            action_time = min(pos * char_to_time_ratio, total_duration - 0.5)
            action_timings.append((action_time, tool))
            print(f"Scheduled {tool} at {action_time:.2f} seconds")

        try:
            # Initialize state
            is_mouth_open = False
            is_action_in_progress = False
            
            # Start playback
            sd.play(audio, self.sample_rate, blocking=False)
            start_time = time.time()
            
            action_idx = 0
            last_amplitude_check = 0
            
            while sd.get_stream().active or action_idx < len(action_timings):  # Keep running until all actions are done
                current_time = time.time() - start_time
                
                # Handle scheduled actions
                while action_idx < len(action_timings):
                    action_time, tool = action_timings[action_idx]
                    
                    # If it's time for the next action
                    if current_time >= action_time:
                        print(f"\nExecuting action {tool} at {current_time:.2f}s")
                        
                        # Close mouth if needed before action
                        if is_mouth_open:
                            self.tooling.run_tool("MouthClose")
                            is_mouth_open = False
                            time.sleep(0.1)
                        
                        # Execute the action
                        is_action_in_progress = True
                        self.tooling.run_tool(tool)
                        
                        # For flop actions, wait for completion
                        if tool in ["TailFlop", "HeadFlop"]:
                            time.sleep(0.3)  # Wait for flop to complete
                        
                        action_idx += 1
                        is_action_in_progress = False
                        continue
                    break
                
                # Only handle mouth movements if audio is still playing
                if sd.get_stream().active and not is_action_in_progress:
                    chunk_start = int(current_time * self.sample_rate)
                    chunk_end = chunk_start + int(0.05 * self.sample_rate)
                    
                    if chunk_end < len(audio):
                        amplitude = np.sqrt(np.mean(audio[chunk_start:chunk_end]**2))
                        
                        # Check next action timing
                        next_action_time = action_timings[action_idx][0] if action_idx < len(action_timings) else float('inf')
                        
                        # Only move mouth if we're not too close to next action
                        if current_time < (next_action_time - 0.2):
                            if amplitude > 0.1 and not is_mouth_open:
                                self.tooling.run_tool("MouthOpen")
                                is_mouth_open = True
                            elif amplitude <= 0.1 and is_mouth_open:
                                self.tooling.run_tool("MouthClose")
                                is_mouth_open = False
                    
                    last_amplitude_check = current_time
                
                time.sleep(0.01)
            
            # Ensure mouth is closed at the end
            if is_mouth_open:
                self.tooling.run_tool("MouthClose")
            
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            try:
                # More robust cleanup
                if is_mouth_open:
                    self.tooling.run_tool("MouthClose")
                self.tooling.reset_state()
            except Exception as e:
                print(f"Error during final cleanup: {e}")

    def _text_to_audio(self, text: str) -> np.ndarray:
        """Convert text to audio"""
        try:
            sentences = sent_tokenize(text)
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
            return audio

        except Exception as e:
            print(f"Error generating audio: {e}")
            return np.zeros(self.sample_rate)

    def cleanup(self):
        """Clean up resources"""
        try:
            self._stop_event.set()
            self.tooling.cleanup()
        except Exception as e:
            print(f"Error during service cleanup: {e}")
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
        print(f"Error: {e}")
    finally:
        if 'tts' in locals():
            tts.cleanup()
