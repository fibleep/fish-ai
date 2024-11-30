import numpy as np
import time
import threading
from rich.console import Console
from rich.live import Live
from rich.text import Text
import torch
import sounddevice as sd
import os
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

class TextToSpeechService:
    def __init__(self):
        self.console = Console(stderr=True)
        
        # Force CPU device for consistency
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.device = torch.device('cpu')
        
        # Load models
        self.console.print("Loading Parler TTS...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        
        # Set Jon's voice description as default
        self.default_description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
        
        self.sample_rate = 24000  # Parler TTS uses 24kHz
        
        self.mouth_shapes = {
            'closed': """
[cyan]    --------
    |      |
    | ---- |
    |      |
    --------[/cyan]""",
            'small': """
[cyan]    --------
    |      |
    | (  ) |
    |      |
    --------[/cyan]""",
            'medium': """
[cyan]    --------
    |      |
    | (()) |
    |      |
    --------[/cyan]""",
            'large': """
[cyan]    --------
    |      |
    |((()))| 
    |      |
    --------[/cyan]"""
        }
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.current_shape = self.mouth_shapes['closed']
        self.cache = {}
        
        self.console.print("[green]TTS Service initialized successfully!")
    
    def _text_to_audio(self, text: str) -> np.ndarray:
        """Convert text to audio using Parler TTS"""
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # Clean up text
        text = text.replace('<<', '').replace('>>', '').replace('<start_of_turn>', '')
        
        with torch.inference_mode():
            try:
                # Prepare inputs
                input_ids = self.tokenizer(self.default_description, return_tensors="pt").input_ids.to(self.device)
                prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                
                # Generate audio
                generation = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids
                )
                
                # Convert to numpy and normalize
                audio = generation.cpu().numpy().squeeze()
                audio = audio / np.abs(audio).max()
                
                # Cache the result
                if len(self.cache) < 100:  # Limit cache size
                    self.cache[text] = audio
                
                return audio
                
            except Exception as e:
                self.console.print(f"[red]Error generating audio: {e}")
                return np.zeros(self.sample_rate)
    
    def _animate(self, audio: np.ndarray, sample_rate: int):
        """Animate mouth based on audio amplitude"""
        frame_duration = 0.016  # ~60fps
        window_size = int(0.016 * sample_rate)
        
        def update_animation():
            with Live(Text.from_markup(self.current_shape), 
                     console=self.console, 
                     refresh_per_second=60,
                     transient=True) as live:
                for i in range(0, len(audio), window_size):
                    if self._stop_event.is_set():
                        break
                        
                    frame = audio[i:i + window_size]
                    if len(frame) == 0:
                        break
                        
                    amplitude = np.sqrt(np.mean(frame**2))
                    
                    with self._lock:
                        if amplitude < 0.05:
                            self.current_shape = self.mouth_shapes['closed']
                        elif amplitude < 0.1:
                            self.current_shape = self.mouth_shapes['small']
                        elif amplitude < 0.2:
                            self.current_shape = self.mouth_shapes['medium']
                        else:
                            self.current_shape = self.mouth_shapes['large']
                    
                    live.update(Text.from_markup(self.current_shape))
                    time.sleep(frame_duration)
                
                self.current_shape = self.mouth_shapes['closed']
                live.update(Text.from_markup(self.current_shape))
        
        animation_thread = threading.Thread(target=update_animation)
        animation_thread.start()
        return animation_thread
    
    def long_form_synthesize(self, text: str, **kwargs) -> tuple[int, np.ndarray]:
        """Handle longer text by synthesizing in chunks"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        all_audio = []
        
        for sentence in sentences:
            if not sentence:
                continue
            
            # Generate audio for sentence
            audio = self._text_to_audio(sentence)
            all_audio.append(audio)
            
            # Add small silence between sentences
            silence = np.zeros(int(0.2 * self.sample_rate))
            all_audio.append(silence)
        
        if not all_audio:
            return self.sample_rate, np.zeros(1000)
        
        # Combine all audio
        full_audio = np.concatenate(all_audio)
        
        # Start animation
        animation_thread = self._animate(full_audio, self.sample_rate)
        
        # Play audio
        sd.play(full_audio, self.sample_rate)
        sd.wait()
        
        # Wait for animation to finish
        animation_thread.join()
        
        return self.sample_rate, full_audio
    
    def cleanup(self):
        self._stop_event.set()

# Example usage
if __name__ == "__main__":
    tts = TextToSpeechService()
    text = "Hello! This is a test of the Parler TTS system. How does it sound?"
    tts.long_form_synthesize(text)
    tts.cleanup()
