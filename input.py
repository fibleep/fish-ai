import time
import threading
import asyncio
import queue
from queue import Queue
import numpy as np
import sounddevice as sd
from rich.console import Console
from langchain_core.messages import HumanMessage
from tts import TTSService
from agent_assets.agent import create_graph
from concurrent.futures import ThreadPoolExecutor
import wave
from datetime import datetime
import os
import whisperx
import torch
import gc

class AudioProcessor:
    def __init__(self):
        self.console = Console()
        self.print_audio_devices()
        
        # Create recordings directory
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Initialize thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # Initialize models and services
        self.device = "cpu"
        self.compute_type = "float32"
        self.batch_size = 4
        
        # Initialize WhisperX model
        self.model = whisperx.load_model(
            "tiny", 
            self.device, 
            compute_type=self.compute_type,
            download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisperx")
        )
            
        self.alignment_model = None
        self.metadata = None
        self.default_language = "en"
        
        self.tts = TTSService()
        self.graph = create_graph()
        
        # Find devices
        self.input_device = self.find_device("USB PnP Sound Device")
        self.output_device = self.find_device("UACDemoV1.0")
        
        # Get device info and sample rate
        input_device_info = sd.query_devices(self.input_device)
        self.sample_rate = int(input_device_info['default_samplerate'])
        
        # Audio settings
        self.channels = 1
        self.block_duration = 0.05
        self.block_size = int(self.sample_rate * self.block_duration)
        
        # Buffer management
        self.input_queue_size = 2048
        self.buffer_pool = []
        self.max_buffer_size = 300
        
        # Silence detection settings
        self.silence_threshold = 0.02
        self.silence_duration = 0.5
        self.last_sound_time = time.time()
        self.min_audio_length = int(self.sample_rate * 0.2)
        self.max_audio_length = int(self.sample_rate * 30)
        
        # Processing control
        self.processing_lock = asyncio.Lock()
        self.is_speaking = False
        self.max_retries = 3
        self.retry_delay = 0.3
        
        # Error handling
        self.error_count = 0
        self.max_errors = 5
        self.error_reset_time = 60

    def cleanup_models(self):
        """Basic cleanup"""
        gc.collect()

    def find_device(self, name_fragment):
        """Find a device by a partial name match"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if name_fragment.lower() in device['name'].lower():
                    self.console.print(f"Found device '{device['name']}' at index {i}")
                    return i
            
            if "input" in name_fragment.lower():
                return sd.default.device[0]
            elif "output" in name_fragment.lower():
                return sd.default.device[1]
                
            self.console.print(f"[yellow]Warning: Device with name '{name_fragment}' not found[/yellow]")
            return None
            
        except Exception as e:
            self.console.print(f"[red]Error finding audio device: {str(e)}[/red]")
            return None

    def print_audio_devices(self):
        """Print all available audio devices"""
        try:
            devices = sd.query_devices()
            self.console.print("\n[bold]Available Audio Devices:[/bold]")
            for i, device in enumerate(devices):
                device_type = []
                if device['max_input_channels'] > 0:
                    device_type.append('INPUT')
                if device['max_output_channels'] > 0:
                    device_type.append('OUTPUT')
                
                status = "🟢" if device['hostapi'] == 0 else "🟡"
                self.console.print(f"{status} {i}: {device['name']} "
                                 f"({', '.join(device_type)}) "
                                 f"- SR: {device['default_samplerate']:.0f}Hz "
                                 f"- Latency: {device['default_low_input_latency']*1000:.1f}ms")
            self.console.print()
        except Exception as e:
            self.console.print(f"[red]Error listing audio devices: {str(e)}[/red]")

    async def transcribe_with_whisperx(self, audio_np):
        """Transcribe audio using WhisperX"""
        if len(audio_np) < self.min_audio_length:
            return ""
        
        try:
            # Direct transcription using WhisperX API
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.model.transcribe(
                    audio_np,
                    batch_size=self.batch_size,
                    language=self.default_language
                )
            )
            
            # Load alignment model if needed
            if self.alignment_model is None:
                self.alignment_model, self.metadata = whisperx.load_align_model(
                    language_code=self.default_language,
                    device=self.device
                )
            
            try:
                aligned_result = whisperx.align(
                    result["segments"],
                    self.alignment_model,
                    self.metadata,
                    audio_np,
                    self.device,
                    return_char_alignments=False
                )
                text = " ".join(segment["text"] for segment in aligned_result["segments"]).strip()
            except Exception as align_error:
                self.console.print(f"[yellow]Alignment failed, using unaligned transcription[/yellow]")
                text = " ".join(segment["text"] for segment in result["segments"]).strip()
            
            if text:
                self.console.print(f"[green]Transcribed:[/green] '{text}'")
            
            return text
            
        except Exception as e:
            self.console.print(f"[red]Transcription error: {str(e)}[/red]")
            return ""

    def _record_audio(self, stop_event, data_queue):
        """Record audio with improved buffering"""
        def callback(indata, frames, time, status):
            if status:
                self.console.print(f"[yellow]Status: {status}[/yellow]")
            
            if not self.is_speaking and not self.processing_lock.locked():
                try:
                    data_queue.put_nowait(bytes(indata))
                except queue.Full:
                    try:
                        data_queue.get_nowait()
                        data_queue.put_nowait(bytes(indata))
                    except (queue.Empty, queue.Full):
                        pass

        try:
            if self.input_device is None:
                raise ValueError("Audio input device not found")

            self.console.print("[green]Starting audio recording...[/green]")
            
            with sd.InputStream(
                device=self.input_device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.block_size,
                callback=callback,
                finished_callback=None,
                clip_off=True,
                prime_output_buffers_using_stream_callback=True
            ) as stream:
                while not stop_event.is_set():
                    time.sleep(0.05)
                    
        except Exception as e:
            self.console.print(f"[red]Recording error: {str(e)}[/red]")
            if not stop_event.is_set():
                time.sleep(0.5)
                self._record_audio(stop_event, data_queue)

    async def process_audio_stream(self):
        """Main audio processing loop"""
        data_queue = Queue(maxsize=self.input_queue_size)
        audio_buffer = []
        stop_event = threading.Event()
        last_error_time = time.time()

        recording_thread = threading.Thread(
            target=self._record_audio,
            args=(stop_event, data_queue),
            daemon=True
        )
        recording_thread.start()

        try:
            while True:
                if not data_queue.empty() and not self.is_speaking:
                    audio_chunk = data_queue.get()
                    chunk_np = np.frombuffer(audio_chunk, dtype=np.float32)

                    if time.time() - last_error_time > self.error_reset_time:
                        self.error_count = 0
                        last_error_time = time.time()

                    if np.max(np.abs(chunk_np)) > self.silence_threshold:
                        self.last_sound_time = time.time()
                        audio_buffer.append(audio_chunk)
                        
                        if len(audio_buffer) > self.max_buffer_size:
                            audio_buffer = audio_buffer[-self.max_buffer_size:]
                            
                    elif (
                        time.time() - self.last_sound_time > self.silence_duration
                        and audio_buffer
                        and not self.processing_lock.locked()
                    ):
                        async with self.processing_lock:
                            await self._process_audio_segment(audio_buffer)
                        audio_buffer = []

                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopping audio processing...[/yellow]")
            stop_event.set()
            recording_thread.join()
            self.cleanup_models()
        except Exception as e:
            self.error_count += 1
            self.console.print(f"[red]Stream processing error: {e}[/red]")
            
            if self.error_count < self.max_errors:
                await self.process_audio_stream()
            else:
                self.console.print("[red]Too many errors occurred. Stopping audio processing.[/red]")
        finally:
            self.thread_pool.shutdown(wait=False)

    async def _process_audio_segment(self, audio_buffer):
        """Process audio segments"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                audio_data = b"".join(audio_buffer)
                audio_np = np.frombuffer(audio_data, dtype=np.float32)

                with self.console.status("[bold green]Transcribing with WhisperX...[/bold green]"):
                    text = await self.transcribe_with_whisperx(audio_np)
                    if not text:
                        return

                with self.console.status("[bold green]Generating response...[/bold green]"):
                    response = await self._get_agent_response(text)
                    self.console.print(f"[blue]Assistant:[/blue] {response}")

                self.is_speaking = True
                try:
                    sample_rate, audio_array = self.tts.process_and_speak(response)
                    await self._play_audio(sample_rate, audio_array)
                finally:
                    self.is_speaking = False
                
                break

            except Exception as e:
                retry_count += 1
                self.console.print(f"[red]Error processing audio: {e}[/red]")
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.console.print("[red]Max retries reached. Please try again.[/red]")
            finally:
                self.is_speaking = False

    async def _get_agent_response(self, text):
        """Get agent response"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.graph.invoke(
                    {"messages": [HumanMessage(content=text)]},
                    config={"configurable": {"thread_id": 42}},
                )
            )
            return self._clean_agent_response(response)
        except Exception as e:
            self.console.print(f"[red]Error getting agent response: {e}[/red]")
            return "I apologize, but I encountered an error. Could you please repeat that?"

    def _clean_agent_response(self, response):
        """Clean agent response"""
        try:
            res = response["messages"][-1].content
            if ":" in res:
                res = res.split(":", 1)[1]
            return res.strip()
        except Exception as e:
            self.console.print(f"[red]Error cleaning response: {e}[/red]")
            return "I apologize, but I couldn't process that correctly. Could you try again?"

    async def _play_audio(self, sample_rate, audio_array):
        """Play audio with file saving"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_filename = os.path.join(self.recordings_dir, f"assistant_response_{timestamp}.wav")
            
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            with wave.open(response_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            if self.output_device is not None:
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: sd.play(audio_array, sample_rate, device=self.output_device) or sd.wait()
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: sd.play(audio_array, sample_rate) or sd.wait()
                )
        except Exception as e:
            self.console.print(f"[red]Error playing audio: {e}[/red]")
            self.is_speaking = False

async def main():
    processor = AudioProcessor()
    console = Console()

    console.print("Assistant started! Press Ctrl+C to exit.")
    
    try:
        await processor.process_audio_stream()
    except KeyboardInterrupt:
        console.print("\nExiting...")
    except Exception as e:
        console.print(f"Error: {e}")
        await main()

    console.print("Session ended.")

if __name__ == "__main__":
    asyncio.run(main())