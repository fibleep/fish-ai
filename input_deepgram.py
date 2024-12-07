import time
import threading
import asyncio
import queue
from queue import Queue
import numpy as np
import sounddevice as sd
from rich.console import Console
from langchain_core.messages import HumanMessage
import pyaudio
import websockets
import json
import os
from tts_elevenlabs import ElevenLabsTTSService
from tts_espeak import ESpeakTTSService
from agent_assets.agent import create_graph
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from scipy import signal
from load_dotenv import load_dotenv

load_dotenv()

class AudioProcessor:
    def __init__(self):
        self.console = Console()
        
        # Create recordings directory
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Initialize thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # Find devices first to get correct sample rates
        self.input_device = self.find_device("USB PnP Sound Device")
        self.output_device = self.find_device("UACDemoV1.0")
        
        # Get device specific sample rates
        self.input_device_info = sd.query_devices(self.input_device)
        self.output_device_info = sd.query_devices(self.output_device)
        
        # Speech recognition settings
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")
            
        self.audio_queue = asyncio.Queue()
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = int(self.input_device_info['default_samplerate'])
        self.CHUNK = 1024
        self.is_running = False
        
        # Initialize services
        # self.tts = ElevenLabsTTSService()
        self.tts = ESpeakTTSService()
        self.graph = create_graph()
        
        # Silence detection settings
        self.silence_threshold = 0.01
        self.silence_duration = 2.0
        self.last_sound_time = time.time()
        self.current_transcript = ""
        self.processing_transcript = False
        
        # Processing control
        self.processing_lock = asyncio.Lock()
        self.is_speaking = False
        self.max_retries = 3
        self.retry_delay = 0.3
        
        self.console.print(f"Input device sample rate: {self.RATE}")
        self.console.print(f"Output device sample rate: {int(self.output_device_info['default_samplerate'])}")

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

    def resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio using scipy"""
        number_of_samples = round(len(audio_data) * float(target_sr) / orig_sr)
        resampled = signal.resample(audio_data, number_of_samples)
        return resampled

    def audio_callback(self, input_data, frame_count, time_info, status_flags):
        """Callback for audio input with resampling"""
        try:
            # Convert to float32 array
            audio_data = np.frombuffer(input_data, dtype=np.float32)
            
            # Resample to 16kHz for Deepgram if needed
            if self.RATE != 16000:
                audio_data = self.resample_audio(audio_data, self.RATE, 16000)
            
            # Check for sound level
            if np.max(np.abs(audio_data)) > self.silence_threshold:
                self.last_sound_time = time.time()
            
            # Convert to int16 for Deepgram and ensure we're sending bytes
            audio_data_int16 = (audio_data * 32767).astype(np.int16)
            try:
                self.audio_queue.put_nowait(audio_data_int16.tobytes())
            except asyncio.QueueFull:
                # If queue is full, try to clear old data
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data_int16.tobytes())
                except:
                    pass
            
        except Exception as e:
            self.console.print(f"[red]Error in audio callback: {str(e)}[/red]")
        
        return (input_data, pyaudio.paContinue)

    async def process_audio(self):
        """Process audio using Deepgram"""
        protocols = ["token", self.deepgram_api_key]
        
        try:
            async with websockets.connect(
                "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&model=nova-2",
                subprotocols=protocols
            ) as ws:
                await asyncio.gather(
                    self.send_audio(ws),
                    self.receive_transcription(ws),
                    self.check_silence()
                )
        except Exception as e:
            self.console.print(f"[red]Connection error: {str(e)}[/red]")
            raise

    async def send_audio(self, ws):
        """Send audio data to Deepgram with keep-alive"""
        try:
            keep_alive_interval = 5
            last_keep_alive = time.time()
            
            while True:
                try:
                    data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                    if not self.is_speaking:
                        if isinstance(data, bytes):
                            await ws.send(data)
                        else:
                            await ws.send(bytes(data))
                except asyncio.TimeoutError:
                    # Send keep-alive if needed
                    if time.time() - last_keep_alive > keep_alive_interval:
                        keep_alive_msg = json.dumps({"type": "KeepAlive"})
                        await ws.send(keep_alive_msg)
                        last_keep_alive = time.time()
                    continue
                except Exception as e:
                    self.console.print(f"[yellow]Error sending data: {str(e)}[/yellow]")
                    await asyncio.sleep(0.1)
                    continue
                
        except Exception as e:
            self.console.print(f"[red]Send audio error: {str(e)}[/red]")
            try:
                await ws.send(json.dumps({"type": "CloseStream"}))
            except:
                pass
            raise

    async def receive_transcription(self, ws):
        """Receive and process transcriptions"""
        async for msg in ws:
            if self.is_speaking or self.processing_transcript:
                continue
                
            try:
                msg_data = json.loads(msg)
                if msg_data.get("type") == "Results":
                    alternatives = msg_data.get("channel", {}).get("alternatives", [])
                    if alternatives and alternatives[0].get("transcript"):
                        transcript = alternatives[0]["transcript"]
                        if transcript.strip():
                            self.current_transcript = transcript
                            self.console.print(f"[green]Transcribed:[/green] {transcript}")
                            
                            # If this is a final transcription, process it immediately
                            if msg_data.get("speech_final", False):
                                await self.process_current_transcript()
                            
            except json.JSONDecodeError:
                self.console.print("[yellow]Failed to decode message[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error processing transcription: {str(e)}[/red]")

    async def check_silence(self):
        """Check for silence duration and trigger processing"""
        while self.is_running:
            if (not self.is_speaking and 
                not self.processing_transcript and 
                self.current_transcript and 
                time.time() - self.last_sound_time > self.silence_duration):
                
                await self.process_current_transcript()
            
            await asyncio.sleep(0.1)

    async def process_current_transcript(self):
        """Process the current transcript if available"""
        if self.current_transcript and not self.processing_transcript:
            self.processing_transcript = True
            transcript_to_process = self.current_transcript
            self.current_transcript = ""
            
            async with self.processing_lock:
                await self.process_transcript(transcript_to_process)
            
            self.processing_transcript = False

    async def process_transcript(self, transcript):
        """Process the transcript and generate response"""
        try:
            with self.console.status("[bold green]Generating response...[/bold green]"):
                response = await self._get_agent_response(transcript)
                self.console.print(f"[blue]Assistant:[/blue] {response}")

            self.is_speaking = True
            try:
                audio_array = self.tts.process_and_speak(response)
                await self._play_audio(self.RATE, audio_array)
            finally:
                self.is_speaking = False

        except Exception as e:
            self.console.print(f"[red]Error processing transcript: {str(e)}[/red]")
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
        """Play audio response"""
        try:
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

    async def run(self):
        """Main run loop with proper sample rate handling"""
        if not self.is_running:
            self.is_running = True
            
            # Initialize PyAudio with correct sample rate
            audio = pyaudio.PyAudio()
            try:
                stream = audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK,
                    stream_callback=self.audio_callback,
                    input_device_index=self.input_device
                )
                
                stream.start_stream()
                self.console.print("[green]Audio stream started successfully[/green]")
                
                try:
                    await self.process_audio()
                finally:
                    stream.stop_stream()
                    stream.close()
                    
            except Exception as e:
                self.console.print(f"[red]Error opening audio stream: {e}[/red]")
                # Fallback to default device if specific device fails
                try:
                    self.console.print("[yellow]Trying default audio device...[/yellow]")
                    stream = audio.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=44100,  # Common fallback rate
                        input=True,
                        frames_per_buffer=self.CHUNK,
                        stream_callback=self.audio_callback
                    )
                    stream.start_stream()
                    self.console.print("[green]Fallback audio stream started successfully[/green]")
                    
                    try:
                        await self.process_audio()
                    finally:
                        stream.stop_stream()
                        stream.close()
                        
                except Exception as fallback_error:
                    self.console.print(f"[red]Fallback audio stream failed: {fallback_error}[/red]")
            
            finally:
                audio.terminate()
                self.is_running = False

async def main():
    processor = AudioProcessor()
    console = Console()

    console.print("[bold green]Assistant started! Press Ctrl+C to exit.[/bold green]")
    
    try:
        await processor.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        await main()

    console.print("[bold]Session ended.[/bold]")

if __name__ == "__main__":
    asyncio.run(main())