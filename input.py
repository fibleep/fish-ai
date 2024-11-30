import time
import threading
import asyncio
from queue import Queue
import numpy as np
import whisper
import sounddevice as sd
from langchain_core.messages import HumanMessage
from rich.console import Console
from tts import TTSService 
from agent_assets.agent import create_graph


class AudioProcessor:
    def __init__(self):
        self.console = Console()
        # Load Whisper with specific device and threading settings
        self.stt = whisper.load_model(
            "base.en",
            device="cpu",  # Use CPU for better compatibility
            download_root=None  # Use default download location
        )
        self.tts = TTSService()
        self.graph = create_graph()
        self.silence_threshold = 0.01
        self.silence_duration = 3.0
        self.last_sound_time = time.time()
        # Audio recording settings
        self.sample_rate = 16000
        self.channels = 1
        self.block_duration = 0.05  # 50ms blocks
        self.block_size = int(self.sample_rate * self.block_duration)
        
        # Add processing lock and status flags
        self.processing_lock = asyncio.Lock()
        self.is_speaking = False
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    def _transcribe(self, audio_np):
        try:
            # Reshape and pad audio if necessary
            if len(audio_np.shape) == 1:
                audio_np = audio_np.reshape(-1)
            
            # Ensure audio length is sufficient (at least 1 second)
            min_length = self.sample_rate
            if len(audio_np) < min_length:
                audio_np = np.pad(audio_np, (0, min_length - len(audio_np)))
            
            # Normalize audio if not already normalized
            if np.abs(audio_np).max() > 1:
                audio_np = audio_np / np.abs(audio_np).max()

            # Run transcription with specific parameters
            result = self.stt.transcribe(
                audio_np,
                fp16=False,
                language='en',
                task='transcribe',
                temperature=0.0,  # Use greedy decoding
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )
            
            transcribed_text = result["text"].strip()
            if transcribed_text:
                return transcribed_text
            return ""
            
        except Exception as e:
            self.console.print(f"[red]Transcription error: {e}")
            return ""

    def _record_audio(self, stop_event, data_queue):
        def callback(indata, frames, time, status):
            if status:
                self.console.print(f"[red]Status: {status}")
            # Only queue data if we're not currently processing or speaking
            if not self.is_speaking and not self.processing_lock.locked():
                data_queue.put(bytes(indata))

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
                blocksize=self.block_size,
                callback=callback
            ):
                self.console.print("[green]Recording started...")
                while not stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            self.console.print(f"[red]Recording error: {e}")
            # Attempt to restart recording after error
            if not stop_event.is_set():
                time.sleep(1)
                self._record_audio(stop_event, data_queue)

    async def process_audio_stream(self):
        data_queue = Queue()
        audio_buffer = []
        stop_event = threading.Event()

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
                    chunk_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

                    # Check for silence and not currently processing
                    if np.max(np.abs(chunk_np)) > self.silence_threshold:
                        self.last_sound_time = time.time()
                        audio_buffer.append(audio_chunk)
                    elif (
                        time.time() - self.last_sound_time > self.silence_duration
                        and audio_buffer
                        and not self.processing_lock.locked()
                    ):
                        # Process the accumulated audio
                        async with self.processing_lock:
                            await self._process_audio_segment(audio_buffer)
                        audio_buffer = []

                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            stop_event.set()
            recording_thread.join()
        except Exception as e:
            self.console.print(f"[red]Stream processing error: {e}")
            # Attempt to restart the stream
            await self.process_audio_stream()

    async def _process_audio_segment(self, audio_buffer):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Combine audio chunks and convert to numpy array
                audio_data = b"".join(audio_buffer)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # 1. Transcribe audio
                with self.console.status("[yellow]Transcribing...", spinner="dots"):
                    text = self._transcribe(audio_np)
                    if not text:
                        return
                    self.console.print(f"[yellow]You: {text}")

                # 2. Get agent response
                with self.console.status("[blue]Generating response...", spinner="dots"):
                    response = await self._get_agent_response(text)
                    self.console.print(f"[cyan]Assistant: {response}")

                # 3. Synthesize and play voice response
                self.is_speaking = True
                with self.console.status("Synthesizing speech...", spinner="dots"):
                    try:
                        sample_rate, audio_array = self.tts.process_and_speak(response)
                        await self._play_audio(sample_rate, audio_array)
                    finally:
                        self.is_speaking = False
                
                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                self.console.print(f"[red]Error processing audio (attempt {retry_count}/{self.max_retries}): {e}")
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.console.print("[red]Max retries reached. Please try again.")
            finally:
                self.is_speaking = False

    async def _get_agent_response(self, text):
        try:
            response = self.graph.invoke(
                {"messages": [HumanMessage(content=text)]},
                config={"configurable": {"thread_id": 42}},
            )
            final_response = self._clean_agent_response(response)
            return final_response
        except Exception as e:
            self.console.print(f"[red]Error getting agent response: {e}")
            return "I apologize, but I encountered an error processing your request. Could you please repeat that?"

    def _clean_agent_response(self, response):
        try:
            res = response["messages"][-1].content
            res = res.split(":")[1:]
            res = "".join(res).strip()
            return res
        except (KeyError, IndexError) as e:
            self.console.print(f"[red]Error cleaning response: {e}")
            return "I apologize, but I couldn't process the response correctly. Could you please try again?"

    async def _play_audio(self, sample_rate, audio_array):
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: sd.play(audio_array, sample_rate) or sd.wait()
            )
        except Exception as e:
            self.console.print(f"[red]Error playing audio: {e}")
            self.is_speaking = False  # Ensure flag is reset on error


async def main():
    processor = AudioProcessor()
    console = Console()

    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    try:
        await processor.process_audio_stream()
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
    except Exception as e:
        console.print(f"[red]Error: {e}")
        # Attempt to restart the main loop
        await main()

    console.print("[blue]Session ended.")


if __name__ == "__main__":
    asyncio.run(main())
