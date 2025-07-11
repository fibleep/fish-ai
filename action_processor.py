import os
import time
import numpy as np
import sounddevice as sd
import google.generativeai as genai
import google.generativeai as voiceai # Alias for TTS
from google.generativeai import types
from tooling import Tooling
from rich.console import Console
import io
from pydub import AudioSegment
import warnings
import librosa
import hashlib
import json
import pathlib
from typing import Optional, Dict, List, Tuple
from collections import deque
from dotenv import load_dotenv
import wave
import asyncio
import re
from mcp_wrapper_simple import SimpleMCPWrapper

load_dotenv()

class ActionProcessor:
    def __init__(self, cache_dir="tts_cache"):
        self.console = Console()
        self.console.print("[cyan]Initializing AudioProcessor...[/cyan]")
        
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        warnings.filterwarnings('ignore', category=UserWarning)
        
        self.find_and_set_devices()
        
        self.console.print("\n[cyan]Initializing Arduino tooling...[/cyan]")
        self.tooling = Tooling()
        
        # Use the main 'genai' for any potential text-based models
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Use the aliased 'voiceai' specifically for the TTS model
        self.tts_model = voiceai.GenerativeModel("gemini-2.5-flash-preview-tts")

        self.chunk_size = int(self.device_sample_rate * 0.02)
        self.energy_threshold = 0.1
        self.min_mouth_duration = 0.02
        
        self.last_tool = None
        self.conversation_active = True
        
        self.console.print("[green]AudioProcessor initialization complete![/green]")
        
        self.current_ambient_playback = None

        # MCP and Tool Management
        self.mcp_wrapper: Optional[SimpleMCPWrapper] = None
        self.arduino_tools: List[dict] = []
        self.combined_tools: List[dict] = []
        self._define_arduino_tools()

    async def initialize_mcp_client(self):
        """Initializes the Simple MCP wrapper for Home Assistant tools."""
        try:
            self.console.print("[cyan]Connecting to Home Assistant via SimpleMCPWrapper...[/cyan]")
            self.mcp_wrapper = SimpleMCPWrapper()
            await asyncio.wait_for(self.mcp_wrapper.initialize(), timeout=5.0)
            
            tools_count = len(self.mcp_wrapper.get_tools_for_llm())
            self.console.print(f"[green]✅ Home Assistant connection successful! Found {tools_count} tools.[/green]")
            self._update_combined_tools()
            
        except Exception as e:
            self.console.print(f"[red]❌ Home Assistant connection failed: {e}[/red]")
            self.console.print("[yellow]⚠️ Home Assistant features will not be available.[/yellow]")
            self.mcp_wrapper = None
            self._update_combined_tools()

    def _define_arduino_tools(self):
        """Defines the available Arduino tools in Gemini's format."""
        self.arduino_tools = [
            {
                "name": "MoveTail",
                "description": "Wags the fish's tail for a moment. Use to express excitement or acknowledgement.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            },
        ]
        self._update_combined_tools()
        
    def _update_combined_tools(self):
        """Combines Arduino and Home Assistant tools into a single list."""
        ha_tools = self.mcp_wrapper.get_tools_for_llm() if self.mcp_wrapper else []
        all_tools = self.arduino_tools + ha_tools
        self.combined_tools = all_tools
        self.console.print(f"Total tools available: {len(all_tools)}: {all_tools}")

    async def execute_tool_call(self, tool_call) -> dict:
        """Execute a tool call and return the result"""
        print(f"🛠️ Executing tool call: {tool_call.name}")
        print(f"📝 Tool args: {tool_call.args}")
        
        # Check if it's an Arduino tool first
        for arduino_tool in self.arduino_tools:
            if arduino_tool["name"] == tool_call.name:
                print(f"🤖 Arduino tool detected: {tool_call.name}")
                # Handle Arduino tools through tooling
                try:
                    result = self.tooling.run_tool(tool_call.name, **tool_call.args)
                    print(f"✅ Arduino tool result: {result}")
                    return {"success": True, "result": result}
                except Exception as e:
                    print(f"❌ Arduino tool error: {e}")
                    return {"success": False, "error": str(e)}
        
        # Handle MCP tools
        if self.mcp_wrapper:
            print(f"🌐 Executing MCP tool: {tool_call.name}")
            try:
                result = await self.mcp_wrapper.call_tool(tool_call.name, tool_call.args)
                print(f"✅ MCP tool result: {result}")
                return result
            except Exception as e:
                print(f"❌ MCP tool error: {e}")
                return {"success": False, "error": str(e)}
        else:
            print("⚠️ No MCP wrapper available")
            return {"success": False, "error": "No MCP wrapper available"}

    def find_and_set_devices(self):
        devices = sd.query_devices()
        self.console.print("\n[yellow]Available Audio Devices:[/yellow]")
        
        # FORCE device 3 (CONEXANT USB AUDIO) - override everything else
        target_device_index = 3
        self.console.print(f"[bold red]FORCING device index 3 (CONEXANT USB AUDIO)[/bold red]")
        
        output_device_name = os.getenv('OUTPUT_DEVICE_NAME', 'CONEXANT USB AUDIO')
        output_device_index_str = os.getenv('OUTPUT_DEVICE_INDEX')

        if output_device_index_str:
            try:
                target_device_index = int(output_device_index_str)
                self.console.print(f"[bold green]Using output device index from environment variable: {target_device_index}[/bold green]")
            except ValueError:
                self.console.print(f"[red]Invalid OUTPUT_DEVICE_INDEX: '{output_device_index_str}'. Must be an integer.[/red]")

        if target_device_index is None:
            for i, device in enumerate(devices):
                device_type = []
                if device['max_input_channels'] > 0:
                    device_type.append('INPUT')
                if device['max_output_channels'] > 0:
                    device_type.append('OUTPUT')
                
                self.console.print(f"[cyan]{i}:[/cyan] {device['name']} "
                                f"({(', '.join(device_type))}) "
                                f"- Default SR: {device['default_samplerate']}")
                
                if output_device_name in device['name'] and 'OUTPUT' in device_type:
                    target_device_index = i
                    self.console.print(f"[bold green]Found matching output device: {device['name']} at index {i}[/bold green]")

        if target_device_index is None:
            self.output_device = 3 
            self.console.print(f"\n[yellow]Warning: Could not find '{output_device_name}'. Using forced device index: {self.output_device} (CONEXANT USB AUDIO)[/yellow]")
        else:
            self.output_device = target_device_index
            self.console.print(f"\n[green]Using output device index: {self.output_device}[/green]")


        device_info = sd.query_devices(self.output_device)
        default_sample_rate = int(device_info['default_samplerate'])
        
        # Try common sample rates if the default doesn't work
        sample_rates_to_try = [default_sample_rate, 44100, 48000, 22050, 16000]
        
        self.device_sample_rate = None
        for rate in sample_rates_to_try:
            try:
                # Test if this sample rate works
                sd.default.device = (None, self.output_device)
                sd.default.samplerate = rate
                sd.default.channels = (None, 1)
                sd.default.dtype = ('float32', 'float32')
                
                # Try a quick test to see if it works
                test_audio = np.zeros(1024, dtype=np.float32)
                sd.play(test_audio, rate, device=self.output_device)
                sd.stop()
                
                self.device_sample_rate = rate
                self.console.print(f"[green]Using sample rate: {rate} Hz[/green]")
                break
            except Exception as e:
                self.console.print(f"[yellow]Sample rate {rate} Hz failed: {e}[/yellow]")
                continue
        
        if self.device_sample_rate is None:
            # Fallback to a safe default
            self.device_sample_rate = 44100
            self.console.print(f"[red]All sample rates failed, using fallback: {self.device_sample_rate} Hz[/red]")
        
        self.console.print(f"[green]Using output device: {device_info['name']}[/green]")
        self.console.print(f"[green]Final sample rate: {self.device_sample_rate} Hz[/green]")

    def _generate_cache_key(self, text: str, voice_name: str) -> str:
        params = {
            'text': text,
            'voice_name': voice_name,
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        cache_file = self.cache_dir / f"{cache_key}.wav"
        if cache_file.exists():
            self.console.print("[green]Found cached audio![/green]")
            return cache_file.read_bytes()
        return None

    def _save_to_cache(self, cache_key: str, audio_data: bytes):
        cache_file = self.cache_dir / f"{cache_key}.wav"
        cache_file.write_bytes(audio_data)
        self.console.print("[green]Saved audio to cache[/green]")

    def analyze_audio_energy(self, audio: np.ndarray) -> list:
        chunks = np.array_split(audio, len(audio) // self.chunk_size)
        mouth_states = []
        
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
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
        samples = np.array(audio_segment.get_array_of_samples())
        audio = samples.astype(np.float32)
        
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
        
        buffer_samples = int(self.device_sample_rate * 0.5)
        silent_buffer = np.zeros(buffer_samples, dtype=np.float32)
        audio_with_buffer = np.concatenate([audio, silent_buffer])
        
        return audio_with_buffer.astype(np.float32)

    def _generate_audio(self, text: str, voice_name: str = 'Algieba') -> Optional[bytes]:
        self.console.print("\n[cyan]Generating speech with Google...[/cyan]")
        print(f"🎙️ TTS Input text: '{text}'")
        print(f"🎵 Using voice: {voice_name}")
        
        try:
            tts_config = {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": { "voice_name": voice_name }
                    }
                }
            }
            print(f"⚙️ TTS Config: {tts_config}")
            
            print("📞 Calling Google TTS API...")
            response = self.tts_model.generate_content(
                contents=text,
                generation_config=tts_config
            )
            
            # The API returns raw PCM data, so we wrap it in a WAV header
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            print(f"📥 Received PCM data: {len(pcm_data)} bytes")
            
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(1)       # Mono
                    wf.setsampwidth(2)       # 16-bit
                    wf.setframerate(24000)   # Google TTS default sample rate
                    wf.writeframes(pcm_data)
                wav_data = wav_buffer.getvalue()
                print(f"📦 Created WAV file: {len(wav_data)} bytes")
                return wav_data

        except Exception as e:
            self.console.print(f"[red]Error during Google TTS generation: {e}[/red]")
            print(f"❌ Detailed TTS error: {e}")
            return None

    def play_sound(self, sound_file):
        try:
            import subprocess
            # Use aplay directly with plughw:3,0 for consistency
            subprocess.run(["aplay", "-D", "plughw:3,0", sound_file], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"Played sound: {sound_file} on plughw:3,0")
        except Exception as e:
            print(f"Error playing sound: {e}")

    def play_ambient_sound(self, sound_file):
        try:
            self.stop_ambient_sound()
            
            with open(sound_file, 'rb') as f:
                audio_data = f.read()
            
            audio = self.process_audio(audio_data)
            
            def _callback(outdata, frames, time, status):
                nonlocal audio 
                if len(audio) > 0:
                    chunk = audio[:frames]
                    audio = audio[frames:]
                    outdata[:len(chunk)] = chunk.reshape(-1,1)
                    if len(chunk) < frames:
                        outdata[len(chunk):] = 0
                else:
                    outdata.fill(0)
            
            self.current_ambient_playback = sd.OutputStream(
                samplerate=self.device_sample_rate,
                device=self.output_device,
                channels=1,
                callback=_callback,
                blocksize=self.chunk_size
            )
            self.current_ambient_playback.start()
            print(f"Started ambient sound: {sound_file}")
            
        except Exception as e:
            print(f"Error playing ambient sound: {e}")

    def stop_ambient_sound(self):
        if self.current_ambient_playback is not None:
            self.current_ambient_playback.stop()
            self.current_ambient_playback.close()
            self.current_ambient_playback = None
            print("Stopped ambient sound")

    def reset_conversation(self):
        self.conversation_active = False
        
        if self.tooling.serial_conn and self.tooling.serial_conn.is_open:
            self.tooling.move_mouth(False)
            self.last_tool = None
        
        self.conversation_active = True
        
    def preprocess(self, text: str):
        print(f"🔄 Preprocessing text: '{text}'")
        original_text = text  # Keep original for timing calculations
        
        # Extract both Arduino and MCP tools
        cleaned_text, arduino_tool_positions = self.tooling.extract_tools(text)
        mcp_tool_positions = self._extract_mcp_tools(text)
        
        print(f"🧹 Cleaned text: '{cleaned_text}'")
        print(f"🔧 Arduino tool positions: {arduino_tool_positions}")
        print(f"🌐 MCP tool positions: {mcp_tool_positions}")
        
        voice_name = 'Algieba'
        cache_key = self._generate_cache_key(cleaned_text, voice_name)
        print(f"🔑 Cache key: {cache_key}")
        
        audio_data = self._get_cached_audio(cache_key)
        if audio_data is None:
            print("💾 No cached audio found, generating new audio...")
            audio_data = self._generate_audio(cleaned_text, voice_name)
            if audio_data:
                print("💾 Saving audio to cache...")
                self._save_to_cache(cache_key, audio_data)
        else:
            print("💾 Using cached audio")
        
        if not audio_data:
            self.console.print("[red]Failed to generate or retrieve audio. Skipping playback.[/red]")
            print("❌ No audio data available")
            return original_text, cleaned_text, arduino_tool_positions, mcp_tool_positions, np.array([], dtype=np.float32)

        print("🎵 Processing audio data...")
        audio = self.process_audio(audio_data)
        print(f"🎵 Processed audio shape: {audio.shape}")
        return original_text, cleaned_text, arduino_tool_positions, mcp_tool_positions, audio

    def _extract_mcp_tools(self, text: str) -> Dict[str, Dict]:
        """Extract MCP tool calls from text and return their positions and parameters"""
        mcp_tools = {}
        
        # Find all MCP tool tokens like <<TurnOn||Dirk>>
        for match in re.finditer(r"<<(.*?)>>", text):
            full_token = match.group(0)  # <<TurnOn||Dirk>>
            tool_content = match.group(1)  # TurnOn||Dirk
            
            if "||" in tool_content:
                tool_name, tool_param = tool_content.split("||", 1)
                
                # Check if this is an MCP tool (not Arduino)
                arduino_tools = ["MoveTail", "HeadFlop", "TailFlop", "MoveHead"]
                if tool_name not in arduino_tools:
                    mcp_tools[full_token] = {
                        'name': tool_name,
                        'param': tool_param,
                        'position': match.start()
                    }
        
        return mcp_tools
        

    async def process_and_speak(self, original_text, cleaned_text, arduino_tool_positions, mcp_tool_positions, audio):
        if audio is None:
            self.console.print("[yellow]No audio data to play.[/yellow]")
            return

        self.console.print(f"Speaking: {cleaned_text}")
        
        mouth_states = self.analyze_audio_energy(audio)
        
        # Calculate timing for MCP tools based on word timing
        total_duration = len(audio) / self.device_sample_rate
        mcp_tool_timings = self._calculate_tool_timings(original_text, cleaned_text, mcp_tool_positions, total_duration)
        executed_mcp_tools = set()
        active_tasks = []
        
        self.console.print("\n[green]Starting playback...[/green]")
        
        # Save audio to temp file and use aplay for consistent output device
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Convert audio to WAV format
            with wave.open(tmp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(int(self.device_sample_rate))
                # Convert float32 audio to int16
                audio_int16 = (audio * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"🎵 Playing TTS via aplay on plughw:3,0: {tmp_file.name}")
            
            # Use aplay with the same device as beeps
            import subprocess
            playback_process = subprocess.Popen(
                ["aplay", "-D", "plughw:3,0", tmp_file.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        playback_start_time = time.time()
        
        chunk_duration = self.chunk_size / self.device_sample_rate
        last_mouth_state = False
        mouth_state_pos = 0
        last_mouth_change_time = 0
        
        self.tooling.move_mouth(False)

        while playback_process.poll() is None:
            current_playback_time = time.time() - playback_start_time
            
            mouth_state_pos = int(current_playback_time / chunk_duration)
            
            # Handle mouth movement
            if mouth_state_pos < len(mouth_states):
                should_open = mouth_states[mouth_state_pos]
                
                if should_open != last_mouth_state and (time.time() - last_mouth_change_time) > self.min_mouth_duration:
                    self.tooling.move_mouth(should_open)
                    last_mouth_state = should_open
                    last_mouth_change_time = time.time()
            
            # Execute MCP tools at the right time
            for tool_token, timing_info in mcp_tool_timings.items():
                if (tool_token not in executed_mcp_tools and 
                    current_playback_time >= timing_info['execute_time']):
                    
                    print(f"🌐 Executing MCP tool {timing_info['name']} with param {timing_info['param']} at {current_playback_time:.2f}s")
                    # Create task and track it
                    task = asyncio.create_task(self._execute_mcp_tool_async(timing_info['name'], timing_info['param']))
                    active_tasks.append(task)
                    executed_mcp_tools.add(tool_token)

            time.sleep(0.01)
            
        # Wait for aplay to finish
        playback_process.wait()
        
        # Clean up temp file
        import os
        try:
            os.unlink(tmp_file.name)
        except:
            pass
        
        self.tooling.move_mouth(False)
        
        # Wait for any remaining MCP tasks to complete
        if active_tasks:
            print(f"⏳ Waiting for {len(active_tasks)} MCP tasks to complete...")
            try:
                await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=5.0)
                print("✅ All MCP tasks completed")
            except asyncio.TimeoutError:
                print("⏰ Some MCP tasks timed out")
            except Exception as e:
                print(f"❌ Error waiting for MCP tasks: {e}")
        
        self.console.print("\n[green]Playback complete![/green]")

    def _calculate_tool_timings(self, original_text: str, cleaned_text: str, mcp_tool_positions: Dict, total_audio_duration: float) -> Dict:
        """Calculate when each MCP tool should be executed during speech based on word timing"""
        timings = {}
        
        # Split text into words and estimate timing
        words = cleaned_text.split()
        if not words:
            return timings
            
        # Estimate words per second (typical speech is 2-3 words per second)
        words_per_second = len(words) / total_audio_duration if total_audio_duration > 0 else 2.5
        
        for tool_token, tool_info in mcp_tool_positions.items():
            # Find where this tool was in the original text
            original_position = tool_info['position']
            
            # Get the text before this tool position in the original text
            text_before_tool = original_text[:original_position]
            
            # Count words before this tool position
            words_before = len(text_before_tool.split())
            
            # Calculate execution time based on word count
            execute_time = max(0, words_before / words_per_second)
            
            timings[tool_token] = {
                'name': tool_info['name'],
                'param': tool_info['param'],
                'execute_time': execute_time,
                'words_before': words_before
            }
            
            print(f"🎯 Tool {tool_info['name']}({tool_info['param']}) scheduled for {execute_time:.2f}s ({words_before} words in)")
            
        return timings

    async def _execute_mcp_tool_async(self, tool_name: str, param: str):
        """Execute an MCP tool asynchronously"""
        if not self.mcp_wrapper:
            print(f"⚠️ No MCP wrapper available for {tool_name}")
            return
            
        try:
            # Map tool names to MCP functions
            if tool_name == "TurnOn":
                # Check if this is "all lights" request
                if param.lower() in ["all lights", "all the lights", "every light", "all lamps"]:
                    print(f"🌟 Detected 'all lights on' request - using HassLightTurnOnAll tool")
                    result = await self.mcp_wrapper.call_tool("mcp_Home_Assistant_HassLightTurnOnAll", {})
                else:
                    result = await self.mcp_wrapper.call_tool("mcp_Home_Assistant_HassTurnOn", {"name": param})
            elif tool_name == "TurnOff":
                # Check if this is "all lights" request
                if param.lower() in ["all lights", "all the lights", "every light", "all lamps"]:
                    print(f"🌟 Detected 'all lights off' request - using HassLightTurnOffAll tool")
                    result = await self.mcp_wrapper.call_tool("mcp_Home_Assistant_HassLightTurnOffAll", {})
                else:
                    result = await self.mcp_wrapper.call_tool("mcp_Home_Assistant_HassTurnOff", {"name": param})
            elif tool_name == "SetBrightness":
                # Parse brightness from param if needed
                result = await self.mcp_wrapper.call_tool("mcp_Home_Assistant_HassLightSet", {"name": param})
            else:
                print(f"⚠️ Unknown MCP tool: {tool_name}")
                return
                
            print(f"✅ MCP tool {tool_name} executed successfully: {result}")
        except Exception as e:
            print(f"❌ Error executing MCP tool {tool_name}: {e}")

    async def cleanup(self):
        self.console.print("Cleaning up ActionProcessor...")
        if self.mcp_wrapper:
            await self.mcp_wrapper.close()
        self.tooling.cleanup()

    def clear_cache(self):
        for item in self.cache_dir.iterdir():
            item.unlink()
        self.console.print("[green]Cache cleared successfully![/green]")

