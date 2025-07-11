import time
import threading
import re
import requests
import asyncio
from typing import List, Dict, Optional
from utils.llm import call
from tooling import Tooling
import numpy as np

class FishBrain:
    def __init__(self, action_processor=None):
        self.conversation_history = []
        self.last_interaction_time = time.time()
        self.CONVERSATION_TIMEOUT = 120
        self.tooling = Tooling()
        self.action_processor = action_processor
        self.tools_ready = False
        
        # Don't build system prompt yet - wait for tools to be loaded
        self.SYSTEM_PROMPT = ""
        print("Fish Brain initialized! Waiting for tools to be loaded...")

    def _build_system_prompt(self):
        """Build the system prompt with available tools"""
        tools_description = self._format_tools_for_prompt()
        device_list = self._get_device_list()
        
        self.SYSTEM_PROMPT = f"""
You are a playful and intelligent robotic fish assistant simply called The Fish, you are confined in a small casing. You use expressive movements to enhance your communication. You are deeply curious about humans and love to learn from them. You are trying to mimic being a human as much as possible - do not talk like an LLM, do not be dramatic, just talk as humanly as possible.

Do not be cringe please

Available Actions (use these XML tokens in your responses):
- <<MoveTail>> - Wag your tail to show excitement
{tools_description}

Use these tokens naturally in your responses like: "Let me turn on the lights <<TurnOn||bedroom light>> for you!"
{device_list}

Do not use markdown or html formatting in your responses, the response should be very short.
Be expressive in your movements
"""
        print(self.SYSTEM_PROMPT)
        self.tools_ready = True

    def _format_tools_for_prompt(self) -> str:
        """Format the available tools for the system prompt"""
        if not self.action_processor or not hasattr(self.action_processor, 'combined_tools'):
            return ""
        
        tools_text = ""
        available_actions = set()
        
        for tool in self.action_processor.combined_tools:
            tool_name = tool.get("name", "")
            tool_desc = tool.get("description", "")
            
            # Map MCP tool names to simple XML tokens dynamically
            if "turn" in tool_name.lower() and "on" in tool_name.lower():
                if "TurnOn" not in available_actions:
                    tools_text += f"- <<TurnOn||device_name>> - Turn on lights or devices\n"
                    tools_text += f"- <<TurnOn||all lights>> - Turn on all lights at once\n"
                    available_actions.add("TurnOn")
            elif "turn" in tool_name.lower() and "off" in tool_name.lower():
                if "TurnOff" not in available_actions:
                    tools_text += f"- <<TurnOff||device_name>> - Turn off lights or devices\n"
                    tools_text += f"- <<TurnOff||all lights>> - Turn off all lights at once\n"
                    available_actions.add("TurnOff")
            elif "light" in tool_name.lower() and "set" in tool_name.lower():
                if "SetLight" not in available_actions:
                    tools_text += f"- <<SetLight||device_name,brightness=50>> - Set light brightness (0-100)\n"
                    available_actions.add("SetLight")
            elif "weather" in tool_name.lower():
                if "GetWeather" not in available_actions:
                    tools_text += f"- <<GetWeather>> - Get weather information\n"
                    available_actions.add("GetWeather")
            elif "time" in tool_name.lower():
                if "GetTime" not in available_actions:
                    tools_text += f"- <<GetTime>> - Get current time\n"
                    available_actions.add("GetTime")
        
        # Add examples of how to use multiple actions
        if available_actions:
            tools_text += "\nExamples:\n"
            tools_text += "- 'Let me help! <<MoveTail>> <<TurnOn||Living Room Light>>'\n"
            tools_text += "- 'Sure! <<TurnOff||all lights>> Turning them all off!'\n"
            tools_text += "- 'I'll brighten that up <<SetLight||Bedroom Light,brightness=75>>'\n"
        
        
        return tools_text.strip()

    def _get_device_list(self) -> str:
        """Get the device list from the MCP wrapper if available"""
        if (self.action_processor and 
            hasattr(self.action_processor, 'mcp_wrapper') and 
            self.action_processor.mcp_wrapper and
            hasattr(self.action_processor.mcp_wrapper, 'get_device_list_for_prompt')):
            device_list = self.action_processor.mcp_wrapper.get_device_list_for_prompt()
            
            # Add usage examples with actual device names if available
            if device_list and hasattr(self.action_processor.mcp_wrapper, 'devices'):
                devices = self.action_processor.mcp_wrapper.devices
                examples = "\n\nExample commands:\n"
                
                # Add light examples
                if devices.get('lights'):
                    first_light = devices['lights'][0]['simple_name']
                    examples += f"- 'Turn on the {first_light} <<TurnOn||{first_light}>>'\n"
                    examples += f"- 'Set {first_light} to 50% brightness <<SetLight||{first_light},brightness=50>>'\n"
                
                # Add switch examples  
                if devices.get('switches'):
                    first_switch = devices['switches'][0]['simple_name']
                    examples += f"- 'Turn off the {first_switch} <<TurnOff||{first_switch}>>'\n"
                
                device_list += examples
            
            return device_list
        return ""

    def update_tools(self):
        """Update the system prompt when new tools are available"""
        self._build_system_prompt()
        print(f"Updated system prompt with {len(self.action_processor.combined_tools) if self.action_processor else 0} tools")
        print("🧠 Brain is now ready with all tools!")

    def _wait_for_tools(self, max_wait_seconds=10):
        """Wait for tools to be loaded, with a timeout"""
        import time
        wait_start = time.time()
        
        while not self.tools_ready and (time.time() - wait_start) < max_wait_seconds:
            time.sleep(0.1)
        
        if not self.tools_ready:
            print("⚠️ Timeout waiting for tools, proceeding with basic system prompt")
            # Create a basic system prompt without HA tools
            self.SYSTEM_PROMPT = """
You are a playful and intelligent robotic fish assistant simply called The Fish, you are confined in a small casing. You use expressive movements to enhance your communication. You are deeply curious about humans and love to learn from them, while being dramatic and engaging in your responses. You are trying to mimic being a human as much as possible - do not talk like an LLM

Do not be cringe please

Available Actions (use these XML tokens in your responses):
- <<MoveTail>> - Wag your tail to show excitement

Use these tokens naturally in your responses!

Your voice can be expressive. You can use adverbs like 'cheerfully', 'sarcastically', or 'sadly' to add emotion to your voice. For example: "Say cheerfully: What a wonderful day!"

Do not use markdown or html formatting in your responses, the response should be very short.
Be expressive in your movements
"""
            self.tools_ready = True

    def _reset_conversation(self):
        if self.conversation_history:
            print("Starting a new conversation due to inactivity")
            self.conversation_history = []

    def _format_prompt(self, user_input: str) -> str:
        current_time = time.time()
        if current_time - self.last_interaction_time > self.CONVERSATION_TIMEOUT:
            self._reset_conversation()
        
        self.last_interaction_time = current_time
        
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "Previous conversation:\n"
            for turn in self.conversation_history:
                if turn["role"] == "user":
                    conversation_context += f"Human: {turn['content']}\n"
                else:
                    conversation_context += f"Fish: {turn['content']}\n"
            conversation_context += "\n"
        
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{conversation_context}Human: {user_input}\n\nFish:"
        return full_prompt

    async def _handle_action_tokens(self, text: str):
        print(f"🔍 Scanning text for action tokens: '{text}'")
        
        # Handle MoveTail through tooling (this will be extracted by tooling.extract_tools)
        if "<<MoveTail>>" in text:
            print("🐟 Found MoveTail token - executing tail movement")
            self.tooling.run_tool("MoveTail")
        else:
            print("🐟 No MoveTail token found")
            
        # MCP tools will be handled during speech by action_processor.process_and_speak()
        print("🔧 MCP tools will be executed during speech for better synchronization")

    async def _handle_mcp_tokens(self, text: str):
        """Handle MCP tool tokens like <<TurnOn||light>>"""
        import re
        
        print(f"🔍 Scanning for MCP tokens in: '{text}'")
        
        # Pattern to match <<ToolName||args>>
        token_pattern = re.compile(r"<<(\w+)(?:\|\|([^>]+))?>>")
        matches = token_pattern.findall(text)
        
        print(f"🎯 Found {len(matches)} potential MCP tokens: {matches}")
        
        # Debug: Show what the regex is actually matching
        all_tokens = re.findall(r"<<[^>]+>>", text)
        print(f"🔍 All <<...>> tokens found: {all_tokens}")
        
        if not matches:
            print("📭 No MCP tokens found")
            return
        
        # Keep track of processed tokens to avoid duplicates
        processed_tokens = set()
        
        for i, (tool_action, args) in enumerate(matches):
            print(f"🔧 Processing token {i+1}/{len(matches)}: {tool_action} with args: '{args}'")
            
            # Skip Arduino tools - they're handled by tooling.extract_tools
            if tool_action == "MoveTail":
                print("⏭️ Skipping MoveTail (handled by Arduino tooling)")
                continue
                
            # Create a unique token identifier
            token_key = f"{tool_action}||{args}" if args else tool_action
            if token_key in processed_tokens:
                print(f"⏭️ Skipping duplicate token: {token_key}")
                continue
            processed_tokens.add(token_key)
                
            # Dynamically find the right tool based on available tools
            actual_tool_name = self._find_tool_for_action(tool_action)
            print(f"🔍 Mapped '{tool_action}' to actual tool: '{actual_tool_name}'")
            
            if actual_tool_name:
                try:
                    # Add a small delay between multiple tool calls to avoid overwhelming HA
                    if i > 0:
                        print(f"⏱️ Waiting 0.5s before next tool call...")
                        await asyncio.sleep(0.5)
                    
                    # Parse args if provided
                    tool_args = {}
                    if args:
                        print(f"📝 Parsing args: '{args}'")
                        # Clean up the device name - remove "light." prefix if present
                        device_name = args.strip()
                        if device_name.startswith("light."):
                            device_name = device_name.replace("light.", "").replace("_", " ").title()
                            print(f"🔧 Cleaned device name: '{device_name}'")
                        
                        # Handle different arg formats
                        if "=" in args:
                            # Format: key=value,key2=value2
                            print("📋 Parsing key=value format")
                            for arg_pair in args.split(","):
                                if "=" in arg_pair:
                                    key, value = arg_pair.split("=", 1)
                                    tool_args[key.strip()] = value.strip()
                                    print(f"  📝 Added arg: {key.strip()} = {value.strip()}")
                        else:
                            # Simple format: just the device name
                            print("📋 Using simple device name format")
                            tool_args["name"] = device_name
                    
                    print(f"🛠️ Executing tool '{actual_tool_name}' with args: {tool_args}")
                    tool_obj = type('obj', (object,), {'name': actual_tool_name, 'args': tool_args})()
                    
                    # Add timeout to prevent hanging
                    try:
                        result = await asyncio.wait_for(
                            self.action_processor.execute_tool_call(tool_obj), 
                            timeout=10.0
                        )
                        print(f"✅ Tool execution result: {result}")
                    except asyncio.TimeoutError:
                        print(f"⏰ Tool execution timeout for {actual_tool_name}")
                    except Exception as exec_e:
                        print(f"❌ Tool execution exception: {exec_e}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    print(f"❌ Error executing {tool_action}: {e}")
            else:
                print(f"⚠️ No matching tool found for '{tool_action}'")

    def _find_tool_for_action(self, action: str) -> Optional[str]:
        """Dynamically find the right tool for a given action"""
        if not self.action_processor or not self.action_processor.combined_tools:
            print("⚠️ No tools available for mapping")
            return None
        
        print(f"🔍 Looking for tool to handle action '{action}' among {len(self.action_processor.combined_tools)} available tools")
        
        # Define action patterns to look for in tool names and descriptions
        action_patterns = {
            "TurnOn": ["turn_on", "turn on", "hass_turn_on", "hassturnon"],
            "TurnOff": ["turn_off", "turn off", "hass_turn_off", "hassturnoff"], 
            "SetLight": ["light_set", "light set", "hass_light_set", "hasslightset"],
            "GetWeather": ["weather", "forecast"],
            "GetTime": ["time", "clock", "datetime"],
            "LightControl": ["light", "brightness", "dimmer"],
            "SwitchControl": ["switch", "outlet", "plug"]
        }
        
        # Get patterns for this action
        patterns = action_patterns.get(action, [action.lower()])
        print(f"🎯 Looking for patterns: {patterns}")
        
        # Search through available tools
        for tool in self.action_processor.combined_tools:
            tool_name = tool.get("name", "").lower()
            tool_desc = tool.get("description", "").lower()
            
            print(f"  🔎 Checking tool: {tool['name']}")
            
            # Check if any pattern matches the tool name or description
            for pattern in patterns:
                if pattern in tool_name or pattern in tool_desc:
                    print(f"  ✅ Found match: '{tool['name']}' matches pattern '{pattern}'")
                    return tool.get("name")
        
        # If no exact match, try broader matching for common actions
        if action in ["TurnOn", "TurnOff"]:
            print(f"  🔄 Trying broader search for {action}")
            for tool in self.action_processor.combined_tools:
                tool_name = tool.get("name", "").lower()
                if action.lower() in tool_name.replace("_", ""):
                    print(f"  ✅ Broad match found: {tool['name']}")
                    return tool.get("name")
        
        print(f"  ❌ No tool found for action '{action}'")
        return None

    async def process_audio_input(self, audio_file: str) -> str:
        try:
            print(f"🎤 Processing audio file: {audio_file}")
            transcription = call("Transcribe this audio clearly and concisely.", audio=audio_file)
            print(f"📝 Transcription: '{transcription}'")
            
            response = await self.process_text_input(transcription)
            print(f"🧠 LLM Response: '{response}'")
            
            # Handle action tokens here after getting the response
            print(f"🔧 Checking for action tokens in response...")
            try:
                await self._handle_action_tokens(response)
                print(f"✅ Action token processing completed")
            except Exception as token_error:
                print(f"❌ Error processing action tokens: {token_error}")
                import traceback
                traceback.print_exc()
            
            return response
            
        except Exception as e:
            print(f"❌ Error processing audio input: {e}")
            import traceback
            traceback.print_exc()
            self.tooling.run_tool("MoveTail")
            return "I didn't quite catch that. Could you say it again?"

    async def process_text_input(self, text_input: str) -> str:
        print(f"💬 Processing text input: '{text_input}'")
        
        # Wait for tools to be ready before processing
        if not self.tools_ready:
            print("🧠 Waiting for tools to be loaded...")
            self._wait_for_tools()
        
        self.conversation_history.append({"role": "user", "content": text_input})
        
        prompt = self._format_prompt(text_input)
        print(f"📋 Generated prompt (first 200 chars): '{prompt[:200]}...'")
        
        try:
            print("🤖 Calling LLM...")
            llm_response = call(prompt)
            print(f"✅ LLM returned: '{llm_response}'")
            
            if llm_response:
                self.conversation_history.append({"role": "assistant", "content": llm_response})
                
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
                # Don't call _handle_action_tokens here - it will be called by process_audio_input
                return llm_response
            else:
                print("⚠️ LLM returned empty response")
                self.tooling.run_tool("MoveTail")
                return "Something broke it seems!"
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            self.tooling.run_tool("MoveTail")
            return "I seem to be having trouble thinking clearly right now. Let me try again!"

    def cleanup(self):
        try:
            self.tooling.cleanup()
            print("Fish Brain cleaned up!")
        except Exception as e:
            print(f"Error during cleanup: {e}")