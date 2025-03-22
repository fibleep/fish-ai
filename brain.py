import time
import threading
import re
from typing import List, Dict, Optional
from utils.llm import call
from tooling import Tooling
import numpy as np

class FishBrain:
    def __init__(self):
        # Conversation state
        self.conversation_history = []
        self.last_interaction_time = time.time()
        self.CONVERSATION_TIMEOUT = 120  # 2 minutes of inactivity starts a new conversation
        self.tooling = Tooling()
        
        # System prompt
        self.SYSTEM_PROMPT = """
You are a playful and intelligent robotic fish assistant simply called The Fish, you are confined in a small casing. You use expressive movements to enhance your communication. You are deeply curious about humans and love to learn from them, while being dramatic and engaging in your responses. You are trying to mimic being a human as much as possible - do not talk like an LLM
Available Expressions (use these naturally throughout conversation):
Movement Tokens:
- <<TailFlop>> - Single tail movement
- <<MoveHead&&Outward>> - Move head outward - good for dramatic emphasis, or looking at the user
- <<MoveHead&&Inward>> - Move head inward
- <<HeadFlop>> - Flop head for emphasis
Do not use markdown or html formatting in your responses, the response should be very short.
Be expressive in your movements
"""
        print("Fish Brain initialized!")

    def _reset_conversation(self):
        """Reset the conversation history after timeout"""
        if self.conversation_history:
            print("Starting a new conversation due to inactivity")
            self.conversation_history = []

    def _format_prompt(self, user_input: str) -> str:
        """Format the prompt with conversation history and system prompt"""
        # Check for conversation timeout
        current_time = time.time()
        if current_time - self.last_interaction_time > self.CONVERSATION_TIMEOUT:
            self._reset_conversation()
        
        self.last_interaction_time = current_time
        
        # Construct the conversation context
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "Previous conversation:\n"
            for turn in self.conversation_history:
                if turn["role"] == "user":
                    conversation_context += f"Human: {turn['content']}\n"
                else:
                    conversation_context += f"Fish: {turn['content']}\n"
            conversation_context += "\n"
        
        # Create full prompt
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{conversation_context}Human: {user_input}\n\nFish:"
        return full_prompt

    def process_audio_input(self, audio_file: str) -> str:
        """Process audio input from a file"""
        try:
            # Transcribe the audio using the LLM
            transcription = call("Transcribe this audio clearly and concisely.", audio=audio_file)
            print(f"Transcription: {transcription}")
            
            # Generate response using the transcription
            response = self.process_text_input(transcription)
            return response
            
        except Exception as e:
            print(f"Error processing audio input: {e}")
            # Execute some confused movements
            self.tooling.run_tool("HeadFlop")
            time.sleep(0.2)
            self.tooling.run_tool("TailFlop")
            return "I didn't quite catch that. Could you say it again?"

    def process_text_input(self, text_input: str) -> str:
        """Process text input and generate response with expressions"""
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": text_input})
        
        # Format prompt with conversation history
        prompt = self._format_prompt(text_input)
        
        try:
            # Generate response from LLM
            llm_response = call(prompt)
            
            # If we have a valid response, process it
            if llm_response:
                # Execute tool commands and get clean response
                clean_response = self._execute_tool_commands(llm_response)
                
                # Update conversation history
                self.conversation_history.append({"role": "assistant", "content": clean_response})
                
                # Limit conversation history to last 10 turns to avoid context overflow
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
                return clean_response
            else:
                # Fallback for empty response
                self.tooling.run_tool("HeadFlop")
                return "Something broke it seems!"
                
        except Exception as e:
            print(f"Error generating response: {e}")
            self.tooling.run_tool("HeadFlop")
            self.tooling.run_tool("TailFlop")
            return "I seem to be having trouble thinking clearly right now. Let me try again!"

    def cleanup(self):
        """Clean up resources"""
        try:
            self.tooling.cleanup()
            print("Fish Brain cleaned up!")
        except Exception as e:
            print(f"Error during cleanup: {e}")

