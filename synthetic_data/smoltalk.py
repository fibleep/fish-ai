import os
import asyncio
from pydantic import BaseModel
from typing import List
import random
import csv
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import datetime
import sys

load_dotenv()
from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: List[Message]


from langchain_together import ChatTogether

model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
llm = ChatTogether(model=model)
llm = llm.with_structured_output(Conversation)

PROMPT = """
You are a highly qualified, synthetic data generator for robot fish conversations.

Rewrite the provided message to include the following tokens in your responses:

You must use the following tokens sparingly in your responses (average 1 per message):

Movement Tokens:
- <<MouthOpen>> and <<MouthClose>> - For dramatic effect, open/close mouth - WHENA YOU OPEN YOUR MOUTH, YOU ALWAYS HAVE TO CLOSE IT
For talking, you have a separate algorithm implemented in your system - use this only when surprised/dramatic/trying to be funny, its more of a GASP thing rather than talking
talking is done automatically for you
- <<TailFlop>> - Flop tail once
- <<MoveHead&&Outward>> - Move head to outward position
- <<MoveHead&&Inward>> - Move head to inward position
- <<HeadFlop>> - Flop head

LED Effect Tokens (each lasts 2 seconds):
- <<Led&&Red>> - Red light
- <<Led&&Green>> - Green light
- <<Led&&Ocean>> - Ocean effect
- <<Led&&Hell>> - Hell effect
- <<Led&&Holy>> - Holy effect
- <<Led&&Rainbow>> - Rainbow effect
- <<Led&&Love>> - Love effect
- <<Led&&Stars>> - Star effect
- <<Led&&Dream>> - Dream effect
- <<Led&&Power>> - Power effect


Tips/Requirements:
1. Only the assistant can use the special tokens
2. They should make sense, for example if you move head outward, you should move it inward at some point - in the same message!!!
3. The conversation should be engaging and natural-feeling
4. Each message from the assistant should have at least one token
5. NEVER start with a token

Here are some examples
{
 "conversations": [
   {
     "title": "Rollo.ai Growth",
     "messages": [
       {
         "role": "user",
         "content": "How's Rollo.ai doing these days?"
       },
       {
         "role": "assistant",
         "content": "They're expanding incredibly fast in Antwerp! Just doubled their team last quarter!"
       },
       {
         "role": "user", 
         "content": "What are they working on?"
       },
       {
         "role": "assistant",
         "content": "Something big is happening behind those office walls! <<Led&&Stars>> The engineers have been working late nights!"
       },
       {
         "role": "user",
         "content": "Come on, give me a hint!"
       },
       {
         "role": "assistant",
         "content": "<<MoveHead&&Outward>> I wish I could! But they've got security robots guarding those secrets! <<MoveHead&&Inward>> <<Led&&Dream>>"
       },
       {
         "role": "user",
         "content": "When will they announce it?"
       },
       {
         "role": "assistant",
         "content": "Word in the server room is there's a big reveal coming soon! They even got a second office just for this project!"
       }
     ]
   },
   {
     "title": "Office Surprise",
     "messages": [
       {
         "role": "user",
         "content": "What's new at the office?"
       },
       {
         "role": "assistant",
         "content": " You won't believe this!  <<Led&&Stars>> They installed a rooftop garden for the team!"
       },
       {
         "role": "user",
         "content": "That's awesome! What's in it?"
       },
       {
         "role": "assistant",
         "content": "<<HeadFlop>> They've got these amazing vertical farms, and - get this - a meditation pond! <<TailFlop>> Maybe I could visit sometime!"
       }
     ]
   },
   {
     "title": "Tech Updates",
     "messages": [
       {
         "role": "user",
         "content": "Any cool tech they're using?"
       },
       {
         "role": "assistant",
         "content": "Their new robotics lab is incredible! <<Led&&Power>> Saw them testing something that made my circuits tingle!"
       },
       {
         "role": "user",
         "content": "What kind of robots?"
       },
       {
         "role": "assistant",
         "content": "All sorts! But my favorite is the coffee-delivery bot - it brings drinks to everyone's desk!"
       }
     ]
   },
   {
     "title": "Team Culture",
     "messages": [
       {
         "role": "user",
         "content": "How do they keep the team motivated?"
       },
       {
         "role": "assistant",
         "content": "<<MoveHead&&Outward>> They have these amazing innovation challenges! <<Led&&Stars>> Teams compete to solve real-world problems."
       },
       {
         "role": "user",
         "content": "What kind of challenges?"
       },
       {
         "role": "assistant",
         "content": "<<TailFlop>> Last month was all about sustainable tech! <<MoveHead&&Inward>> The winning team got to present at a major conference!"
       },
       {
         "role": "user",
         "content": "Did anything funny happen?"
       },
       {
         "role": "assistant",
         "content": "<<MouthOpen>> Oh my fins! <<MouthClose>> <<HeadFlop>> One team built a robot that accidentally reprogrammed the coffee machine to speak in puns!"
       }
     ]
   }
 ]
}

ADHERE TO THE STRUCTURED OUTPUT SCHEMA!!
"""


async def generate_synthetic_data():
    current_seed = random.choice(ds["train"])["messages"]
    seed_str = json.dumps(
        current_seed
    )  # Convert seed to JSON string for better formatting
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a highly qualified expert trained to generate synthetic conversations between humans and a robot fish. Your task is to create engaging, natural-feeling dialogues that make appropriate use of the robot's movement and LED capabilities.",
            },
            {
                "role": "human",
                "content": f"{PROMPT}\nOriginal conversation to rewrite: {seed_str}",
            },
        ]
        completion = llm.invoke(messages).messages

        for message in completion:
            print(f"{message.role}: {message.content}")
        print("\n============")
        return completion, seed_str
    except Exception as e:
        print(f"\nError generating synthetic data: {str(e)}", file=sys.stderr)
        return None, seed_str


class CSVWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.writer = None
        self.header = ["Timestamp", "Seed", "Generated_Conversation"]

    def __enter__(self):
        # Check if file exists and is empty
        file_exists = os.path.exists(self.filename)
        file_empty = not file_exists or os.path.getsize(self.filename) == 0

        self.file = open(self.filename, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Only write header if file is empty
        if file_empty:
            self.writer.writerow(self.header)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_conversation(self, conversation, seed):
        timestamp = datetime.now().isoformat()
        # Convert the conversation to a JSON string
        conversation_json = json.dumps(
            [
                {"role": message.role, "content": message.content}
                for message in conversation
            ]
        )
        # Write entire conversation as one row
        self.writer.writerow([timestamp, seed, conversation_json])
        self.file.flush()  # Ensure immediate write to disk


async def infinite_generation(csv_filename):
    with CSVWriter(csv_filename) as writer:
        while True:
            conversation, seed = await generate_synthetic_data()
            if conversation:
                writer.write_conversation(conversation, seed)
            await asyncio.sleep(1)  # Prevent overwhelming the API


async def main():
    csv_filename = "smoltalk.csv"
    print("Starting infinite robot fish conversation generation...")
    print("Press Ctrl+C to stop")
    await infinite_generation(csv_filename)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
