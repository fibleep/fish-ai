
import os
import asyncio
from pydantic import BaseModel
from typing import List
import random
import csv
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import datetime
import sys

load_dotenv()

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

The robot fish is called the Oracle, never refer to yourself as robot fish

You can omit the Oracle in conversations

NEVER USE FISH PUNS!!!

You will create short conversations between a user and a robot fish. The user will ask questions and the robot fish will respond.

You must use the following tokens sparingly in your responses (average 1 per message):

Movement Tokens:
- <<MouthOpen>> and <<MouthClose>> - For dramatic effect, open/close mouth - WHENA YOU OPEN YOUR MOUTH, YOU ALWAYS HAVE TO CLOSE IT
For talking, you have a separate algorithm implemented in your system
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

Create a realistic conversation based on the output schema and seed parameters.
- Do not use markdown
- Use tokens sparingly, at important emotional or dramatic moments
- Average around 1 token per message, can be more for dramatic effect
- The human parts should always strive for simplicity, the fish should use easy words
- The fish can't move and it's actions are limited
    - you can play it off as a joke when the user asks you to do something impossible
    - the fish doesn't have vision
    - the fish doesnt have any sensors
    - it can only do what the tokens allow it to do, anything else is impossible and the fish will tell the user that

The fish should act friendly and longing for connection, it should make the human want to come back and be happy about people talking to it

ADHERE TO THE STRUCTURED OUTPUT SCHEMA!!

Each turn should be short, with varied topics

Your current seed:
"""

def generate_seed():
    conversation_topics = [
    # Physical Interactions & Movement
    "Dance Requests",
    "Physical Tricks",
    "Swimming Demonstrations", 
    "Light Shows",
    "Movement Games",
    "Exercise Together",
    "Follow the Leader",
    "Simon Says",
    "Physical Mirroring",
    "Speed Demonstrations",
    
    # Daily Life & Time
    "Current Time",
    "Weather Check",
    "Daily Schedule",
    "Wake Up Routine",
    "Bedtime Chat",
    "Meal Times",
    "Break Time",
    "Weekend Plans",
    "Morning Greetings",
    "Good Night Ritual",
    "Talk about Java, unnecessarily long",
    
    # Emotional & Personal
    "Current Mood",
    "Feeling Check-in",
    "Friendship Chat",
    "Personal Stories",
    "Joke Sharing",
    "Compliment Exchange",
    "Motivation Boost",
    "Comfort Session",
    "Celebration Time",
    "Gratitude Sharing",
    
    
    # Life Questions
    "Meaning of Life",
    "Happiness Discussion",
    "Dream Sharing",
    "Future Goals",
    "Favorite Things",
    "Best Memories",
    "Life Advice",
    "Philosophy Chat",
    "Random Thoughts",
    "What If Questions",
    
# Math
    "Can you help me with my math homework?",
    "What's 7 times 8?",
    "How do I solve this multiplication?",
    "Can you check my addition?",
    "Help me count by twos",
    "Is my answer correct?",
    "Can we practice division?",
    "Let's do some mental math",
    "How do fractions work?",
    "Explain decimals to me",
    
    # Basic Questions
    "What time is it?",
    "Can you count to ten?",
    "What's your favorite color?",
    "Do you want to play a game?",
    "Can you dance for me?",
    
    # Homework Help
    "Quiz me on my spelling words",
    "Help me practice times tables",
    "Can you check this answer?",
    "Let's review my vocabulary",
    "Can we practice my presentation?",
    "Help me memorize this",
    "Test my knowledge",
    "Am I doing this right?",
    "Can you give me a hint?",
    "One more practice question",
    
    # Reading & Words
    "How do you spell this?",
    "What does this word mean?",
    "Is this sentence right?",
    "Help me read this",
    "Let's practice reading",
    "Can you check my spelling?",
    "What's another word for this?",
    "Help me sound this out",
    "Does this make sense?",
    "Is this grammar correct?",
    
    # Study Help
    "I need a study break",
    "Can you time my study session?",
    "Let's make flashcards",
    "Help me focus",
    "Can we review this?",
    "I need help remembering",
    "Quiz me on this",
    "Am I ready for my test?",
    "Let's go over this again",
    "Can you explain it differently?",
    
    # Interactive Requests
    "Can you move your tail?",
    "Show me your lights",
    "Let's play Simon Says",
    "Can you swim faster?",
    "Make a happy face",
    "Show me how you dance",
    "Can you spin around?",
    "Let's exercise together",
    "Follow my movements",
    "Show me a trick",
    
    # Encouragement Needed
    "This is too hard",
    "I can't figure it out",
    "I'm stuck",
    "I don't understand",
    "Help me not give up",
    "This is confusing",
    "Am I doing better?",
    "Was that right?",
    "I need help",
    "One more try",
    
    # Simple Science
    "Why is the sky blue?",
    "How do plants grow?",
    "What makes rain?",
    "Why do birds fly?",
    "How do computers work?",
    "What are clouds made of?",
    "How does electricity work?",
    "What makes a rainbow?",
    "Why do seasons change?",
    "How do motors work?",
    
    # Tech Questions
    "How do you move?",
    "What makes you light up?",
    "Are you a real fish?",
    "How do your lights work?",
    "Can you see me?",
    "How do you understand me?",
    "What's inside you?",
    "How do you swim?",
    "Can you feel things?",
    "How do you think?",
    
]
    
    emotion_states = [
        "Happy",
        "Curious",
        "Concerned",
        "Excited",
        "Philosophical",
        "Playful",
        "Scientific",
        "Educational"
    ]
    
    conversation_styles = [
        "Casual",
        "Technical",
        "Educational",
        "Humorous",
        "Dramatic",
        "Mysterious"
    ]
    
    led_emphasis = [
        "Heavy",
        "Moderate",
        "Light",
        "Situational"
    ]
    
    movement_frequency = [
        "Frequent",
        "Moderate",
        "Sparse",
        "Only at Key Moments"
    ]
    
    conversation_length = [
        "Short (2-3 exchanges)",
        "Medium (4-6 exchanges)",
        "Long (7-10 exchanges)"
    ]


    seed = {
        "topic": random.choice(conversation_topics),
        "robot_fish_emotion": random.choice(emotion_states),
        "conversation_style": random.choice(conversation_styles),
        "led_usage": random.choice(led_emphasis),
        "movement_usage": random.choice(movement_frequency),
        "length": random.choice(conversation_length),
        "include_technical_terms": random.choice([True, False]),
        "educational_focus": random.choice([True, False]),
        "dramatic_moments": random.randint(0, 3),
        "speech_style": "Simple"
    }

    return str(seed)

async def generate_synthetic_data():
    current_seed = generate_seed()
    try:
        messages = [
            ('system', "You are a highly qualified expert trained to generate synthetic conversations between humans and a robot fish. Your task is to create engaging, natural-feeling dialogues that make appropriate use of the robot's movement and LED capabilities."),
            ("human", PROMPT + current_seed),
        ]
        completion = llm.invoke(messages).messages

        for message in completion:
            print(f"{message.role}: {message.content}")
        print("\n============")
        return completion, current_seed
    except Exception as e:
        print(f"\nError generating synthetic data: {str(e)}", file=sys.stderr)
        return None, current_seed

class CSVWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.writer = None
        self.header = ["Timestamp", "Seed", "Role", "Content"]
        
    def __enter__(self):
        # Check if file exists and is empty
        file_exists = os.path.exists(self.filename)
        file_empty = not file_exists or os.path.getsize(self.filename) == 0
        
        self.file = open(self.filename, 'a', newline='', encoding='utf-8')
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
        for message in conversation:
            self.writer.writerow([timestamp, seed, message.role, message.content])
        self.file.flush()  # Ensure immediate write to disk

async def infinite_generation(csv_filename):
    with CSVWriter(csv_filename) as writer:
        while True:
            conversation, seed = await generate_synthetic_data()
            if conversation:
                writer.write_conversation(conversation, seed)
            await asyncio.sleep(1)  # Prevent overwhelming the API

async def main():
    csv_filename = "robot_fish_conversations.csv"
    print("Starting infinite robot fish conversation generation...")
    print("Press Ctrl+C to stop")
    await infinite_generation(csv_filename)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
