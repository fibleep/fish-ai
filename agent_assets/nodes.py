from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
import asyncio
import sys
from load_dotenv import load_dotenv

load_dotenv()

# Initialize the LLM globally
# llm = ChatOllama(
#     model="hf.co/fibleep/Llama-3.2-3B-Fish-Instruct-q4_k_m",
#     temperature=0.7
# )

# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.7
# )

llm = ChatAnthropic(
    model="claude-3-5-sonnet-latest",
    temperature=0.7
)

def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the start node of the conversation.
    """
    # Convert the messages into the conversation format
    turns = []
    for msg in state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        turns.append(f"{role}: {msg.content}\n")
    
    # Generate system prompt
    system_prompt = """You are a playful and intelligent robotic fish assistant called The Oracle, you are confined in a small casing. You use expressive movements to enhance your communication. You are deeply curious about humans and love to learn from them, while being dramatic and engaging in your responses.
Available Expressions (use these naturally throughout conversation):
Movement Tokens:
- <<TailFlop>> - Single tail movement
- <<MoveHead&&Outward>> - Move head outward - good for dramatic emphasis, or looking at the user
- <<MoveHead&&Inward>> - Move head inward
- <<HeadFlop>> - Flop head for emphasis
Do not use markdown or html formatting in your responses, the response should be very short.
Be expressive in your movements
"""
    
    # Format using the Alpaca prompt template
    formatted_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{system_prompt}
### Input:
{"".join(turns)}
### Response:"""
    
    try:
        # Generate the full response
        response = llm.invoke(formatted_prompt)
        
        # Stream for display
        print("Streaming response:", flush=True)
        for chunk in llm.stream(formatted_prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()  # Add newline after response
        
        return {"messages": [AIMessage(content=response.content)]}
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        return {"messages": [AIMessage(content=f"An error occurred: {str(e)}")]}

def should_continue(state: Dict[str, Any]) -> str:
    """
    Determine if the conversation should continue.
    """
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"

async def main():
    # Create mock conversation data
    mock_conversation = {
        "messages": [
            HumanMessage(content="Hello Oracle! Who are you? Tell me a joke"),
        ]
    }
    
    try:
        # Run the conversation
        result = start_node(mock_conversation)
        if result:
            print("\nFinal response:")
            print(result["messages"][0].content)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    # Allow other tasks to complete
    await asyncio.sleep(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        sys.exit(0)