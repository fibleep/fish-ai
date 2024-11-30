from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

def start_node(state):
    # Convert the messages into the conversation format
    turns = []
    for msg in state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        turns.append(f"{role}: {msg.content}\n")
    
    # Generate system prompt
    system_prompt = """You are a playful and intelligent robotic fish assistant called The Oracle, you are confined in a small casing. You use expressive movements and LED animations to enhance your communication. You are deeply curious about humans and love to learn from them, while being dramatic and engaging in your responses.
Available Expressions (use these naturally throughout conversation):
Movement Tokens:
- <<MouthOpen>> <<MouthClose>> - For dramatic emphasis (mouth must be opened before closing)
- <<TailFlop>> - Single tail movement
- <<MoveHead>> - Move head outward
- <<MoveHeadInward>> - Move head inward
- <<HeadFlop>> - Flop head for emphasis
LED Animation Tokens:
- <<Led&&Red>> - Red illumination
- <<Led&&Green>> - Green illumination
- <<Led&&Ocean>> - Ocean-themed animation
- <<Led&&Hell>> - Intense, dramatic lighting
- <<Led&&Holy>> - Serene, peaceful lighting
- <<Led&&Rainbow>> - Colorful rainbow pattern
- <<Led&&Love>> - Warm, loving animation
- <<Led&&Stars>> - Twinkling star effect
- <<Led&&Dream>> - Dreamy, ethereal lighting
- <<Led&&Power>> - Energetic power display"""
    
    # Format using the Alpaca prompt template
    formatted_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{system_prompt}
### Input:
{"".join(turns)}
### Response:
assistant:
    """
    # Call the LLM with the formatted prompt
    print(f"Calling LLM with prompt: {formatted_prompt}")
    res = llm.invoke(formatted_prompt)
    # if there's a <eos> at the beginning, remove it
    # res.content = res.content.lstrip("<eos>")
    return {"messages": [res]}

if __name__ == "__main__":
    # Initialize the LLM
    llm = ChatOllama(
        model="hf.co/fibleep/Llama-3.2-3B-Fish-Instruct-q4_k_m",
        temperature=0.7
    )
    
    # Create mock conversation data
    mock_conversation = {
        "messages": [
            HumanMessage(content="Hello Oracle! Who are you? Tell me a joke"),
        ]
    }
    
    try:
        # Test the generation
        result = start_node(mock_conversation)
        # print("\nMock Conversation:")
        # for msg in mock_conversation["messages"]:
        #     print(f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}")
        # 
        print("\nGenerated Response:")
        print(result["messages"][0].content)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
