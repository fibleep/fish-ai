from langgraph.graph import StateGraph, END, START
from .state import AgentState
from langchain_groq import ChatGroq
from .nodes import start_node
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# llm = ChatGroq(
#     model="llama3-groq-70b-8192-tool-use-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# ) 
def create_graph():
    """Create a graph from the input file."""
    graph = StateGraph(AgentState)
    graph.add_node("start_node", start_node)
    
    graph.add_edge(START, "start_node")
    graph.add_edge("start_node", END)
    
    graph = graph.compile(checkpointer=memory)

    return graph
