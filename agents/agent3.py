from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds two numbers together."""
    return a + b

@tool
def substract(a: int, b: int) -> int:
    """This is a substraction function that substracts two numbers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """This is a multiplication function that multiplies two numbers."""
    return a * b

tools = [add, substract, multiply]

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> bool:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return False
    return True

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        True: "tools",
        False: END
    }
)
graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 34 plus 21. Then use the result and substract 5 from it. The result of that multiply by 4." )]}
print_stream(app.stream(inputs, stream_mode="values"))