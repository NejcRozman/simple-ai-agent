from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """This function updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document updated with content: {content}"

@tool
def save(filename: str) -> str:
    """This function saves the current document content to a file."""
    
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)
        print(f"Document saved to {filename}")
        return f"Document saved to {filename}"
    except Exception as e:
        return f"Failed to save document: {str(e)}"

tools = [update, save]

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assitant. You are going to help the user update and save a document.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save the document, use the 'save' tool with the desired filename.
    - Make sure to always show the current document state after modifications.

    The current document content is:
    {document_content}

    """)
    
    if not state["messages"]:
        user_input = "I'm ready to start drafting my document."
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + state["messages"] + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"] + [user_message, response])}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)
           
app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()