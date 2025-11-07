import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAgent Response:", response.content)

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":

    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    #print(result['messages'])
    conversation_history = result['messages']
    
    user_input = input("Enter: ")

with open("conversation_log.txt", "w") as f:
    f.write("Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"Agent: {message.content}\n")
    f.write("End of Conversation")

print("Conversation log saved to conversation_log.txt")