import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from operator import add
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()

PDF_DIRECTORY = "./research_papers/"
PERSIST_DIRECTORY = "./RAGfiles_literature/"
COLLECTION_NAME = "literature_review"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def extract_paper_metadata(document):
    """Extract paper title and first author from filename format: Author-Title.pdf"""
    metadata = document.metadata
    source_file = os.path.basename(metadata.get('source', ''))
    
    # Remove .pdf extension
    filename = source_file.replace('.pdf', '')
    
    # Split by first hyphen to get author and title
    if '-' in filename:
        parts = filename.split('-', 1)  # Split only on first hyphen
        first_author = parts[0].strip()
        title = parts[1].strip().replace('_', ' ')  # Replace underscores with spaces
    else:
        first_author = 'Unknown Author'
        title = filename.replace('_', ' ')
    
    return {
        'title': title,
        'first_author': first_author,
        'source_file': source_file
    }

def load_and_process_pdfs():
    """Load PDFs and create/update vector database if needed."""

    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"Created directory: {PDF_DIRECTORY}")
        print("Please add PDF files to this directory and restart.")
        return None

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIRECTORY}")
        return None
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")

    db_exists = os.path.exists(PERSIST_DIRECTORY)
    needs_rebuild = False

    if db_exists:
        metadata_file = os.path.join(PERSIST_DIRECTORY, "metadata.txt")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                stored_files = set(f.read().splitlines())
            current_files = set(pdf_files)
            
            if stored_files != current_files:
                print("New PDFs detected. Rebuilding database...")
                needs_rebuild = True
            else:
                print("Database is up to date. Loading existing database...")
        else:
            needs_rebuild = True
    else:
        needs_rebuild = True
        print("No existing database found. Creating new database...")

    if needs_rebuild:
        loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDFs")
        
        # Extract and add metadata to documents
        for doc in documents:
            paper_metadata = extract_paper_metadata(doc)
            doc.metadata.update(paper_metadata)
            print(f"  Processed: {paper_metadata['first_author']} - {paper_metadata['title']}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} text chunks")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        
        metadata_file = os.path.join(PERSIST_DIRECTORY, "metadata.txt")
        with open(metadata_file, 'w') as f:
            f.write('\n'.join(pdf_files))
        
        print("Vector database created successfully!")
    else:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print("Loaded existing vector database!")
    
    return vectorstore

vectorstore = load_and_process_pdfs()

if vectorstore is None:
    print("Cannot proceed without PDFs. Exiting...")
    exit(1)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVAL_K}
)

@tool
def search_literature(query: str) -> str:
    """
    Search through research papers to find relevant information.
    Returns content with citations indicating which paper each piece of information comes from.
    Use this tool to answer questions about research papers, find specific information, 
    compare findings, or gather information for literature reviews.
    """
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant information found in the research papers."
    
    # Format results with citations
    results = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get('title', 'Unknown Title')
        author = doc.metadata.get('first_author', 'Unknown Author')
        
        citation = f"[{author}, '{title}']"
        content = doc.page_content.strip()
        
        results.append(f"Source {i} {citation}:\n{content}\n")
    
    return "\n---\n".join(results)

tools = [search_literature]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# System prompt that enforces citation
SYSTEM_PROMPT = """You are a literature review assistant specialized in analyzing research papers.

Your responsibilities:
1. Answer questions about research papers accurately and comprehensively
2. ALWAYS cite your sources using the format [Author, 'Title'] after each statement
3. When comparing papers, clearly indicate which paper each finding comes from
4. Summarize papers with proper citations
5. Generate literature reviews with consistent citation format

Citation Rules:
- Every factual statement must be followed by a citation
- Use the exact citation format provided by the search_literature tool
- When multiple sources support a statement, list all citations
- Be precise about which paper each piece of information comes from

Maintain professional academic tone and provide detailed, well-structured responses."""

def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with tools and system prompt."""
    messages = state['messages']
    
    # Add system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should use tools or end the conversation."""
    last_message = state['messages'][-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise end
    return "end"

def call_tools(state: AgentState) -> AgentState:
    """Execute tool calls and return results."""
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls
    
    # Create a dictionary of tools for easy lookup
    tools_dict = {tool.name: tool for tool in tools}
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        if tool_name not in tools_dict:
            result = f"Error: Tool {tool_name} not found"
        else:
            tool = tools_dict[tool_name]
            result = tool.invoke(tool_args)
        
        # Create tool message with result
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call['id'],
            name=tool_name
        )
        results.append(tool_message)
    
    return {'messages': results}

graph = StateGraph(AgentState)

graph.add_node("agent", call_llm)
graph.add_node("tools", call_tools)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

graph.set_entry_point("agent")

literature_agent = graph.compile()

def run_agent():
    """Run the literature review agent with multi-turn conversation."""
    print("\n=== LITERATURE REVIEW AGENT ===")
    print("Ask questions about your research papers. Type 'exit' or 'quit' to end.\n")
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Invoke agent with full conversation history
        result = literature_agent.invoke({"messages": conversation_history})
        
        # Extract the final response
        final_message = result['messages'][-1]
        
        # Handle response content
        if isinstance(final_message.content, list):
            response_text = final_message.content[0].get("text", str(final_message.content))
        else:
            response_text = final_message.content
        
        print(f"\nAgent: {response_text}\n")
        
        # Update conversation history with all messages from this turn
        conversation_history = result['messages']


if __name__ == "__main__":
    run_agent()