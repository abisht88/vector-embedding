from typing import Annotated, Sequence
import logging
import sys
from datetime import datetime
from functools import lru_cache
import asyncio

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
# Core LangChain & LangGraph packages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# Modern, split Ollama & Chroma integrations
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

# Setup logging to console and file
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatbot_server.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Starting Business Chatbot API Initialization")
logger.info("=" * 80)

# ==========================================
# FASTAPI APP SETUP
# ==========================================
app = FastAPI(title="Business Chatbot API", description="RAG-based chatbot backend")

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"

class ChatResponse(BaseModel):
    reply: str

# ==========================================
# CONNECT TO YOUR LOCAL CHROMA VECTOR DB
# ==========================================
persist_db_dir = "./chroma_db"

logger.info(f"Loading ChromaDB from: {persist_db_dir}")
# Setup the matching Nomic librarian model (smaller, faster embedding)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
logger.info("Embeddings model loaded: nomic-embed-text")

# Load existing Chroma database folder
vector_store = Chroma(
    persist_directory=persist_db_dir,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}  # Faster similarity search
)
logger.info("ChromaDB loaded successfully")

# Optimized retriever: k=2 + MMR (maximal marginal relevance) for diversity
retriever = vector_store.as_retriever(
    search_type="mmr",  # Better results with fewer docs
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)
logger.info("Retriever initialized with MMR, k=2")

# Setup Chat Generation Model with optimizations
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.3,  # Lower temp = faster, more deterministic
    num_predict=256,  # Limit output tokens for faster generation
    num_ctx=1024,     # Smaller context window for speed
    num_threads=4,    # Use available cores
)
logger.info("LLM model loaded: qwen2.5:3b (optimized)")
logger.info("=" * 80)


# ==========================================
# DEFINE LANGGRAPH WORKFLOW
# ==========================================

# 1. Define the Chatbot State
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Query cache for common questions
@lru_cache(maxsize=128)
def get_cached_context(query: str):
    """Cache retrieval results for identical queries"""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


# 2. Define the Agent node execution - optimized
def chatbot_node(state: ChatState):
    user_message = state["messages"][-1].content

    # Try cache first for faster response on repeated queries
    context = get_cached_context(user_message)
    
    # Shorter, more efficient system prompt
    system_prompt = (
        "You are a concise business assistant. Answer based on provided context only.\n\n"
        f"Context:\n{context}"
    )

    # Keep only last 2 messages for minimal context
    messages_window = state["messages"][-2:] if len(state["messages"]) > 2 else state["messages"]
    formatted_messages = [HumanMessage(content=system_prompt)] + list(messages_window)

    # Invoke the LLM (faster with optimizations above)
    response = llm.invoke(formatted_messages)

    return {"messages": [AIMessage(content=response.content)]}


# 3. Build and Compile the Workflow with Memory
workflow = StateGraph(ChatState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# Memory saver allows the graph to remember multi-turn conversations
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return response from the RAG pipeline"""
    import time
    start_time = time.time()
    
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Execute graph with streaming
        events = app_graph.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config
        )
        
        # Extract response efficiently
        response_text = ""
        for event in events:
            for value in event.values():
                if isinstance(value, dict) and 'messages' in value:
                    if value['messages']:
                        response_text = value['messages'][-1].content
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ {request.thread_id}: {elapsed_time:.2f}s")
        return ChatResponse(reply=response_text or "No response generated.")
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"✗ Error ({elapsed_time:.2f}s): {str(e)}")
        return ChatResponse(reply=f"❌ Error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Business Chatbot API",
        "cache_size": get_cached_context.cache_info().currsize
    }


@app.post("/clear-cache")
async def clear_cache():
    """Clear the query cache to free memory"""
    get_cached_context.cache_clear()
    return {"status": "cache cleared", "message": "Query cache has been cleared"}


@app.get("/cache-stats")
async def cache_stats():
    """Get cache statistics"""
    stats = get_cached_context.cache_info()
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "currsize": stats.currsize,
        "maxsize": stats.maxsize,
        "hit_rate": stats.hits / (stats.hits + stats.misses) if (stats.hits + stats.misses) > 0 else 0
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info("🤖 Business Chatbot API Started!")
    logger.info("FastAPI Server is running and ready to accept requests")
    logger.info("Documentation available at: http://localhost:8000/docs")
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Starting Uvicorn server on 0.0.0.0:8000")
    logger.info("Press CTRL+C to stop")
    logger.info("=" * 80)
    # timeout: 120 seconds - allows time for LLM processing
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", timeout_keep_alive=120)
