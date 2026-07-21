import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict

# Core LangChain & LangGraph packages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Modern, split Ollama & Chroma integrations
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# ==========================================
# CONNECT TO YOUR LOCAL CHROMA VECTOR DB
# ==========================================
persist_db_dir = "./chroma_db"

# Setup the matching Nomic librarian model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load existing Chroma database folder
vector_store = Chroma(
    persist_directory=persist_db_dir,
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Setup Chat Generation Model (ChatOllama)
llm = ChatOllama(model="llama3")


# ==========================================
# DEFINE LANGGRAPH WORKFLOW
# ==========================================

# 1. Define the Chatbot State
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 2. Define the Agent node execution
def chatbot_node(state: ChatState):
    user_message = state["messages"][-1].content

    # Extract structural chunks from PDFs using user's last message
    docs = retriever.invoke(user_message)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Establish conversational system instructions
    system_prompt = (
        "You are a helpful business assistant.\n"
        "Use the following context from our company documents to answer the question.\n"
        "If you do not know the answer, say you do not know. Do not make things up.\n\n"
        "Answers the questions in a readable format.\n"
        f"Context:\n{context}"
    )

    # Prepend the system prompt context to the full chat history
    formatted_messages = [HumanMessage(content=system_prompt)] + list(state["messages"])

    # Invoke the conversation chat layer
    response = llm.invoke(formatted_messages)

    return {"messages": [AIMessage(content=response.content)]}


# 3. Build and Compile the Workflow with Memory
workflow = StateGraph(ChatState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# Memory saver allows the graph to remember multi-turn conversations
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ==========================================
# INTERACTIVE TERMINAL LOOP
# ==========================================

def start_chat():
    # Thread ID separates different user chat sessions
    config = {"configurable": {"thread_id": "business_bot_session_1"}}

    print("\n" + "=" * 50)
    print("🤖 Business Chatbot Initialized successfully!")
    print("Type your questions below. Type 'exit', 'quit', or 'bye' to stop.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            # Check for exit/break triggers
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n🤖 Assistant: Goodbye! Have a great day.")
                break

            if not user_input:
                continue

            # Stream the graph execution
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            # Print the final output from the chatbot node
            for event in events:
                for value in event.values():
                    print(f"Bot: {value['messages'][-1].content}\n")

        except KeyboardInterrupt:
            print("\n🤖 Assistant: Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}\n")


if __name__ == "__main__":
    start_chat()
