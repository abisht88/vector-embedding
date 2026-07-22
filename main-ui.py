import streamlit as st
import time
import json
import uuid
from pathlib import Path
import requests

# PAGE CONFIG
st.set_page_config(
    page_title="spearBot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# PERSISTENCE
HISTORY_FILE = Path(__file__).parent / "chat_history.json"

def load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_history(history: dict):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

def make_title(messages: list) -> str:
    for m in messages:
        if m["role"] == "user":
            return (m["content"][:40] + "…") if len(m["content"]) > 40 else m["content"]
    return "New chat"

def generate_response(user_input: str, thread_id: str = "default_thread") -> str:
    """
    Calls the backend API and returns the assistant's reply as a string.
    Make sure the backend is running at http://localhost:8000
    
    Timeout: 120 seconds (allows time for document retrieval and LLM generation)
    """
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": user_input, "thread_id": thread_id},
            timeout=120,  # Increased from 30 to 120 seconds for LLM processing
        )
        response.raise_for_status()
        data = response.json()
        return data.get("reply", "No response from backend")
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out after 120 seconds. The backend is taking too long to respond. Please try again or check if the backend is overloaded."
    except requests.exceptions.ConnectionError:
        return "⚠️ Backend server is not running. Start it with: python main.py"
    except requests.exceptions.HTTPError as e:
        return f"⚠️ Backend returned an error: {str(e)}"
    except Exception as e:
        return f"⚠️ Error contacting backend: {str(e)}"

# CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(160deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    #MainMenu, header, footer { visibility: hidden; }
    
    /* Chat header */
    .chat-header {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        margin-bottom: 1rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #0d1b3e 0%, #142a5c 55%, #0d1b3e 100%);
        box-shadow: 0 8px 28px rgba(0,0,0,0.45);
        border: 1px solid rgba(0, 212, 255, 0.25);
    }
    .chat-header h1 {
        color: #ffffff;
        background: linear-gradient(90deg, #00d4ff, #ffd166);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(0, 212, 255, 0.25);
    }
    .chat-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        letter-spacing: 0.3px;
    }
    
    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.07) !important;
        border-radius: 16px !important;
        padding: 0.9rem 1.1rem !important;
        margin-bottom: 0.7rem !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessageContent"] * {
        color: #f4f4f5 !important;
    }
    
    /* User message */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #f59e0b, #ea580c) !important;
        margin-left: 10%;
    }
    
    /* Assistant message */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(45, 212, 191, 0.08) !important;
        border: 1px solid rgba(45, 212, 191, 0.2) !important;
        margin-right: 10%;
    }
    
    /* Chat input */
    [data-testid="stBottom"] { background: transparent !important; }
    [data-testid="stBottom"] > div { background: linear-gradient(180deg, rgba(15,12,41,0) 0%, #16213e 45%) !important; }
    [data-testid="stBottomBlockContainer"] { background: transparent !important; }
    
    .stChatInput { background: transparent !important; }
    .stChatInput textarea, .stChatInput > div {
        border-radius: 14px !important;
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(45, 212, 191, 0.3) !important;
    }
    .stChatInput textarea { color: white !important; }
    .stChatInput textarea::placeholder { color: rgba(255,255,255,0.5) !important; }
    .stChatInput button {
        background: linear-gradient(135deg, #ec4899, #8b5cf6) !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(56, 189, 248, 0.08) !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
        color: #e2e8f0 !important;
    }
    div[data-baseweb="popover"] li { background: #1e293b !important; color: #e2e8f0 !important; }
    div[data-baseweb="popover"] li:hover { background: rgba(56, 189, 248, 0.2) !important; }
    section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #e2e8f0 !important; }
    
    /* Code blocks */
    code {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #7dd3fc !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 5px;
        padding: 0.1rem 0.4rem;
    }
    pre, pre code {
        background: rgba(0,0,0,0.3) !important;
        color: #e4e4e7 !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        color: white;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(6, 182, 212, 0.35);
        color: white;
    }
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        text-align: left !important;
        justify-content: flex-start !important;
        background: transparent !important;
        color: #cbd5e1 !important;
        font-weight: 400 !important;
        box-shadow: none !important;
        padding: 0.4rem 0.6rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        transform: none;
        box-shadow: none;
    }
    
    /* Primary sidebar button (New chat) */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) > button {
        background: linear-gradient(135deg, #4f46e5, #06b6d4) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 0.6rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) > button:hover {
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4) !important;
    }
    
    /* Delete button */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(2) > button {
        background: linear-gradient(135deg, #f43f5e, #b91c1c) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 0.6rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(2) > button:hover {
        box-shadow: 0 6px 16px rgba(244, 63, 94, 0.4) !important;
    }
    
    /* Conversation item */
    .conv-item {
        padding: 0.5rem 0.7rem;
        border-radius: 8px;
        margin-bottom: 0.25rem;
        font-size: 0.88rem;
        color: #cbd5e1;
        cursor: pointer;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .conv-item.active {
        background: rgba(250, 204, 21, 0.15);
        color: #facc15 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(127, 90, 240, 0.5); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# SESSION STATE & HISTORY
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "current_id" not in st.session_state:
    if st.session_state.history:
        st.session_state.current_id = list(st.session_state.history.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.history[new_id] = {"title": "New chat", "messages": []}
        st.session_state.current_id = new_id

# SIDEBAR
with st.sidebar:
    st.markdown("### 💡 About")
    st.caption("An AI-powered assistant that enables employees to instantly search, understand, and navigate company policies and knowledge resources.")
    st.markdown("---")

    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.history[new_id] = {"title": "New chat", "messages": []}
        st.session_state.current_id = new_id
        save_history(st.session_state.history)
        st.rerun()

    if st.button("🗑️ Delete this chat", use_container_width=True):
        del st.session_state.history[st.session_state.current_id]
        if not st.session_state.history:
            new_id = str(uuid.uuid4())
            st.session_state.history[new_id] = {"title": "New chat", "messages": []}
            st.session_state.current_id = new_id
        else:
            st.session_state.current_id = list(st.session_state.history.keys())[-1]
        save_history(st.session_state.history)
        st.rerun()

    st.markdown("---")
    st.markdown("**Recent chats**")

    for conv_id in reversed(list(st.session_state.history.keys())):
        conv = st.session_state.history[conv_id]
        label = ("📍 " if conv_id == st.session_state.current_id else "") + conv["title"]
        if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
            st.session_state.current_id = conv_id
            st.rerun()

messages = st.session_state.history[st.session_state.current_id]["messages"]

# HEADER
st.markdown("""
<div class="chat-header">
    <h1>SPEARBot</h1>
    <p>Ask me anything — I'm here to help.</p>
</div>
""", unsafe_allow_html=True)

# WELCOME SECTION (only when conversation is empty)
if not messages:
    st.markdown("<div style='text-align:center; color:#a1a1aa; margin-bottom:1rem;'>Try asking:</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    suggestions = ["⚖️ Policy information", "🏢 Company details", "📄 Documentation"]
    for col, s in zip(cols, suggestions):
        with col:
            if st.button(s, use_container_width=True):
                messages.append({"role": "user", "content": s.split(" ", 1)[1]})
                st.rerun()

# RENDER CHAT HISTORY
for msg in messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# GENERATE RESPONSE IF LAST MESSAGE IS FROM USER
if messages and messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        
        # Show animated loading dots while waiting for response
        loading_dots = ["⏳ Processing.", "⏳ Processing..", "⏳ Processing..."]
        for i in range(15):  # Show dots for up to 15 cycles (30 seconds)
            placeholder.markdown(loading_dots[i % 3])
            time.sleep(0.2)
        
        # Get response from backend
        full_text = generate_response(messages[-1]["content"], st.session_state.current_id)
        
        # Replace loading indicator with actual response
        placeholder.empty()
        
        # Stream response with animation
        streamed = ""
        for word in full_text.split(" "):
            streamed += word + " "
            placeholder.markdown(streamed + "▌")
            time.sleep(0.02)
        placeholder.markdown(full_text)
    
    messages.append({"role": "assistant", "content": full_text})
    st.session_state.history[st.session_state.current_id]["title"] = make_title(messages)
    save_history(st.session_state.history)

# CHAT INPUT
if prompt := st.chat_input("Type your message..."):
    messages.append({"role": "user", "content": prompt})
    save_history(st.session_state.history)
    st.rerun()
