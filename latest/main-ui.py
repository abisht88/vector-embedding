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


def upload_document(file_bytes: bytes, filename: str, thread_id: str) -> dict:
    """Send an uploaded file to the backend so it gets embedded and can be
    used to answer questions in this conversation."""
    try:
        response = requests.post(
            "http://localhost:8000/upload",
            files={"file": (filename, file_bytes)},
            data={"thread_id": thread_id},
            timeout=60,
        )
        response.raise_for_status()
        return {"ok": True, **response.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Backend server is not running. Start it with: python main.py"}
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return {"ok": False, "error": detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def diagnose_render_logs(limit: int = 100, max_errors: int = 5) -> dict:
    """Ask the backend to pull recent logs from the deployed dummy service
    and diagnose the errors using the troubleshooting knowledge base."""
    try:
        response = requests.get(
            "http://localhost:8000/diagnose-logs",
            params={"limit": limit, "max_errors": max_errors},
            timeout=90,
        )
        response.raise_for_status()
        return {"ok": True, **response.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Backend server is not running. Start it with: python main.py"}
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return {"ok": False, "error": detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}

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

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# THEME TOKENS
THEMES = {
    "dark": dict(
        app_bg="linear-gradient(160deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
        text="#f4f4f5",
        banner_bg="linear-gradient(120deg, #0d1b3e 0%, #142a5c 55%, #0d1b3e 100%)",
        banner_border="rgba(0, 212, 255, 0.25)",
        banner_shadow="rgba(0,0,0,0.45)",
        title_grad="linear-gradient(90deg, #00d4ff, #ffd166)",
        title_shadow="rgba(0, 212, 255, 0.25)",
        subtitle="rgba(255,255,255,0.85)",
        bubble_bg="rgba(255,255,255,0.07)",
        bubble_border="rgba(255,255,255,0.1)",
        bubble_shadow="rgba(0,0,0,0.25)",
        user_bubble="linear-gradient(135deg, #f59e0b, #ea580c)",
        assistant_bubble="rgba(45, 212, 191, 0.08)",
        assistant_border="rgba(45, 212, 191, 0.2)",
        bottom_fade="linear-gradient(180deg, rgba(15,12,41,0) 0%, #16213e 45%)",
        input_bg="rgba(255,255,255,0.07)",
        input_border="rgba(45, 212, 191, 0.3)",
        input_text="#ffffff",
        placeholder="rgba(255,255,255,0.5)",
        send_bg="linear-gradient(135deg, #ec4899, #8b5cf6)",
        send_text="#ffffff",
        chip_bg="rgba(255,255,255,0.07)",
        chip_border="rgba(45, 212, 191, 0.3)",
        chip_hover="rgba(45, 212, 191, 0.22)",
        icon_color="#e2e8f0",
        attach_chip_bg="rgba(139, 92, 246, 0.15)",
        attach_chip_border="rgba(139, 92, 246, 0.3)",
        sidebar_bg="linear-gradient(180deg, #0f172a 0%, #1e293b 100%)",
        sidebar_border="rgba(56, 189, 248, 0.15)",
        sidebar_text="#e2e8f0",
        sidebar_hover="rgba(56, 189, 248, 0.12)",
        select_bg="rgba(56, 189, 248, 0.08)",
        select_border="rgba(56, 189, 248, 0.25)",
        popover_bg="#1e293b",
        popover_hover="rgba(56, 189, 248, 0.2)",
        code_bg="rgba(56, 189, 248, 0.15)",
        code_text="#7dd3fc",
        code_border="rgba(56, 189, 248, 0.3)",
        pre_bg="rgba(0,0,0,0.3)",
        btn_bg="linear-gradient(135deg, #2563eb, #06b6d4)",
        btn_text="#ffffff",
        primary_btn_bg="linear-gradient(135deg, #4f46e5, #06b6d4)",
        primary_btn_shadow="rgba(79, 70, 229, 0.4)",
        danger_btn_bg="linear-gradient(135deg, #f43f5e, #b91c1c)",
        danger_btn_shadow="rgba(244, 63, 94, 0.4)",
        scrollbar="rgba(127, 90, 240, 0.5)",
    ),
    "light": dict(
        app_bg="linear-gradient(160deg, #f8fafc 0%, #eef2f7 50%, #f8fafc 100%)",
        text="#0f172a",
        banner_bg="linear-gradient(120deg, #ffffff 0%, #eef4ff 55%, #ffffff 100%)",
        banner_border="rgba(37, 99, 235, 0.15)",
        banner_shadow="rgba(15,23,42,0.08)",
        title_grad="linear-gradient(90deg, #0891b2, #d97706)",
        title_shadow="rgba(8, 145, 178, 0.15)",
        subtitle="rgba(15,23,42,0.7)",
        bubble_bg="rgba(255,255,255,0.92)",
        bubble_border="rgba(15,23,42,0.08)",
        bubble_shadow="rgba(15,23,42,0.05)",
        user_bubble="linear-gradient(135deg, #fed7aa, #fdba74)",
        assistant_bubble="rgba(45, 212, 191, 0.1)",
        assistant_border="rgba(20, 184, 166, 0.25)",
        bottom_fade="linear-gradient(180deg, rgba(248,250,252,0) 0%, #f1f5f9 45%)",
        input_bg="#ffffff",
        input_border="rgba(20, 184, 166, 0.35)",
        input_text="#0f172a",
        placeholder="rgba(15,23,42,0.4)",
        send_bg="linear-gradient(135deg, #ec4899, #8b5cf6)",
        send_text="#ffffff",
        chip_bg="rgba(15,23,42,0.05)",
        chip_border="rgba(15,23,42,0.08)",
        chip_hover="rgba(20, 184, 166, 0.16)",
        icon_color="#0f172a",
        attach_chip_bg="rgba(139, 92, 246, 0.1)",
        attach_chip_border="rgba(139, 92, 246, 0.25)",
        sidebar_bg="linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%)",
        sidebar_border="rgba(15,23,42,0.06)",
        sidebar_text="#0f172a",
        sidebar_hover="rgba(56, 189, 248, 0.12)",
        select_bg="rgba(56, 189, 248, 0.08)",
        select_border="rgba(56, 189, 248, 0.3)",
        popover_bg="#ffffff",
        popover_hover="rgba(56, 189, 248, 0.15)",
        code_bg="rgba(15,23,42,0.06)",
        code_text="#0e7490",
        code_border="rgba(15,23,42,0.1)",
        pre_bg="rgba(15,23,42,0.04)",
        btn_bg="linear-gradient(135deg, #2563eb, #06b6d4)",
        btn_text="#ffffff",
        primary_btn_bg="linear-gradient(135deg, #4f46e5, #06b6d4)",
        primary_btn_shadow="rgba(79, 70, 229, 0.25)",
        danger_btn_bg="linear-gradient(135deg, #f43f5e, #b91c1c)",
        danger_btn_shadow="rgba(244, 63, 94, 0.25)",
        scrollbar="rgba(100, 116, 139, 0.4)",
    ),
}


def build_css(t: dict) -> str:
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {t['app_bg']}; }}
    #MainMenu, header, footer {{ visibility: hidden; }}

    /* Chat header */
    .chat-header {{
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        margin-bottom: 1rem;
        border-radius: 18px;
        background: {t['banner_bg']};
        box-shadow: 0 8px 28px {t['banner_shadow']};
        border: 1px solid {t['banner_border']};
    }}
    .chat-header h1 {{
        background: {t['title_grad']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px {t['title_shadow']};
    }}
    .chat-header p {{
        color: {t['subtitle']};
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        letter-spacing: 0.3px;
    }}

    /* Chat messages */
    div[data-testid="stChatMessage"] {{
        background: {t['bubble_bg']} !important;
        border-radius: 16px !important;
        padding: 0.9rem 1.1rem !important;
        margin-bottom: 0.7rem !important;
        border: 1px solid {t['bubble_border']} !important;
        box-shadow: 0 4px 14px {t['bubble_shadow']};
    }}
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessageContent"] * {{
        color: {t['text']} !important;
    }}

    /* User message */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {{
        background: {t['user_bubble']} !important;
        margin-left: 10%;
    }}

    /* Assistant message */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {{
        background: {t['assistant_bubble']} !important;
        border: 1px solid {t['assistant_border']} !important;
        margin-right: 10%;
    }}

    /* Chat input */
    [data-testid="stBottom"] {{ background: transparent !important; }}
    [data-testid="stBottom"] > div {{ background: {t['bottom_fade']} !important; }}
    [data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}

    .stChatInput {{ background: transparent !important; }}
    /* Style the outer pill only — not the textarea — so we never get a
       box-inside-a-box look, and keep every row vertically centered. */
    .stChatInput > div {{
        border-radius: 14px !important;
        background: {t['input_bg']} !important;
        border: 1px solid {t['input_border']} !important;
    }}
    .stChatInput > div > div {{ align-items: center !important; }}
    /* Streamlit renders its own light-gray textarea wrapper a couple of
       levels above the <textarea> regardless of our theme; neutralize it so
       white (dark-mode) text isn't sitting on a light background. */
    .stChatInput div:has(> textarea),
    .stChatInput div:has(> div > textarea) {{
        background: transparent !important;
    }}
    .stChatInput textarea {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: {t['input_text']} !important;
    }}
    .stChatInput textarea::placeholder {{ color: {t['placeholder']} !important; }}
    .stChatInput button[data-testid="stChatInputSubmitButton"] {{
        background: {t['send_bg']} !important;
        border-radius: 10px !important;
        color: {t['send_text']} !important;
        border: none !important;
    }}

    /* Attach (paperclip) button — style only the real inner button so the
       outer presentation wrapper stays invisible (avoids a double-icon look). */
    [data-testid="stChatInputFileUploadButton"] {{
        background: transparent !important;
        border: none !important;
    }}
    [data-testid="stChatInputFileUploadButton"] button {{
        background: {t['chip_bg']} !important;
        border: 1px solid {t['chip_border']} !important;
        border-radius: 10px !important;
        transition: background 0.12s ease, transform 0.12s ease;
    }}
    [data-testid="stChatInputFileUploadButton"] button:hover {{
        background: {t['chip_hover']} !important;
        transform: translateY(-1px);
    }}
    [data-testid="stChatInputFileUploadButton"] svg {{ fill: {t['icon_color']} !important; }}
    [data-testid^="stChatInputFile"]:not([data-testid="stChatInputFileUploadButton"]) {{
        background: {t['attach_chip_bg']} !important;
        border: 1px solid {t['attach_chip_border']} !important;
        border-radius: 10px !important;
        color: {t['text']} !important;
    }}
    [data-testid^="stChatInputFile"]:not([data-testid="stChatInputFileUploadButton"]) * {{ color: {t['text']} !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {t['sidebar_bg']};
        border-right: 1px solid {t['sidebar_border']};
    }}
    section[data-testid="stSidebar"] * {{ color: {t['sidebar_text']} !important; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: {t['select_bg']} !important;
        border-color: {t['select_border']} !important;
        color: {t['sidebar_text']} !important;
    }}
    div[data-baseweb="popover"] li {{ background: {t['popover_bg']} !important; color: {t['sidebar_text']} !important; }}
    div[data-baseweb="popover"] li:hover {{ background: {t['popover_hover']} !important; }}
    section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ color: {t['sidebar_text']} !important; }}

    /* Code blocks */
    code {{
        background: {t['code_bg']} !important;
        color: {t['code_text']} !important;
        border: 1px solid {t['code_border']};
        border-radius: 5px;
        padding: 0.1rem 0.4rem;
    }}
    pre, pre code {{
        background: {t['pre_bg']} !important;
        color: {t['text']} !important;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 10px;
        border: none;
        background: {t['btn_bg']};
        color: {t['btn_text']};
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(6, 182, 212, 0.35);
        color: {t['btn_text']};
    }}

    /* Sidebar buttons */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
        text-align: left !important;
        justify-content: flex-start !important;
        background: transparent !important;
        font-weight: 400 !important;
        box-shadow: none !important;
        padding: 0.4rem 0.6rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
        border-radius: 8px !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
        background: {t['sidebar_hover']} !important;
        transform: none;
        box-shadow: none;
    }}

    /* Theme toggle row (buttons 1 & 2) */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) > button,
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(2) > button {{
        text-align: center !important;
        justify-content: center !important;
        padding: 0.5rem !important;
        font-weight: 600 !important;
        background: {t['select_bg']} !important;
        border: 1px solid {t['select_border']} !important;
    }}

    /* Primary sidebar button (New chat) */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(3) > button {{
        background: {t['primary_btn_bg']} !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 0.6rem !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(3) > button:hover {{
        box-shadow: 0 6px 16px {t['primary_btn_shadow']} !important;
    }}

    /* Delete button */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(4) > button {{
        background: {t['danger_btn_bg']} !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 0.6rem !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(4) > button:hover {{
        box-shadow: 0 6px 16px {t['danger_btn_shadow']} !important;
    }}

    /* Diagnose logs button */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(5) > button {{
        background: linear-gradient(135deg, #7c3aed, #4338ca) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 0.6rem !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(5) > button:hover {{
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.4) !important;
    }}

    /* Conversation item */
    .conv-item {{
        padding: 0.5rem 0.7rem;
        border-radius: 8px;
        margin-bottom: 0.25rem;
        font-size: 0.88rem;
        cursor: pointer;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .conv-item.active {{
        background: rgba(250, 204, 21, 0.15);
        color: #facc15 !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {t['scrollbar']}; border-radius: 8px; }}
</style>
"""


st.markdown(build_css(THEMES[st.session_state.theme]), unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("### 💡 About")
    st.caption("An AI-powered assistant that enables employees to instantly search, understand, and navigate company policies and knowledge resources.")
    st.markdown("---")

    st.markdown("**Theme**")
    tcols = st.columns(2)
    if tcols[0].button("☀️ Light", use_container_width=True):
        st.session_state.theme = "light"
        st.rerun()
    if tcols[1].button("🌙 Dark", use_container_width=True):
        st.session_state.theme = "dark"
        st.rerun()

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

    if st.button("🩺 Diagnose Render logs", use_container_width=True):
        with st.spinner("Fetching logs and diagnosing errors…"):
            result = diagnose_render_logs()
        messages = st.session_state.history[st.session_state.current_id]["messages"]
        if not result["ok"]:
            reply = f"⚠️ Could not diagnose logs: {result['error']}"
        elif result["count"] == 0:
            reply = "✅ No errors or warnings found in the recent logs."
        else:
            lines = [f"🩺 **Found {result['count']} issue(s) in the dummy service logs:**\n"]
            for d in result["diagnoses"]:
                first_line = d["message"].splitlines()[0]
                lines.append(f"---\n**Log:** `{first_line}`\n\n**Diagnosis:** {d['diagnosis']}")
            reply = "\n\n".join(lines)
        messages.append({"role": "assistant", "content": reply})
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

# CHAT INPUT (paperclip lets the user attach a document to read on the fly)
chat_value = st.chat_input(
    "Type your message...",
    accept_file=True,
    file_type=["pdf", "docx", "txt", "csv", "xlsx", "md"],
)

if chat_value:
    prompt = chat_value.text or ""
    content = prompt

    if chat_value.files:
        uploaded_file = chat_value.files[0]
        with st.spinner(f"Reading {uploaded_file.name}…"):
            result = upload_document(
                uploaded_file.getvalue(),
                uploaded_file.name,
                st.session_state.current_id,
            )
        if result["ok"]:
            note = f"📎 *Attached document: {uploaded_file.name} ({result['chunks']} chunks indexed)*"
        else:
            note = f"📎 *Attached document: {uploaded_file.name} — ⚠️ {result['error']}*"
        # If the user didn't type a question, default to asking for a summary
        # so the backend gets a real query instead of just the note.
        query_text = prompt or f"Please summarize the uploaded document {uploaded_file.name}."
        content = f"{query_text}\n\n{note}".strip()

    if content:
        messages.append({"role": "user", "content": content})
        save_history(st.session_state.history)
        st.rerun()
