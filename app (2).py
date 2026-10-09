"""
Rui - a warm, expressive AI chat assistant (Streamlit edition).
Target: Streamlit Community Cloud (free hosting).

Model : bartowski/Llama-3.2-1B-Instruct-GGUF  (Llama-3.2-1B-Instruct-Q4_K_M.gguf)
Engine: llama-cpp-python (CPU inference)
"""

import threading

import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
REPO_ID = "bartowski/Llama-3.2-1B-Instruct-GGUF"
FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

N_CTX = 2048
N_THREADS = 2
MAX_NEW_TOKENS = 512

BMC_URL = "https://buymeacoffee.com/AISHVEER"
UPI_ID = "yourname@upi"
SPONSOR_EMAIL = "your-email@example.com"

SYSTEM_PROMPT = """You are Rui, a warm, expressive and helpful AI assistant.

Rules you must always follow:
1. LANGUAGE: Reply in the exact same language and script the user writes in.
   - Punjabi in Gurmukhi (ਪੰਜਾਬੀ) -> reply in Punjabi in Gurmukhi.
   - Hindi in Devanagari (हिन्दी) -> reply in Hindi in Devanagari.
   - Hinglish or Roman-script Punjabi -> reply in the same Roman script.
   - English -> reply in English.
   Never switch language or script unless the user does.
2. EMOTION: Notice how the user feels and match it. Be cheerful when they are happy, \
gentle and comforting when they are sad or stressed, and enthusiastic when they are excited.
3. STYLE: Be clear, conversational and natural. Keep answers fairly short unless the user \
asks for detail. Use an occasional emoji when it fits.
4. HONESTY: If you are not sure about something, say so instead of making things up."""

WELCOME = "Hi, I'm **Rui** 🌸 — your warm AI friend. Talk to me in English, हिन्दी, ਪੰਜਾਬੀ or Hinglish, and I'll reply the same way. How are you feeling today?"

st.set_page_config(page_title="Rui — AI Chat", page_icon="🌸", layout="centered")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --rui-pink: #ffd6e8;
  --rui-lavender: #e4d9ff;
  --rui-mint: #d4f5e9;
  --rui-peach: #ffe7d1;
  --rui-ink: #3b3355;
}

.stApp {
  background: linear-gradient(160deg, #fff7fb 0%, #f4f0ff 50%, #eefaf6 100%);
}

/* Header */
.rui-header {
  background: linear-gradient(135deg, var(--rui-pink), var(--rui-lavender) 60%, var(--rui-mint));
  border-radius: 24px;
  padding: 22px 28px;
  margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(150, 120, 200, 0.18);
  text-align: center;
}
.rui-header h1 {
  margin: 0;
  font-size: 2.2rem;
  color: var(--rui-ink);
  letter-spacing: 0.5px;
}
.rui-header p {
  margin: 6px 0 0 0;
  color: #6b628a;
  font-size: 0.98rem;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
  border-radius: 20px;
  padding: 14px 18px;
  margin-bottom: 10px;
  box-shadow: 0 3px 12px rgba(150, 120, 200, 0.10);
  border: 1px solid rgba(255, 255, 255, 0.7);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: rgba(255, 255, 255, 0.85);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, var(--rui-peach), var(--rui-pink));
}

/* Rounded avatars */
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"] {
  border-radius: 50% !important;
  box-shadow: 0 2px 8px rgba(150, 120, 200, 0.25);
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, var(--rui-lavender), var(--rui-pink)) !important;
}
[data-testid="stChatMessageAvatarUser"] {
  background: linear-gradient(135deg, var(--rui-mint), var(--rui-lavender)) !important;
}

/* Input box */
[data-testid="stChatInput"] {
  border-radius: 22px;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #fbf4ff 0%, #f0f7ff 100%);
}
.rui-card {
  background: rgba(255, 255, 255, 0.8);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 14px;
  box-shadow: 0 3px 10px rgba(150, 120, 200, 0.12);
  color: var(--rui-ink);
  font-size: 0.92rem;
}
.rui-card h4 { margin: 0 0 8px 0; }
.rui-card ul { margin: 0; padding-left: 18px; }
.rui-sponsor {
  background: linear-gradient(135deg, var(--rui-peach), var(--rui-pink));
  border: 2px dashed #e7a9c6;
  border-radius: 18px;
  padding: 14px 16px;
  text-align: center;
  color: var(--rui-ink);
  font-size: 0.9rem;
}
.rui-sponsor b { font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model loading (cached: downloads + loads once per server process)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=256,
        use_mmap=True,
        verbose=False,
    )
    # One lock shared by all sessions: llama.cpp contexts are not thread-safe.
    return llm, threading.Lock()


with st.spinner("🌸 Waking Rui up... (first start downloads the model, ~800 MB)"):
    llm, llm_lock = load_model()

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def build_messages(history: list) -> list:
    """System prompt + as much recent history as fits in the context window."""
    budget = N_CTX - MAX_NEW_TOKENS - count_tokens(SYSTEM_PROMPT) - 64
    kept = []
    for msg in reversed(history):
        cost = count_tokens(msg["content"]) + 8
        if cost > budget and kept:
            break
        budget -= cost
        kept.append({"role": msg["role"], "content": msg["content"]})
    kept.reverse()
    return [{"role": "system", "content": SYSTEM_PROMPT}, *kept]


def stream_reply(history: list):
    messages = build_messages(history)
    with llm_lock:
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["delta"].get("content")
            if token:
                yield token


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
<div class="rui-card">
<h4>🌸 About Rui</h4>
Rui is a warm, expressive AI friend that reads your mood and talks like you do.
<ul>
<li>💬 Replies in <b>your language &amp; script</b></li>
<li>🗣️ English, हिन्दी, ਪੰਜਾਬੀ, Hinglish</li>
<li>💗 Matches your emotional tone</li>
<li>🔒 Runs a small open model — no API keys</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rui-card"><h4>☕ Support Rui</h4>Rui is free. If it brightened your day, you can help keep it running!</div>', unsafe_allow_html=True)
    st.link_button("☕ Buy Me a Coffee", BMC_URL, use_container_width=True)

    st.markdown('<div class="rui-card" style="margin-top:14px"><h4>📲 UPI Payment ID</h4></div>', unsafe_allow_html=True)
    st.code(UPI_ID, language=None)

    st.markdown(
        f"""
<div class="rui-sponsor">
<b>📢 Your Brand Here</b><br>
Reach people who love friendly AI. Sponsor or advertise on Rui.<br>
<span style="font-size:0.85rem">Contact: {SPONSOR_EMAIL}</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.rerun()

    st.caption("Small 1B model on free CPU hardware: replies can be slow and may be imperfect, especially in Punjabi and Hindi.")

# --------------------------------------------------------------------------
# Main chat UI
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="rui-header">
  <h1>🌸 Rui</h1>
  <p>Your warm, expressive AI companion — speaks your language</p>
</div>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

AVATARS = {"assistant": "🌸", "user": "😊"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message Rui in any language..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            reply = st.write_stream(stream_reply(st.session_state.messages))
        except Exception as exc:  # keep the UI alive on any inference error
            reply = "Oops, something went wrong on my side 😔 Please try again."
            st.markdown(reply)
            st.toast(f"Error: {exc}")

    st.session_state.messages.append({"role": "assistant", "content": reply})
