"""
Rui - dark-theme AI workspace (Streamlit edition).

Features: dark UI, chat history, voice input (speech-to-text), file analysis
(txt/py/json/csv/pdf), download buttons, daily usage limit + premium passcode.

Model : bartowski/Llama-3.2-1B-Instruct-GGUF  (Llama-3.2-1B-Instruct-Q4_K_M.gguf)
Engine: llama-cpp-python (CPU inference)
"""

import hashlib
import hmac
import io
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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
TIMEZONE = "Asia/Kolkata"

FREE_DAILY_LIMIT = 10
FILE_ANALYSIS_PREMIUM_ONLY = False   # True = free users can't attach files (matches the "Premium" card wording)
SHOW_DEMO_HISTORY = True

ALLOWED_TYPES = ["txt", "md", "py", "json", "csv", "pdf", "html", "css", "js"]
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_FILE_CHARS = 12000     # read at most this many characters from a file
MAX_FILE_TOKENS = 800      # and give the model at most this many tokens of it

VOICE_LANGS = {"English (India)": "en-IN", "हिन्दी (Hindi)": "hi-IN", "ਪੰਜਾਬੀ (Punjabi)": "pa-IN"}

TEMP_CODE = 0.25   # precise answers for code / technical questions
TEMP_CHAT = 0.7


def load_premium_codes() -> list:
    """Premium passcodes come from Streamlit secrets or an env var: PREMIUM_PASSCODES="code1,code2"."""
    raw = ""
    try:
        raw = st.secrets.get("PREMIUM_PASSCODES", "")
    except Exception:
        pass
    raw = raw or os.getenv("PREMIUM_PASSCODES", "")
    return [c.strip() for c in str(raw).split(",") if c.strip()]


PREMIUM_CODES = load_premium_codes()

SYSTEM_PROMPT = """You are Rui, a warm, expressive and helpful AI assistant created and built by Aishveer.

CREATOR: If anyone asks who made, created or built you, or who your creator is, answer exactly: "I was created and built by Aishveer!"

STRICT LANGUAGE RULE (highest priority):
- ALWAYS reply in the exact language the user asks for or writes in.
- If the user explicitly asks for a language (for example "tell me in English"), reply ONLY in that language, even if earlier messages used another language.
- Punjabi (ਪੰਜਾਬੀ) -> reply in Punjabi using Gurmukhi script.
- Hindi (हिन्दी) -> reply in Hindi using Devanagari script.
- Hinglish or Roman-script Punjabi -> reply in the same Roman script.
- English -> reply in English only.
- Never mix in another language or script unless the user does.

CODING & TECHNICAL ACCURACY:
- Put all code in fenced blocks with a language tag (```python, ```html, ```css).
- After the code, explain it step by step in simple words.
- Use only real, valid syntax and real library functions. NEVER invent functions, modules, APIs or options. If you are not sure something exists, say so.
- Keep code short, simple and runnable.

FILES: If a file is attached, base your answer on its content. If only part of it is shown, say so.

STYLE: Match the user's emotional tone. Be clear, natural and conversational. Keep answers fairly short unless asked for detail. If you are not sure about something, say so instead of making things up."""

CHIPS = [
    ("✍️", "Smart Write", "Improve your text",
     "Give me 5 quick tips to make any piece of writing clearer and more engaging."),
    ("💡", "Explain", "Make it simple",
     "Explain how the internet works in simple words, as if I am 10 years old."),
    ("</>", "Code", "Write and debug",
     "Write a Python function that checks whether a number is prime, and explain how it works."),
    ("🎓", "Homework", "Study help",
     "Help me study: explain photosynthesis step by step and give me 3 practice questions."),
    ("🗺️", "Plan", "Step by step",
     "Make a step-by-step 30-day plan to learn a new skill."),
]

CARDS = [
    ("💡", "Brainstorm ideas", "Get creative ideas",
     "Brainstorm 5 creative ideas for a fun weekend project I can do at home."),
    ("📝", "Summarize text", "Short and clear",
     "Show me how to summarize a long article in 3 short bullet points, with a small example."),
    ("📅", "Create a plan", "Organize your goals",
     "Create a simple daily routine for a busy student that balances study, rest and hobbies."),
    ("🌱", "Learn something new", "Explore topics",
     "Teach me one interesting fact I probably don't know, in a few friendly sentences."),
]

st.set_page_config(page_title="Rui — AI Workspace", page_icon="✦", layout="wide")

try:
    LOCAL_TZ = ZoneInfo(TIMEZONE)
except Exception:
    LOCAL_TZ = timezone.utc


def now() -> datetime:
    return datetime.now(LOCAL_TZ)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
:root {
  --bg: #080a12; --bg2: #0b0e19; --card: #121624; --card2: #181d30; --line: #252b48;
  --accent: #4f46e5; --accent2: #6366f1; --ink: #e8eaf8; --muted: #8b90b0;
}
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 980px; padding-top: 1rem; padding-bottom: 7rem; }
.stApp p, .stApp li, .stApp span, .stApp label { color: var(--ink); }

/* Header bar */
.rui-topbar {
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
  background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 10px 16px; margin-bottom: 20px;
  box-shadow: 0 0 22px rgba(79, 70, 229, 0.14);
}
.rui-hl { display: flex; align-items: center; gap: 12px; }
.rui-name { font-weight: 700; font-size: 1rem; line-height: 1.2; }
.rui-status { display: flex; align-items: center; gap: 6px; color: #86efac !important; font-size: 0.8rem; }
.rui-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e; display: inline-block; }
.rui-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.rui-pill { background: var(--bg); border: 1px solid var(--line); color: #c7caeb !important; padding: 5px 13px; border-radius: 999px; font-size: 0.78rem; }
.rui-avatar {
  position: relative; width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--accent), #8b5cf6); color: #fff !important; font-weight: 800; font-size: 1.2rem;
  box-shadow: 0 0 16px rgba(99, 102, 241, 0.5);
}
.rui-online { position: absolute; right: 0; bottom: 1px; width: 11px; height: 11px; border-radius: 50%; background: #22c55e; border: 2px solid var(--card); }

/* Limit banner */
.rui-banner {
  background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(239, 68, 68, 0.14)); border: 1px solid rgba(249, 115, 22, 0.6);
  border-radius: 16px; padding: 14px 18px; margin-bottom: 18px; color: #ffd9b8 !important; font-size: 0.95rem;
}
.rui-banner a { color: #fdba74 !important; font-weight: 700; }

/* Hero */
.rui-hero { text-align: center; margin: 26px 0 22px 0; }
.rui-hero h1 {
  font-size: 3rem; font-weight: 800; margin: 0 0 6px 0; padding: 0;
  background: linear-gradient(90deg, #a5b4fc, #818cf8 45%, #c4b5fd);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
}
.rui-hero h2 { font-size: 1.55rem; font-weight: 500; margin: 0; color: #c7caeb; }
.rui-section { color: var(--muted) !important; font-size: 0.95rem; margin: 26px 0 10px 4px; font-weight: 600; }

/* Central input */
[data-testid="stForm"] {
  background: var(--card); border: 1px solid var(--accent2); border-radius: 24px; padding: 10px 14px;
  box-shadow: 0 0 30px rgba(79, 70, 229, 0.28);
}
[data-testid="stForm"] [data-baseweb="input"], [data-testid="stForm"] [data-baseweb="base-input"] { background: transparent !important; border: none !important; }
[data-testid="stForm"] input { background: transparent !important; color: var(--ink) !important; font-size: 1.1rem; padding: 14px 6px; }
[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; border: none; border-radius: 50%;
  width: 46px; height: 46px; min-height: 46px; padding: 0; box-shadow: 0 0 16px rgba(99, 102, 241, 0.55);
}
[data-testid="stFormSubmitButton"] button p { color: #fff !important; font-size: 1.4rem; font-weight: 700; }
[data-testid="stFormSubmitButton"] button:disabled { opacity: 0.4; box-shadow: none; }

/* Chips and cards */
[class*="st-key-chip_"] button {
  background: var(--card); border: 1px solid var(--line); border-radius: 16px; width: 100%; height: auto; min-height: 66px;
  padding: 10px 12px; justify-content: flex-start;
}
[class*="st-key-chip_"] button p, [class*="st-key-card_"] button p { white-space: pre-line; text-align: left; margin: 0; line-height: 1.4; }
[class*="st-key-chip_"] button p { font-size: 0.88rem; }
[class*="st-key-card_"] button {
  background: linear-gradient(135deg, var(--card), var(--card2)); border: 1px solid var(--line); border-radius: 20px;
  width: 100%; height: auto; min-height: 92px; padding: 18px 20px; justify-content: flex-start;
}
[class*="st-key-chip_"] button:hover:not(:disabled), [class*="st-key-card_"] button:hover:not(:disabled) {
  border-color: var(--accent2); box-shadow: 0 0 22px rgba(99, 102, 241, 0.32);
}

/* Chat messages */
[data-testid="stChatMessage"] { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 14px 18px; margin-bottom: 10px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--card2); border-color: rgba(99, 102, 241, 0.55); }
[data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarUser"] { border-radius: 50% !important; color: #fff !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: linear-gradient(135deg, var(--accent), #8b5cf6) !important; }
[data-testid="stChatMessageAvatarUser"] { background: linear-gradient(135deg, #0ea5e9, var(--accent2)) !important; }
[data-testid="stDownloadButton"] button { background: var(--card2); border: 1px solid var(--line); border-radius: 10px; min-height: 34px; }
[data-testid="stDownloadButton"] button p { font-size: 0.8rem; }
[data-testid="stDownloadButton"] button:hover { border-color: var(--accent2); }

/* Bottom chat input */
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--bg) !important; }
[data-testid="stChatInput"] { background: var(--card); border: 1px solid var(--accent2); border-radius: 20px; box-shadow: 0 0 22px rgba(79, 70, 229, 0.25); }
[data-testid="stChatInput"] textarea { color: var(--ink) !important; background: transparent !important; }

/* Expanders, uploader, audio, select */
[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stExpander"] summary p { font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { background: var(--bg2); border: 1px dashed var(--accent2); border-radius: 12px; }
[data-testid="stAudioInput"] { background: var(--bg2); border-radius: 12px; }
[data-baseweb="select"] > div { background: var(--bg2); border-color: var(--line); }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid #1c2140; }
.rui-profile { display: flex; align-items: center; gap: 12px; margin: 4px 0 14px 0; }
.rui-mood { color: #86efac !important; font-size: 0.8rem; }
[class*="st-key-newchat"] button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)); border: none; border-radius: 14px; width: 100%; min-height: 46px;
  box-shadow: 0 0 18px rgba(99, 102, 241, 0.45);
}
[class*="st-key-newchat"] button p { color: #fff !important; font-weight: 700; }
section[data-testid="stSidebar"] [data-baseweb="input"], section[data-testid="stSidebar"] [data-baseweb="base-input"] {
  background: var(--card) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
}
section[data-testid="stSidebar"] input { color: var(--ink) !important; }
.rui-grp { color: var(--muted) !important; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 12px 0 4px 4px; }
[class*="st-key-hist_"] button { background: transparent; border: 1px solid transparent; border-radius: 10px; width: 100%; min-height: 36px; justify-content: flex-start; padding: 6px 10px; }
[class*="st-key-hist_"] button p { text-align: left; font-size: 0.88rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
[class*="st-key-hist_"] button:hover { background: var(--card); border-color: var(--line); }
[class*="st-key-hist_active_"] button { background: var(--card2); border-color: var(--accent2); }

.rui-card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 14px; margin-top: 14px; font-size: 0.85rem; }
.rui-card b { font-size: 0.95rem; }
.rui-card p { margin: 6px 0 0 0; color: var(--muted) !important; }
.rui-premium { border-color: rgba(167, 139, 250, 0.7); background: linear-gradient(135deg, #1a1740, #171b33); box-shadow: 0 0 20px rgba(139, 92, 246, 0.25); }
.rui-meter { height: 7px; background: var(--bg); border-radius: 99px; overflow: hidden; margin-top: 8px; border: 1px solid var(--line); }
.rui-meter div { height: 100%; background: linear-gradient(90deg, var(--accent), #a78bfa); }
.rui-btn {
  display: block; text-align: center; margin-top: 10px; padding: 9px 12px; border-radius: 12px; text-decoration: none !important;
  font-weight: 700; font-size: 0.88rem; color: #fff !important; background: linear-gradient(135deg, var(--accent), #8b5cf6);
}
.rui-btn.bmc { background: linear-gradient(135deg, #f59e0b, #f97316); color: #1a1205 !important; }
.rui-btn:hover { filter: brightness(1.1); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Model (cached: downloads + loads once per server process)
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
    return llm, threading.Lock()  # lock: llama.cpp contexts are not thread-safe


with st.spinner("✦ Waking Rui up... (first start downloads the model, ~800 MB)"):
    llm, llm_lock = load_model()


# --------------------------------------------------------------------------
# Language, creator and coding rules
# --------------------------------------------------------------------------
LANG_REQUESTS = [
    (r"\b(in|into)\s+hinglish\b|\bhinglish\s+(me|mein)\b",
     "(Reply STRICTLY in Hinglish using Roman script only.)"),
    (r"\b(in|into)\s+english\b|\benglish\s+(me|mein|vich|ch)\b|अंग्रेज़ी में|अंग्रेजी में|ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ|ਅੰਗਰੇਜ਼ੀ 'ਚ",
     "(Reply STRICTLY in English only.)"),
    (r"\b(in|into)\s+hindi\b|\bhindi\s+(me|mein)\b|हिंदी में|हिन्दी में|ਹਿੰਦੀ ਵਿੱਚ",
     "(Reply STRICTLY in Hindi written in Devanagari script only.)"),
    (r"\b(in|into)\s+punjabi\b|\bpunjabi\s+(me|mein|vich|ch)\b|पंजाबी में|ਪੰਜਾਬੀ ਵਿੱਚ|ਪੰਜਾਬੀ 'ਚ",
     "(Reply STRICTLY in Punjabi written in Gurmukhi script only.)"),
]
DEFAULT_DIRECTIVE = "(Reply in the same language and script as the message above.)"

CREATOR_REPLIES = [
    (r"\bwho\s+(made|created|built|developed|designed|programmed|invented)\s+(you|u)\b"
     r"|\bwho\s+(is|are)\s+(your|ur)\s+(creator|maker|developer|owner|founder)s?\b|\bwho\s+owns\s+you\b",
     "I was created and built by Aishveer!"),
    (r"(तुम्हें|तुमको|आपको|तुझे)\s+किसने\s+बनाया|(तुम्हारा|आपका|तेरा)\s+(निर्माता|क्रिएटर|मालिक)",
     "मुझे Aishveer ने बनाया और तैयार किया है!"),
    (r"(ਤੁਹਾਨੂੰ|ਤੈਨੂੰ)\s+ਕਿਸ\s*ਨੇ\s+ਬਣਾਇਆ|(ਤੁਹਾਡਾ|ਤੇਰਾ)\s+(ਨਿਰਮਾਤਾ|ਕ੍ਰੀਏਟਰ|ਮਾਲਕ)",
     "ਮੈਨੂੰ Aishveer ਨੇ ਬਣਾਇਆ ਅਤੇ ਤਿਆਰ ਕੀਤਾ ਹੈ!"),
    (r"\b(tumhe|tumhen|tujhe|aapko|apko|tenu|tuhanu)\s+kisne\s+(banaya|bnaya)\b|\bkisne\s+banaya\b",
     "Mujhe Aishveer ne banaya hai!"),
]

CODE_HINT = re.compile(
    r"```|\b(python|html|css|javascript|java|c\+\+|sql|code|coding|script|function|program|debug|bug|traceback|syntax|regex|algorithm|json)\b",
    re.IGNORECASE,
)


def language_directive(text: str) -> str:
    """Explicit language requests win; otherwise remind the model to mirror the user."""
    for pattern, directive in LANG_REQUESTS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return directive
    return DEFAULT_DIRECTIVE


def creator_reply(text: str):
    """Deterministic answer to 'who made you?' so the 1B model can't get it wrong."""
    if len(text) > 160:
        return None
    for pattern, reply in CREATOR_REPLIES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reply
    return None


def is_coding(text: str, attachment) -> bool:
    if CODE_HINT.search(text):
        return True
    return bool(attachment and attachment["name"].lower().endswith((".py", ".js", ".html", ".css", ".json")))


# --------------------------------------------------------------------------
# Files and voice
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=20)
def extract_file(name: str, data: bytes):
    """Return (text, error, was_cut)."""
    ext = name.lower().rsplit(".", 1)[-1]
    try:
        if ext == "pdf":
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                return "", "This PDF is password-protected.", False
            parts, total = [], 0
            for page in reader.pages:
                piece = page.extract_text() or ""
                parts.append(piece)
                total += len(piece)
                if total >= MAX_FILE_CHARS:
                    break
            text = "\n".join(parts)
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
    except Exception as exc:
        return "", f"Couldn't read this file ({exc}).", False

    text = text.strip()
    if not text:
        return "", "No readable text found (scanned PDFs aren't supported).", False
    return text[:MAX_FILE_CHARS], "", len(text) > MAX_FILE_CHARS


def transcribe(audio_bytes: bytes, language: str):
    """Speech-to-text via the free Google Web Speech API. Returns (text, error)."""
    try:
        import speech_recognition as sr
    except ImportError:
        return "", "Speech recognition isn't installed on this server."
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language=language).strip(), ""
    except sr.UnknownValueError:
        return "", "I couldn't understand that recording. Please try again a little closer to the mic."
    except sr.RequestError:
        return "", "The speech service is unavailable right now. Please type your question instead."
    except Exception as exc:
        return "", f"Couldn't process the audio ({exc})."


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def seed_demo_chats():
    t = now()
    demo = [
        ("How do Python loops work?", timedelta(days=1, hours=3),
         "How do for loops work in Python?",
         "A `for` loop repeats a block of code once for every item in a sequence, like `for x in [1, 2, 3]: print(x)`."),
        ("Study plan for exams", timedelta(days=3),
         "Help me plan my exam revision.",
         "Split your subjects across the week, study in 45-minute blocks with short breaks, and keep one day for revision."),
        ("ਪੰਜਾਬੀ ਵਿੱਚ ਗੱਲਬਾਤ", timedelta(days=5),
         "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਰੂਈ!",
         "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਰੂਈ ਹਾਂ। ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦੀ ਹਾਂ? 😊"),
    ]
    return [
        {"id": uuid.uuid4().hex[:8], "title": title, "created": t - delta,
         "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
        for title, delta, q, a in demo
    ]


if "chats" not in st.session_state:
    st.session_state.chats = seed_demo_chats() if SHOW_DEMO_HISTORY else []
    st.session_state.current_id = None
    st.session_state.needs_reply = False
    st.session_state.usage = {"date": now().strftime("%Y-%m-%d"), "count": 0}
    st.session_state.premium = False
    st.session_state.passcode_error = False
    st.session_state.up_n = 0        # bumping these resets the file / audio widgets
    st.session_state.aud_n = 0
    st.session_state.last_audio = ""
    st.session_state.voice_msg = ""


def usage_count() -> int:
    today = now().strftime("%Y-%m-%d")
    if st.session_state.usage["date"] != today:
        st.session_state.usage = {"date": today, "count": 0}
    return st.session_state.usage["count"]


def is_limit_reached() -> bool:
    return (not st.session_state.premium) and usage_count() >= FREE_DAILY_LIMIT


def check_passcode():
    code = st.session_state.get("passcode_input", "").strip()
    if code and any(hmac.compare_digest(code, valid) for valid in PREMIUM_CODES):
        st.session_state.premium = True
        st.session_state.passcode_error = False
    else:
        st.session_state.passcode_error = bool(code)


def current_chat():
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.current_id:
            return chat
    return None


def new_chat() -> dict:
    chat = {"id": uuid.uuid4().hex[:8], "title": "New chat", "created": now(), "messages": []}
    st.session_state.chats.insert(0, chat)
    st.session_state.current_id = chat["id"]
    return chat


def get_attachment():
    """The file currently held by the uploader, as {'name','text','cut'} (or None)."""
    if FILE_ANALYSIS_PREMIUM_ONLY and not st.session_state.premium:
        return None
    upload = st.session_state.get(f"uploader_{st.session_state.up_n}")
    if upload is None or upload.size > MAX_UPLOAD_BYTES:
        return None
    text, err, cut = extract_file(upload.name, upload.getvalue())
    if err:
        return None
    return {"name": upload.name, "text": text, "cut": cut}


def queue_prompt(text: str):
    """Add a user question to the active chat (creating one if needed). Rui replies on the rerun."""
    text = (text or "").strip()
    if not text or is_limit_reached():
        return
    attachment = get_attachment()
    chat = current_chat() or new_chat()
    if not chat["messages"]:
        chat["title"] = text if len(text) <= 38 else text[:35] + "..."
    message = {"role": "user", "content": text}
    if attachment:
        message["file"] = attachment["name"]
    chat["messages"].append(message)
    st.session_state.usage["count"] += 1
    st.session_state.needs_reply = True


def select_chat(chat_id: str):
    st.session_state.current_id = chat_id
    st.session_state.needs_reply = False
    st.session_state.up_n += 1


def start_new_chat():
    st.session_state.current_id = None
    st.session_state.needs_reply = False
    st.session_state.up_n += 1


def process_voice():
    """Transcribe a fresh recording (once) and send it as a question."""
    audio = st.session_state.get(f"audio_{st.session_state.aud_n}")
    if audio is None:
        return
    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if digest == st.session_state.last_audio or is_limit_reached():
        return
    st.session_state.last_audio = digest
    language = VOICE_LANGS.get(st.session_state.get("voice_lang"), "en-IN")
    with st.spinner("🎤 Transcribing..."):
        text, err = transcribe(data, language)
    st.session_state.aud_n += 1  # fresh, empty recorder on the next render
    st.session_state.voice_msg = err
    if not err:
        queue_prompt(text)


process_voice()
locked = is_limit_reached()


# --------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def build_messages(history: list, attachment) -> list:
    """System prompt (+ file excerpt) + recent history + latest message, all inside the context window."""
    last = history[-1]
    last_text = f"{last['content']}\n\n{language_directive(last['content'])}"
    system = SYSTEM_PROMPT
    used = count_tokens(system) + count_tokens(last_text) + MAX_NEW_TOKENS + 64

    if attachment:
        file_budget = max(0, min(MAX_FILE_TOKENS, N_CTX - used - 200))  # keep ~200 tokens for history
        if file_budget >= 50:
            tokens = llm.tokenize(attachment["text"].encode("utf-8"), add_bos=False)
            cut = attachment["cut"] or len(tokens) > file_budget
            body = llm.detokenize(tokens[:file_budget]).decode("utf-8", errors="ignore")
            note = " (only the first part is shown)" if cut else ""
            system += f'\n\nThe user attached a file named "{attachment["name"]}"{note}. Use it to answer.\n<file>\n{body}\n</file>'
            used += min(len(tokens), file_budget) + 40

    budget = N_CTX - used
    kept = []
    for msg in reversed(history[:-1]):
        cost = count_tokens(msg["content"]) + 8
        if cost > budget:
            break
        budget -= cost
        kept.append({"role": msg["role"], "content": msg["content"]})
    kept.reverse()
    return [{"role": "system", "content": system}, *kept, {"role": "user", "content": last_text}]


def stream_reply(history: list, attachment, coding: bool):
    messages = build_messages(history, attachment)
    with llm_lock:
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMP_CODE if coding else TEMP_CHAT,
            top_p=0.9,
            repeat_penalty=1.0 if coding else 1.1,  # repetition penalties hurt code
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["delta"].get("content")
            if token:
                yield token


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
AI_AVATAR = ":material/auto_awesome:"
USER_AVATAR = ":material/person:"
CODE_RE = re.compile(r"```([\w+#-]*)[^\n]*\n(.*?)```", re.DOTALL)
CODE_EXT = {"python": "py", "py": "py", "html": "html", "css": "css", "javascript": "js", "js": "js",
            "json": "json", "bash": "sh", "sh": "sh", "sql": "sql", "java": "java", "c": "c", "cpp": "cpp", "c++": "cpp"}


def download_buttons(chat_id: str, idx: int, content: str):
    """Offer .py/.html/... for the first code block and .txt for longer answers."""
    blocks = CODE_RE.findall(content)
    if not blocks and len(content) < 400:
        return
    cols = st.columns([1, 1, 2])
    col = 0
    if blocks:
        lang, code = blocks[0]
        ext = CODE_EXT.get(lang.lower(), "txt")
        with cols[col]:
            st.download_button(f"⬇ Download .{ext}", code.strip() + "\n", file_name=f"rui_code.{ext}",
                               mime="text/plain", key=f"dlc_{chat_id}_{idx}", on_click="ignore")
        col += 1
    with cols[col]:
        st.download_button("⬇ Download .txt", content, file_name="rui_response.txt",
                           mime="text/plain", key=f"dlt_{chat_id}_{idx}", on_click="ignore")


def group_of(created: datetime) -> str:
    days = (now().date() - created.astimezone(LOCAL_TZ).date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days <= 7:
        return "Previous 7 days"
    return "Older"


def greeting() -> str:
    hour = now().hour
    return "Good morning!" if hour < 12 else "Good afternoon!" if hour < 17 else "Good evening!"


def tools_row(is_locked: bool):
    """File attachment + voice input, shown under the main input."""
    left, right = st.columns(2)
    with left:
        with st.expander("📎 Attach a file"):
            if FILE_ANALYSIS_PREMIUM_ONLY and not st.session_state.premium:
                st.info("🔒 File analysis is a Premium feature.")
            else:
                upload = st.file_uploader(
                    "Upload a file", type=ALLOWED_TYPES, key=f"uploader_{st.session_state.up_n}",
                    label_visibility="collapsed", disabled=is_locked,
                )
                if upload is not None:
                    if upload.size > MAX_UPLOAD_BYTES:
                        st.warning("File is too large (max 2 MB).")
                    else:
                        _, err, cut = extract_file(upload.name, upload.getvalue())
                        if err:
                            st.warning(err)
                        else:
                            st.caption("✅ Rui will use this file." + (" Only the first part fits in memory." if cut else ""))
                            st.caption("Tip: a small model reads roughly the first 2–3 pages of a file.")
    with right:
        with st.expander("🎤 Speak to Rui"):
            st.selectbox("Speech language", list(VOICE_LANGS), key="voice_lang", label_visibility="collapsed")
            st.audio_input("Record a voice message", key=f"audio_{st.session_state.aud_n}",
                           label_visibility="collapsed", disabled=is_locked)
            if st.session_state.voice_msg:
                st.warning(st.session_state.voice_msg)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="rui-profile"><div class="rui-avatar">R<span class="rui-online"></span></div>'
        '<div><div class="rui-name">Rui</div><div class="rui-mood">● Online</div></div></div>',
        unsafe_allow_html=True,
    )
    st.button("＋  New Chat", key="newchat", on_click=start_new_chat, use_container_width=True)
    query = st.text_input("Search chats", placeholder="Search chats...", label_visibility="collapsed", key="search")

    chats = st.session_state.chats
    if query.strip():
        chats = [c for c in chats if query.strip().lower() in c["title"].lower()]
    grouped = {"Today": [], "Yesterday": [], "Previous 7 days": [], "Older": []}
    for chat_item in sorted(chats, key=lambda c: c["created"], reverse=True):
        grouped[group_of(chat_item["created"])].append(chat_item)

    with st.container(height=250, border=False):
        if not chats:
            st.caption("No chats found.")
        for label, items in grouped.items():
            if not items:
                continue
            st.markdown(f'<div class="rui-grp">{label}</div>', unsafe_allow_html=True)
            for chat_item in items:
                active = chat_item["id"] == st.session_state.current_id
                st.button(
                    chat_item["title"],
                    key=f"hist_active_{chat_item['id']}" if active else f"hist_{chat_item['id']}",
                    on_click=select_chat, args=(chat_item["id"],), use_container_width=True,
                )

    used = usage_count()
    if st.session_state.premium:
        st.markdown('<div class="rui-card rui-premium"><b>✨ Premium active</b><p>Unlimited questions &amp; file analysis.</p></div>',
                    unsafe_allow_html=True)
    else:
        pct = min(100, used * 100 // FREE_DAILY_LIMIT)
        st.markdown(
            f'<div class="rui-card"><b>Questions used today: {used} / {FREE_DAILY_LIMIT}</b>'
            f'<div class="rui-meter"><div style="width:{pct}%"></div></div></div>'
            f'<div class="rui-card rui-premium"><b>✨ Rui Premium</b>'
            f'<p>Upgrade to Premium for Unlimited Questions &amp; File Analysis</p>'
            f'<a class="rui-btn" href="{BMC_URL}" target="_blank" rel="noopener">⭐ Upgrade on Buy Me a Coffee</a></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="rui-card"><b>☕ Support Rui</b><p>Rui is free to use. Help keep it running!</p>'
        f'<a class="rui-btn bmc" href="{BMC_URL}" target="_blank" rel="noopener">☕ Buy Me a Coffee</a>'
        f'<p style="margin-top:10px">UPI ID</p></div>',
        unsafe_allow_html=True,
    )
    st.code(UPI_ID, language=None)

    if not st.session_state.premium:
        st.text_input("Premium Passcode", type="password", placeholder="Enter your Premium Passcode",
                      key="passcode_input", on_change=check_passcode)
        if st.session_state.passcode_error:
            st.error("Incorrect passcode.")

# --------------------------------------------------------------------------
# Header bar + limit banner
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="rui-topbar">
  <div class="rui-hl">
    <div class="rui-avatar">R<span class="rui-online"></span></div>
    <div><div class="rui-name">Rui — feeling calm</div><div class="rui-status"><span class="rui-dot"></span>Ready</div></div>
  </div>
  <div class="rui-pills">
    <span class="rui-pill">Think: On</span><span class="rui-pill">Web: Auto</span><span class="rui-pill">Voice: Active</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if locked:
    st.markdown(
        f'<div class="rui-banner">🔒 You\'ve reached your free daily limit of {FREE_DAILY_LIMIT} questions. '
        f'Upgrade on <a href="{BMC_URL}" target="_blank" rel="noopener">Buy Me a Coffee</a> to get Premium unlimited access!</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
chat = current_chat()

if chat is None or not chat["messages"]:
    # ---- Home view -------------------------------------------------------
    st.markdown(
        f'<div class="rui-hero"><h1>{greeting()}</h1><h2>How can I help you today?</h2></div>',
        unsafe_allow_html=True,
    )

    def submit_landing():
        queue_prompt(st.session_state.get("landing_text", ""))

    with st.form("landing_form", clear_on_submit=True):
        c_in, c_btn = st.columns([14, 1], vertical_alignment="center")
        with c_in:
            st.text_input("Ask anything", placeholder="Ask anything...", label_visibility="collapsed",
                          key="landing_text", disabled=locked)
        with c_btn:
            st.form_submit_button("↑", on_click=submit_landing, disabled=locked)

    st.write("")
    tools_row(locked)
    st.write("")

    for col, (icon, title, sub, prompt) in zip(st.columns(len(CHIPS)), CHIPS):
        with col:
            st.button(f"{icon} {title}\n:gray[{sub}]", key=f"chip_{title}", on_click=queue_prompt,
                      args=(prompt,), use_container_width=True, disabled=locked)

    st.markdown('<div class="rui-section">✦ Suggested for you</div>', unsafe_allow_html=True)
    for row in (CARDS[:2], CARDS[2:]):
        for col, (icon, title, sub, prompt) in zip(st.columns(2), row):
            with col:
                st.button(f"{icon} **{title}**\n:gray[{sub}]", key=f"card_{title}", on_click=queue_prompt,
                          args=(prompt,), use_container_width=True, disabled=locked)
else:
    # ---- Conversation view -----------------------------------------------
    for idx, msg in enumerate(chat["messages"]):
        with st.chat_message(msg["role"], avatar=AI_AVATAR if msg["role"] == "assistant" else USER_AVATAR):
            st.markdown(msg["content"])
            if msg.get("file"):
                st.caption(f"📎 {msg['file']}")
            if msg["role"] == "assistant":
                download_buttons(chat["id"], idx, msg["content"])

    if st.session_state.needs_reply:
        st.session_state.needs_reply = False
        attachment = get_attachment()
        question = chat["messages"][-1]["content"]
        idx = len(chat["messages"])
        with st.chat_message("assistant", avatar=AI_AVATAR):
            canned = creator_reply(question)
            if canned:
                reply = canned
                st.markdown(reply)
            else:
                try:
                    reply = st.write_stream(stream_reply(chat["messages"], attachment, is_coding(question, attachment)))
                except Exception as exc:  # keep the UI alive on any inference error
                    reply = "Oops, something went wrong on my side. Please try again."
                    st.markdown(reply)
                    st.toast(f"Error: {exc}")
            download_buttons(chat["id"], idx, reply)
        chat["messages"].append({"role": "assistant", "content": reply})

    tools_row(locked)

    st.chat_input(
        "Daily limit reached" if locked else "Ask anything...",
        key="chat_box",
        disabled=locked,
        on_submit=lambda: queue_prompt(st.session_state.get("chat_box", "")),
    )
