"""
BVRIT Hyderabad College FAQ Chatbot - RAG Powered
WhatsApp-style dark chat UI
"""
import os
import time
import dotenv
dotenv.load_dotenv(override=True)

import streamlit as st
from src.ingest import load_and_index_document
from src.retriever import get_retriever
from src.generator import generate_answer, FREE_MODEL

COLLEGE_IMAGE_PATH = "assets/college.png"
_IMAGE_KEYWORDS   = ["image", "photo", "picture", "look", "looks like", "appearance", "show me", "how does it look", "what does it look"]
_BVRIT_KEYWORDS   = ["bvrit", "bvrith", "bvrit hyderabad", "bvrit narsapur"]  # explicit BVRIT only

def _is_image_query(text: str, history: list | None = None) -> bool:
    """
    Returns True only when:
      1. The current query contains an image-related word, AND
      2. Either the current query mentions BVRIT explicitly,
         OR the last few messages in history mention BVRIT and the
         current query refers back (e.g. "that college", "the same college", "it").
    """
    t = text.lower()
    has_image_word = any(kw in t for kw in _IMAGE_KEYWORDS)
    if not has_image_word:
        return False

    # Direct BVRIT mention in current query
    if any(kw in t for kw in _BVRIT_KEYWORDS):
        return True

    # Back-reference words — only valid if a recent message mentioned BVRIT
    back_refs = [
        "that college", "the college", "this college", "same college",
        "the same", "that one", "the one", "the institute",
        "the university", "the campus", "previously asked",
        "asked before", "i asked", "the above",
    ]
    has_backref = any(_re.search(r'\b' + _re.escape(br) + r'\b', t) for br in back_refs)
    if has_backref and history:
        # Check last 6 message contents for BVRIT mention
        recent = " ".join(
            m.get("content", "").lower()
            for m in history[-6:]
        )
        if any(kw in recent for kw in _BVRIT_KEYWORDS):
            return True

    return False

import re as _re
def _strip_inline_citations(text: str) -> str:
    """Remove [Section: ...] and [Source N] tags the LLM embeds in its answer."""
    return _re.sub(r'\s*\[(?:Section|Source)[^\]]*\]', '', text).strip()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BVRIT Hyderabad FAQ",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ══════════════════════════════════════════════
       CSS CUSTOM PROPERTIES — DARK (default)
    ══════════════════════════════════════════════ */
    :root {
        --bg-app:        #111b21;
        --bg-sidebar:    #0a1217;
        --bg-header:     #1a2f38;
        --bg-bubble-bot: #1c2b33;
        --bg-bubble-usr: #005c4b;
        --bg-input:      #1c2b33;
        --bg-card:       rgba(255,255,255,0.03);
        --bg-collapsed:  #1f2c33;

        --border-card:   rgba(255,255,255,0.07);
        --border-input:  rgba(0,168,132,0.2);
        --border-side:   rgba(0,168,132,0.12);
        --border-header: rgba(0,168,132,0.12);
        --border-msg:    rgba(255,255,255,0.05);
        --border-cite:   rgba(255,255,255,0.05);
        --border-collapsed: rgba(255,255,255,0.1);

        --txt-primary:   #e9edef;
        --txt-secondary: rgba(233,237,239,0.45);
        --txt-label:     rgba(255,255,255,0.35);
        --txt-value:     #e9edef;
        --txt-welcome-h: #e9edef;
        --txt-welcome-p: rgba(233,237,239,0.45);

        --accent:        #00a884;
        --accent-dark:   #075e54;
        --accent-mid:    #0a7a6e;
        --accent-text:   #00c49a;

        --shadow-card:   0 1px 4px rgba(0,0,0,0.25);
        --shadow-header: 0 2px 12px rgba(0,0,0,0.3);
        --shadow-input:  0 2px 12px rgba(0,0,0,0.2);
        --shadow-send:   0 2px 8px rgba(0,168,132,0.3);
    }

    /* ══════════════════════════════════════════════
       LIGHT MODE OVERRIDES
    ══════════════════════════════════════════════ */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-app:        #f0f4f6;
            --bg-sidebar:    #ffffff;
            --bg-header:     #ffffff;
            --bg-bubble-bot: #ffffff;
            --bg-bubble-usr: #dcf8c6;
            --bg-input:      #ffffff;
            --bg-card:       rgba(0,0,0,0.03);
            --bg-collapsed:  #ffffff;

            --border-card:   rgba(0,0,0,0.08);
            --border-input:  rgba(0,168,132,0.3);
            --border-side:   rgba(0,168,132,0.15);
            --border-header: rgba(0,168,132,0.15);
            --border-msg:    rgba(0,0,0,0.07);
            --border-cite:   rgba(0,0,0,0.07);
            --border-collapsed: rgba(0,0,0,0.1);

            --txt-primary:   #111b21;
            --txt-secondary: rgba(17,27,33,0.55);
            --txt-label:     rgba(17,27,33,0.45);
            --txt-value:     #111b21;
            --txt-welcome-h: #111b21;
            --txt-welcome-p: rgba(17,27,33,0.5);

            --accent:        #00a884;
            --accent-dark:   #075e54;
            --accent-mid:    #0a7a6e;
            --accent-text:   #007a62;

            --shadow-card:   0 1px 6px rgba(0,0,0,0.08);
            --shadow-header: 0 2px 12px rgba(0,0,0,0.08);
            --shadow-input:  0 2px 8px rgba(0,0,0,0.06);
            --shadow-send:   0 2px 8px rgba(0,168,132,0.25);
        }
    }

    /* Streamlit also sets its own theme class — override for both */
    [data-theme="light"] {
        --bg-app:        #f0f4f6;
        --bg-sidebar:    #ffffff;
        --bg-header:     #ffffff;
        --bg-bubble-bot: #ffffff;
        --bg-bubble-usr: #dcf8c6;
        --bg-input:      #ffffff;
        --bg-card:       rgba(0,0,0,0.03);
        --bg-collapsed:  #ffffff;
        --border-card:   rgba(0,0,0,0.08);
        --border-input:  rgba(0,168,132,0.3);
        --border-side:   rgba(0,168,132,0.15);
        --border-header: rgba(0,168,132,0.15);
        --border-msg:    rgba(0,0,0,0.07);
        --border-cite:   rgba(0,0,0,0.07);
        --border-collapsed: rgba(0,0,0,0.1);
        --txt-primary:   #111b21;
        --txt-secondary: rgba(17,27,33,0.55);
        --txt-label:     rgba(17,27,33,0.45);
        --txt-value:     #111b21;
        --txt-welcome-h: #111b21;
        --txt-welcome-p: rgba(17,27,33,0.5);
        --accent-text:   #007a62;
        --shadow-card:   0 1px 6px rgba(0,0,0,0.08);
        --shadow-header: 0 2px 12px rgba(0,0,0,0.08);
        --shadow-input:  0 2px 8px rgba(0,0,0,0.06);
        --shadow-send:   0 2px 8px rgba(0,168,132,0.25);
    }

    /* ══════════════════════════════════════════════
       LAYOUT
    ══════════════════════════════════════════════ */
    .stApp {
        background: var(--bg-app) !important;
        min-height: 100vh;
        transition: background 0.3s;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar collapse button ── */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: var(--bg-collapsed) !important;
        border-radius: 0 8px 8px 0 !important;
        border: 1px solid var(--border-collapsed) !important;
        border-left: none !important;
        color: var(--accent) !important;
        z-index: 9999 !important;
        transition: background 0.2s !important;
    }
    [data-testid="collapsedControl"]:hover { background: rgba(0,168,132,0.12) !important; }
    [data-testid="collapsedControl"] svg   { fill: var(--accent) !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-side) !important;
        transition: background 0.3s;
    }

    .sidebar-header {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        border-radius: 14px;
        padding: 1.1rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,168,132,0.25);
    }
    .sidebar-header h3 { color: white; font-size: 1rem; font-weight: 700; margin: 0; letter-spacing: 0.3px; }
    .sidebar-header p  { color: rgba(255,255,255,0.8); font-size: 0.7rem; margin-top: 0.3rem; }

    .info-card {
        background: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 10px;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s, background 0.2s;
    }
    .info-card:hover { border-color: rgba(0,168,132,0.3); }
    .info-card .label {
        color: var(--txt-label);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .info-card .value       { color: var(--txt-value); font-size: 0.88rem; font-weight: 600; }
    .info-card .value-green { color: var(--accent); font-size: 0.88rem; font-weight: 600; }

    /* ── Chat header ── */
    .chat-header {
        background: var(--bg-header);
        padding: 0.75rem 1.4rem;
        border: 1px solid var(--border-header);
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.75rem;
        box-shadow: var(--shadow-header);
        transition: background 0.3s;
    }
    .chat-header .avatar {
        width: 42px; height: 42px;
        background: linear-gradient(135deg, var(--accent), var(--accent-mid));
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; flex-shrink: 0;
        box-shadow: 0 0 0 3px rgba(0,168,132,0.2);
    }
    .chat-header .info h2 {
        color: var(--txt-primary);
        font-size: 0.97rem; font-weight: 700; margin: 0; letter-spacing: 0.2px;
    }
    .chat-header .info p {
        color: var(--txt-secondary);
        font-size: 0.72rem; margin: 0.1rem 0 0; font-style: italic;
    }
    .chat-header .status-dot {
        display: inline-block;
        width: 7px; height: 7px;
        background: var(--accent);
        border-radius: 50%;
        margin-right: 4px;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(0.85); }
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.2rem 0 !important;
        animation: msg-in 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) both;
    }
    @keyframes msg-in {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessageContent"] {
        background: var(--bg-bubble-bot) !important;
        border-radius: 12px !important;
        color: var(--txt-primary) !important;
        border: 1px solid var(--border-msg) !important;
        box-shadow: var(--shadow-card) !important;
        font-size: 0.9rem !important;
        line-height: 1.55 !important;
        transition: box-shadow 0.2s, background 0.2s !important;
    }
    [data-testid="stChatMessageContent"]:hover {
        box-shadow: 0 3px 10px rgba(0,0,0,0.15) !important;
    }
    /* Streamlit marks user messages — make them green-tinted */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: var(--bg-bubble-usr) !important;
    }

    /* ── Avatars ── */
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #00a884, #0d6b8e) !important;
        border-radius: 50% !important; border: none !important;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #1a6fa8, #1e3a5f) !important;
        border-radius: 50% !important; border: none !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] svg,
    [data-testid="stChatMessageAvatarAssistant"] * { color: white !important; fill: white !important; }
    [data-testid="stChatMessageAvatarUser"] svg,
    [data-testid="stChatMessageAvatarUser"] * { color: #a8d8f0 !important; fill: #a8d8f0 !important; }

    /* ── Citations ── */
    .citation-tag {
        display: inline-block;
        background: rgba(0,168,132,0.1);
        border: 1px solid rgba(0,168,132,0.25);
        color: var(--accent-text);
        font-size: 0.64rem;
        padding: 0.12rem 0.55rem;
        border-radius: 20px;
        margin: 0.15rem 0.1rem;
        transition: background 0.15s;
    }
    .citation-tag:hover { background: rgba(0,168,132,0.22); }
    .citations-row {
        margin-top: 0.55rem;
        padding-top: 0.45rem;
        border-top: 1px solid var(--border-cite);
    }
    .refused-badge {
        display: inline-block;
        background: linear-gradient(90deg, #c0392b, #e74c3c);
        color: white;
        font-size: 0.6rem; font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 4px; margin-bottom: 0.45rem;
        text-transform: uppercase; letter-spacing: 0.5px;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] > div {
        background: var(--bg-input) !important;
        border: 1.5px solid var(--border-input) !important;
        border-radius: 28px !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
        box-shadow: var(--shadow-input) !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(0,168,132,0.12), var(--shadow-input) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--txt-primary) !important;
        background: transparent !important;
        font-size: 0.92rem !important;
        caret-color: var(--accent) !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #00a884, #075e54) !important;
        border-radius: 50% !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
        box-shadow: var(--shadow-send) !important;
    }
    [data-testid="stChatInput"] button:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 4px 14px rgba(0,168,132,0.45) !important;
    }

    /* ── Suggestion chips ── */
    div[data-testid="stColumns"] .stButton > button {
        background: rgba(0,168,132,0.07) !important;
        border: 1px solid rgba(0,168,132,0.25) !important;
        color: var(--accent-text) !important;
        border-radius: 22px !important;
        font-size: 0.78rem !important;
        padding: 0.45rem 0.75rem !important;
        white-space: normal !important;
        line-height: 1.35 !important;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    div[data-testid="stColumns"] .stButton > button:hover {
        background: rgba(0,168,132,0.16) !important;
        border-color: rgba(0,168,132,0.5) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 14px rgba(0,168,132,0.18) !important;
    }

    /* ── Sidebar buttons ── */
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--bg-card) !important;
        color: var(--txt-secondary) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        transition: all 0.15s !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0,168,132,0.1) !important;
        border-color: rgba(0,168,132,0.3) !important;
        color: var(--accent) !important;
    }

    /* ── Welcome screen ── */
    .welcome-wrap {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 50vh; text-align: center;
        padding: 2rem 1rem 1rem;
        animation: fade-up 0.5s ease both;
    }
    @keyframes fade-up {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .welcome-wrap .icon {
        font-size: 3.8rem; margin-bottom: 0.9rem;
        filter: drop-shadow(0 4px 12px rgba(0,168,132,0.3));
    }
    .welcome-wrap h3 {
        color: var(--txt-welcome-h);
        font-size: 1.2rem; font-weight: 700;
        margin: 0 0 0.4rem; letter-spacing: 0.2px;
    }
    .welcome-wrap p {
        color: var(--txt-welcome-p);
        font-size: 0.83rem; max-width: 360px; line-height: 1.6; margin-bottom: 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(0,168,132,0.25);
        border-radius: 4px;
        transition: background 0.2s;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,168,132,0.45); }
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []        # {role, content, citations, refused, time}
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
    st.session_state.vectorstore = None
    st.session_state.chunk_count = 0
    st.session_state.retriever = None
if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = None

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-header"><h3>💬 BVRIT Hyderabad</h3>'
        '<p>FAQ Chatbot · RAG Powered</p></div>',
        unsafe_allow_html=True,
    )

    doc_path = "bvrit_knowledge_base.docx"

    if not st.session_state.index_ready:
        with st.status("📚 Indexing knowledge base…", expanded=True) as status:
            try:
                vs, n = load_and_index_document(doc_path)
                st.session_state.vectorstore = vs
                st.session_state.chunk_count = n
                st.session_state.retriever = get_retriever(vs)
                st.session_state.index_ready = True
                status.update(label=f"✅ Ready — {n} chunks indexed", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Indexing failed", state="error")
                st.error(str(e))
                st.stop()

    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">Knowledge Base</div>
            <div class="value-green">● LIVE · {st.session_state.chunk_count} chunks</div>
        </div>
        <div class="info-card">
            <div class="label">Model</div>
            <div class="value">{FREE_MODEL.split("/")[-1]}</div>
        </div>
        <div class="info-card">
            <div class="label">Document</div>
            <div class="value" style="font-size:0.75rem;">{doc_path}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:0.8rem 0;'>", unsafe_allow_html=True)

    top_k = st.slider("Top-K results", min_value=3, max_value=10, value=5)

    sections = [
        "All", "About BVRITH", "Departments", "Admissions",
        "Fee Structure", "Placements", "Campus & Facilities", "Faculty", "Contact",
    ]
    selected_section = st.selectbox("Section filter", sections)
    section_filter = None if selected_section == "All" else selected_section

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:0.8rem 0;'>", unsafe_allow_html=True)

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.18);font-size:0.62rem;"
        "margin-top:1.5rem;'>BVRIT · GenAI Lab · Day 4</p>",
        unsafe_allow_html=True,
    )

# ─── Chat Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="avatar">🎓</div>
    <div class="info">
        <h2>BVRIT Hyderabad FAQ</h2>
        <p><span class="status-dot"></span>Powered by RAG · Answers grounded in the official knowledge base</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Chat History ────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="icon">💬</div>
        <h3>Ask me anything about BVRIT!</h3>
        <p>Admissions · Fees · Placements · Departments · Campus · Faculty · Contact</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Suggested questions ──
    SUGGESTIONS = [
        "📋 What B.Tech departments are offered at BVRIT?",
        "💰 What is the fee structure at BVRIT Hyderabad?",
        "🏆 What are the placement stats at BVRIT?",
    ]
    st.markdown("<div style='display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap;padding:0 1rem 1rem;'>", unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTIONS))
    for i, (col, suggestion) in enumerate(zip(cols, SUGGESTIONS)):
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                # Strip the emoji prefix to get the clean question
                clean = suggestion.split(" ", 1)[1]
                st.session_state.suggested_query = clean
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role = msg["role"]  # "user" or "assistant"
        with st.chat_message(role):
            if role == "assistant" and msg.get("refused"):
                st.markdown('<span class="refused-badge">⛔ Out of scope</span>', unsafe_allow_html=True)

            st.markdown(_strip_inline_citations(msg["content"]))

            if role == "assistant" and msg.get("citations"):
                cite_html = '<div class="citations-row">'
                for c in msg["citations"]:
                    cite_html += f'<span class="citation-tag">{c}</span>'
                cite_html += "</div>"
                st.markdown(cite_html, unsafe_allow_html=True)

            if role == "assistant" and msg.get("latency"):
                st.caption(f"⏱ {msg['latency']:.1f}s · {msg.get('time','')}")

            # Show college image if the original query was image-related
            if role == "assistant" and msg.get("show_image") and os.path.exists(COLLEGE_IMAGE_PATH):
                st.image(COLLEGE_IMAGE_PATH, caption="BVRIT Hyderabad — College Entrance", use_container_width=True)

# ─── Chat Input (st.chat_input) ──────────────────────────────────────────────────
query = st.chat_input("Ask about BVRIT Hyderabad…")

# Pick up a suggestion click if no typed query
if not query and st.session_state.suggested_query:
    query = st.session_state.suggested_query
    st.session_state.suggested_query = None

if query:
    now = time.strftime("%I:%M %p")

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query, "time": now})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        show_image = _is_image_query(query, st.session_state.messages) and os.path.exists(COLLEGE_IMAGE_PATH)

        if show_image:
            # Don't call LLM for image queries — just give a friendly response
            answer = "Here's a photo of BVRIT Hyderabad College of Engineering for Women! 📸"
            citations = ["[Section: Campus & Facilities]"]
            refused = False
            latency = 0.0
        else:
            with st.spinner("Thinking…"):
                t0 = time.time()
                answer, citations, refused = generate_answer(
                    retriever=st.session_state.retriever,
                    query=query,
                    top_k=top_k,
                    section_filter=section_filter,
                )
                latency = time.time() - t0

        if refused:
            st.markdown('<span class="refused-badge">⛔ Out of scope</span>', unsafe_allow_html=True)

        st.markdown(_strip_inline_citations(answer))

        if citations:
            cite_html = '<div class="citations-row">'
            for c in citations:
                cite_html += f'<span class="citation-tag">{c}</span>'
            cite_html += "</div>"
            st.markdown(cite_html, unsafe_allow_html=True)

        if latency > 0:
            st.caption(f"⏱ {latency:.1f}s · {time.strftime('%I:%M %p')}")

        # Show college image if query is about how the college looks
        if show_image:
            st.image(COLLEGE_IMAGE_PATH, caption="BVRIT Hyderabad — College Entrance", use_container_width=True)

    # Save assistant message
    now = time.strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "refused": refused,
        "latency": latency,
        "time": now,
        "show_image": show_image,
    })
