"""
BVRITH FAQ Chatbot - RAG-Powered
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
from agentic_chain import generate_with_tools

@st.cache_resource(show_spinner=False)
def _load_index(doc_path: str):
    return load_and_index_document(doc_path)

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
    page_title="Zia · BVRITH FAQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        --bg-app:        #1a0f0a;
        --bg-sidebar:    #150d08;
        --bg-header:     rgba(255,255,255,0.04);
        --bg-bubble-bot: rgba(255,255,255,0.05);
        --bg-input:      rgba(255,255,255,0.06);
        --bg-card:       rgba(255,255,255,0.04);

        --border-card:   rgba(180,120,80,0.2);
        --border-input:  rgba(160,90,60,0.4);
        --border-side:   rgba(140,80,50,0.25);
        --border-msg:    rgba(180,120,80,0.12);

        --txt-primary:   #f5ede4;
        --txt-secondary: rgba(245,237,228,0.5);
        --txt-label:     rgba(245,237,228,0.35);

        --accent:        #9b4f2e;
        --accent2:       #6b2d1e;
        --accent-text:   #d4956a;

        --shadow-card:   0 4px 24px rgba(80,30,10,0.3);
        --shadow-header: 0 8px 32px rgba(0,0,0,0.5);
        --shadow-input:  0 4px 20px rgba(100,40,20,0.25);
        --shadow-send:   0 4px 16px rgba(155,79,46,0.5);
    }

    /* ══ LAYOUT ══ */
    .stApp {
        background: linear-gradient(135deg, #1a0f0a 0%, #2a1508 50%, #1e0e07 100%) !important;
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: rgba(155,79,46,0.2) !important;
        border-radius: 0 8px 8px 0 !important;
        border: 1px solid rgba(155,79,46,0.3) !important;
        border-left: none !important;
        z-index: 9999 !important;
    }
    [data-testid="collapsedControl"] svg { fill: var(--accent-text) !important; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150d08 0%, #1e1008 100%) !important;
        border-right: 1px solid rgba(140,80,50,0.25) !important;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #7a3520, #4a1a0e);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(100,40,10,0.5);
        position: relative;
        overflow: hidden;
    }
    .sidebar-header::before {
        content: '';
        position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,200,150,0.08) 0%, transparent 60%);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer {
        0%,100% { transform: translate(-30%,-30%); }
        50%      { transform: translate(-20%,-20%); }
    }
    .sidebar-header h3 { color: #f5ede4; font-size: 1.05rem; font-weight: 700; margin: 0; }
    .sidebar-header p  { color: rgba(245,237,228,0.65); font-size: 0.7rem; margin-top: 0.3rem; }

    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(180,120,80,0.2);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.2s;
    }
    .info-card:hover { border-color: rgba(155,79,46,0.45); background: rgba(155,79,46,0.08); }
    .info-card .label { color: var(--txt-label); font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.8px; }
    .info-card .value { color: var(--txt-primary); font-size: 0.85rem; font-weight: 600; }
    .info-card .value-green {
        background: linear-gradient(90deg, #d4956a, #9b4f2e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 0.85rem; font-weight: 700;
    }

    /* ── Chat header ── */
    .chat-header {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        padding: 1rem 1.5rem;
        border: 1px solid rgba(180,120,80,0.2);
        border-radius: 16px;
        display: flex; align-items: center; gap: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,200,150,0.06);
    }
    .chat-header .avatar {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #7a3520, #9b4f2e);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; flex-shrink: 0;
        box-shadow: 0 0 0 3px rgba(155,79,46,0.3), 0 4px 16px rgba(100,40,10,0.5);
    }
    .chat-header .info h2 {
        color: #f5ede4; font-size: 1.05rem; font-weight: 700; margin: 0;
        background: linear-gradient(90deg, #f5ede4, #d4956a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .chat-header .info p { color: var(--txt-secondary); font-size: 0.73rem; margin: 0.15rem 0 0; }
    .chat-header .status-dot {
        display: inline-block; width: 7px; height: 7px;
        background: #c17a4a; border-radius: 50%; margin-right: 5px;
        box-shadow: 0 0 6px #c17a4a;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%,100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.4; transform: scale(0.8); }
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.3rem 0 !important;
        animation: msg-in 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) both;
    }
    @keyframes msg-in {
        from { opacity: 0; transform: translateY(12px) scale(0.98); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    [data-testid="stChatMessageContent"] {
        background: rgba(255,255,255,0.06) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        color: #f0eeff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2) !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(122,53,32,0.4), rgba(100,35,20,0.4)) !important;
        border-color: rgba(155,79,46,0.35) !important;
    }

    /* ── Avatars ── */
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #7a3520, #9b4f2e) !important;
        border-radius: 50% !important; border: none !important;
        box-shadow: 0 0 12px rgba(155,79,46,0.4) !important;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #9b4f2e, #6b2d1e) !important;
        border-radius: 50% !important; border: none !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] svg,
    [data-testid="stChatMessageAvatarAssistant"] * { color: #f5ede4 !important; fill: #f5ede4 !important; }
    [data-testid="stChatMessageAvatarUser"] svg,
    [data-testid="stChatMessageAvatarUser"] * { color: #f5ede4 !important; fill: #f5ede4 !important; }

    /* ── Citations ── */
    .citation-tag {
        display: inline-block;
        background: rgba(155,79,46,0.15);
        border: 1px solid rgba(155,79,46,0.35);
        color: #d4956a;
        font-size: 0.65rem; font-weight: 500;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        margin: 0.15rem 0.1rem;
        transition: all 0.2s;
    }
    .citation-tag:hover { background: rgba(155,79,46,0.3); transform: translateY(-1px); }
    .citations-row {
        margin-top: 0.6rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(180,120,80,0.15);
    }
    .refused-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7a1e1e, #a83232);
        color: #f5ede4; font-size: 0.62rem; font-weight: 700;
        padding: 0.2rem 0.6rem; border-radius: 6px;
        margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(120,30,30,0.4);
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] > div {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1.5px solid rgba(155,79,46,0.35) !important;
        border-radius: 32px !important;
        box-shadow: 0 4px 20px rgba(80,30,10,0.3) !important;
        transition: all 0.3s !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #9b4f2e !important;
        box-shadow: 0 0 0 3px rgba(155,79,46,0.2), 0 4px 20px rgba(100,40,20,0.3) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f5ede4 !important;
        background: transparent !important;
        font-size: 0.93rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #7a3520, #9b4f2e) !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 16px rgba(155,79,46,0.5) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stChatInput"] button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 20px rgba(155,79,46,0.7) !important;
    }

    /* ── Topic / Suggestion chips ── */
    div[data-testid="stColumns"] .stButton > button {
        background: rgba(155,79,46,0.1) !important;
        border: 1px solid rgba(155,79,46,0.3) !important;
        color: #d4956a !important;
        border-radius: 24px !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 0.8rem !important;
        white-space: normal !important;
        line-height: 1.4 !important;
        transition: all 0.25s !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="stColumns"] .stButton > button:hover {
        background: rgba(155,79,46,0.25) !important;
        border-color: #9b4f2e !important;
        color: #f5ede4 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(100,40,20,0.4) !important;
    }

    /* ── Sidebar buttons ── */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.04) !important;
        color: rgba(245,237,228,0.6) !important;
        border: 1px solid rgba(180,120,80,0.2) !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        transition: all 0.2s !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(155,79,46,0.15) !important;
        border-color: rgba(155,79,46,0.4) !important;
        color: #d4956a !important;
    }

    /* ── Welcome screen ── */
    .welcome-wrap {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 52vh; text-align: center;
        padding: 2rem 1rem 1rem;
        animation: fade-up 0.6s ease both;
    }
    @keyframes fade-up {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .welcome-wrap .icon {
        font-size: 4.5rem; margin-bottom: 1rem;
        filter: drop-shadow(0 0 24px rgba(155,79,46,0.6));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%,100% { transform: translateY(0); }
        50%      { transform: translateY(-8px); }
    }
    .welcome-wrap h3 {
        font-size: 1.5rem; font-weight: 700; margin: 0 0 0.5rem;
        background: linear-gradient(90deg, #f5ede4, #d4956a, #9b4f2e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .welcome-wrap p {
        color: rgba(245,237,228,0.5);
        font-size: 0.87rem; max-width: 380px; line-height: 1.7;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(155,79,46,0.3); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(155,79,46,0.55); }

    .stSelectbox label, .stSlider label { color: rgba(245,237,228,0.6) !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Memory imports (Day 5) ────────────────────────────────────────────────────
from src.memory import (
    load_profile, save_profile, delete_profile,
    load_history, save_history,
    build_personalised_system_prompt, update_profile_from_turn,
    compress_history, is_clear_data_command,
    PRIVACY_NOTICE, _SUMMARISE_EVERY,
)

# ─── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []     # compressed history sent to LLM
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
    st.session_state.vectorstore = None
    st.session_state.chunk_count = 0
    st.session_state.retriever = None
if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "profile" not in st.session_state:
    st.session_state.profile = load_profile(st.session_state.user_id)
if "llm_history" not in st.session_state:
    # Restore history from ChromaDB (separate collection from KB)
    st.session_state.llm_history = load_history(st.session_state.user_id)
if "privacy_shown" not in st.session_state:
    st.session_state.privacy_shown = not st.session_state.profile.get("is_new", True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-header"><h3>🎓 Zia · BVRITH FAQ</h3>'
        '<p>RAG-Powered AI Assistant</p></div>',
        unsafe_allow_html=True,
    )

    # Get the directory where this script is located
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(_script_dir, "bvrit_knowledge_base.docx")

    if not st.session_state.index_ready:
        with st.status("📚 Indexing knowledge base…", expanded=True) as status:
            try:
                vs, n = _load_index(doc_path)
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
            <div class="label">Status</div>
            <div class="value-green">● Ready to answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:0.8rem 0;'>", unsafe_allow_html=True)

    top_k = 5
    section_filter = None

    # Section filter dropdown
    SECTION_OPTIONS = [
        "All Sections",
        "About BVRIT",
        "Departments",
        "Admissions",
        "Fee Structure",
        "Placements",
        "Campus & Facilities",
        "Faculty",
        "Contact",
    ]
    selected_section = st.selectbox(
        "🔍 Filter by Section",
        SECTION_OPTIONS,
        help="Restrict retrieval to a specific knowledge base section",
    )
    if selected_section != "All Sections":
        section_filter = selected_section

    use_tools = st.checkbox("🔧 Enable tools (fee calc, date checker)", value=True, 
                            help="When enabled, the chatbot can use fee_calculator and date_checker tools for computations.")

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm_history = []
        save_history(st.session_state.user_id, [])  # wipe from ChromaDB too
        st.rerun()

    # Profile display (Exercise 3+4)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:0.8rem 0;'>", unsafe_allow_html=True)
    _p = st.session_state.profile
    if _p.get("name"):
        st.markdown(f"👤 **{_p['name']}**")
    if _p.get("branch_interest"):
        st.caption(f"🎓 {_p['branch_interest']}")
    if _p.get("detail_level", "normal") != "normal":
        st.caption(f"📝 Style: {_p['detail_level']}")
    _turns = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.caption(f"💬 {_turns} turns this session")

    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.18);font-size:0.62rem;"
        "margin-top:1.5rem;'>BVRITH · GenAI Lab · Day 5</p>",
        unsafe_allow_html=True,
    )

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])

with tab_chat:
    st.markdown("""
    <div class="chat-header">
        <div class="avatar">🎓</div>
        <div class="info">
            <h2>Zia — BVRITH FAQ</h2>
            <p><span class="status-dot"></span>RAG-powered · Answers grounded in the official knowledge base</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-wrap">
            <div class="icon">🎓</div>
            <h3>Hey! I'm Zia 👋</h3>
            <p>Your personal AI guide to BVRITH - College of Engineering for Women.<br>
            Ask me anything — I'm here to help!</p>
            <p style="color:rgba(240,238,255,0.35);font-size:0.75rem;margin-top:1rem;">👇 Tap a topic to get started</p>
        </div>
        """, unsafe_allow_html=True)

        TOPICS = [
            ("🏫", "Admissions", "How do I get admission to BVRIT Hyderabad?"),
            ("💰", "Fee Structure", "What is the fee structure at BVRIT Hyderabad?"),
            ("🏆", "Placements", "What are the placement stats at BVRIT?"),
            ("📚", "Departments", "What B.Tech departments are offered at BVRIT?"),
            ("🏕️", "Campus Life", "Tell me about campus facilities at BVRIT"),
            ("📞", "Contact", "How can I contact BVRIT Hyderabad?"),
        ]
        col1, col2, col3 = st.columns(3)
        for i, (emoji, label, query_text) in enumerate(TOPICS):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(f"{emoji} {label}", key=f"topic_{i}", use_container_width=True):
                    st.session_state.suggested_query = query_text
                    st.rerun()

        SUGGESTIONS = [
            "🏫 How do I get admission to BVRIT?",
            "💰 What are the fees at BVRIT?",
            "🏆 What are the placement stats?",
        ]
        cols = st.columns(len(SUGGESTIONS))
        for i, (col, suggestion) in enumerate(zip(cols, SUGGESTIONS)):
            with col:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    clean = suggestion.split(" ", 1)[1]
                    st.session_state.suggested_query = clean
                    st.rerun()
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
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
                if role == "assistant" and msg.get("show_image") and os.path.exists(COLLEGE_IMAGE_PATH):
                    st.image(COLLEGE_IMAGE_PATH, caption="BVRIT Hyderabad — College Entrance", use_container_width=True)

    # Chat input lives inside tab_chat
    query = st.chat_input("Ask about BVRITH…")

    if not query and st.session_state.suggested_query:
        query = st.session_state.suggested_query
        st.session_state.suggested_query = None

    if query:
        now = time.strftime("%I:%M %p")

        # Exercise 5: privacy notice on first turn
        if not st.session_state.privacy_shown:
            st.session_state.messages.append({"role": "assistant", "content": PRIVACY_NOTICE, "citations": [], "refused": False, "latency": 0.0, "time": now, "show_image": False})
            st.session_state.privacy_shown = True

        # Exercise 5: clear data command
        if is_clear_data_command(query):
            delete_profile(st.session_state.user_id)
            st.session_state.profile = load_profile(st.session_state.user_id)
            st.session_state.llm_history = []
            st.session_state.messages.append({"role": "user", "content": query, "time": now})
            st.session_state.messages.append({"role": "assistant", "content": "✅ Done! I've deleted all your personal data. I'll treat you as a new user from now on. 🔒", "citations": [], "refused": False, "latency": 0.0, "time": now, "show_image": False})
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": query, "time": now})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            show_image = _is_image_query(query, st.session_state.messages) and os.path.exists(COLLEGE_IMAGE_PATH)

            if show_image:
                answer = "Here's a photo of BVRITH - College of Engineering for Women! 📸"
                citations = ["[Section: Campus & Facilities]"]
                refused = False
                latency = 0.0
            else:
                with st.spinner("Thinking…"):
                    t0 = time.time()

                    # Exercise 2: compress history every _SUMMARISE_EVERY turns
                    user_turns = len([m for m in st.session_state.messages if m["role"] == "user"])
                    if user_turns > 0 and user_turns % _SUMMARISE_EVERY == 0:
                        st.session_state.llm_history = compress_history(st.session_state.llm_history)

                    # Exercise 4: personalised system prompt
                    from src.generator import SYSTEM_PROMPT
                    personalised_prompt = build_personalised_system_prompt(SYSTEM_PROMPT, st.session_state.profile)

                    if use_tools:
                        result = generate_with_tools(
                            retriever=st.session_state.retriever,
                            query=query,
                            top_k=top_k,
                            section_filter=section_filter,
                        )
                        answer = result["answer"]
                        citations = result["citations"]
                        refused = result["refused"]
                        latency = result["latency_seconds"]
                        routing = result["routing"]
                        if routing.startswith("tool:"):
                            st.caption(f"🔧 Used: {routing[5:]}")
                        elif routing == "RAG":
                            st.caption("📄 Used: RAG (knowledge base)")
                    else:
                        # Exercise 1: pass full conversation history
                        answer, citations, refused = generate_answer(
                            retriever=st.session_state.retriever,
                            query=query,
                            top_k=top_k,
                            section_filter=section_filter,
                            conversation_history=st.session_state.llm_history,
                            system_prompt_override=personalised_prompt,
                        )
                        latency = time.time() - t0

                    # Exercise 1: append this turn to llm_history
                    st.session_state.llm_history.append({"role": "user", "content": query})
                    st.session_state.llm_history.append({"role": "assistant", "content": answer})

                    # Exercise 3+4: extract facts and update persistent profile
                    st.session_state.profile = update_profile_from_turn(st.session_state.profile, query, answer)
                    save_profile(st.session_state.profile)
                    # Persist history to ChromaDB (separate from KB collection)
                    save_history(st.session_state.user_id, st.session_state.llm_history)

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
            if show_image:
                st.image(COLLEGE_IMAGE_PATH, caption="BVRIT Hyderabad — College Entrance", use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "refused": refused,
            "latency": latency,
            "time": time.strftime("%I:%M %p"),
            "show_image": show_image,
        })
        st.rerun()

# ─── Evaluation Dashboard Tab ────────────────────────────────────────────────────
with tab_eval:
    from src.evaluation import run_evaluation, get_fallback_test_cases

    st.markdown("### 📊 8-Dimension Evaluation Dashboard")
    st.caption("Run the full test suite against the live chatbot and view results with RAGAS scores.")

    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
        st.session_state.eval_report = None

    if st.button("▶️ Run Evaluation Suite", use_container_width=True, key="run_eval"):
        if not st.session_state.index_ready:
            st.error("Knowledge base not ready. Please wait for indexing to complete.")
        else:
            import docx2txt
            try:
                kb_text = docx2txt.process(doc_path)
            except Exception:
                kb_text = ""

            # ── Live progress UI ──
            progress_bar = st.progress(0, text="Preparing test cases…")
            status_box = st.empty()
            live_log = st.empty()
            live_results: list = []

            def on_progress(i, total, tc, result):
                live_results.append(result)
                pct = int(i / total * 100)
                q = (tc.get("question") or "(empty)")[:55]
                dim = tc.get("dimension", "")
                passed_so_far = sum(1 for r in live_results if r.get("passed"))
                icon = "✅" if result.get("passed") else "❌"
                lat = result.get("latency", 0)
                progress_bar.progress(pct, text=f"Test {i}/{total} — {dim}: {q}")
                status_box.markdown(
                    f"**{icon} [{dim}]** `{q}` — `{lat:.1f}s`"
                )
                live_log.markdown(
                    f"✅ **{passed_so_far} passed**  |  "
                    f"❌ **{i - passed_so_far} failed**  |  "
                    f"📊 **{i}/{total} done**"
                )

            results, report = run_evaluation(
                retriever=st.session_state.retriever,
                knowledge_base_text=kb_text,
                top_k=5,
                progress_callback=on_progress,
            )
            progress_bar.progress(100, text="✅ Evaluation complete!")
            status_box.empty()
            st.session_state.eval_results = results
            st.session_state.eval_report = report
            st.rerun()

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        failed = total - passed
        pass_rate = passed / total * 100 if total else 0

        # ── Summary metric cards ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tests", total)
        c2.metric("✅ Passed", passed)
        c3.metric("❌ Failed", failed)
        c4.metric("Pass Rate", f"{pass_rate:.0f}%")

        st.markdown("---")

        # ── Per-dimension bar chart ──
        st.markdown("#### Pass Rate by Dimension")
        dims: dict = {}
        for r in results:
            d = r.get("dimension", "Unknown")
            dims.setdefault(d, {"total": 0, "passed": 0})
            dims[d]["total"] += 1
            if r.get("passed"):
                dims[d]["passed"] += 1

        dim_names = list(dims.keys())
        dim_rates = [dims[d]["passed"] / dims[d]["total"] * 100 for d in dim_names]
        dim_colors = ["#2ecc71" if r == 100 else "#e67e22" if r >= 50 else "#e74c3c" for r in dim_rates]

        bar_html = "<div style='display:flex;flex-direction:column;gap:0.5rem;'>"
        for name, rate, color in zip(dim_names, dim_rates, dim_colors):
            p = dims[name]["passed"]
            t = dims[name]["total"]
            bar_html += f"""
            <div style='display:flex;align-items:center;gap:0.8rem;'>
                <div style='width:120px;font-size:0.8rem;color:#f5ede4;'>{name}</div>
                <div style='flex:1;background:rgba(255,255,255,0.08);border-radius:6px;height:20px;'>
                    <div style='width:{rate:.0f}%;background:{color};border-radius:6px;height:20px;'></div>
                </div>
                <div style='width:60px;font-size:0.8rem;color:#d4956a;text-align:right;'>{p}/{t} ({rate:.0f}%)</div>
            </div>"""
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown("---")

        # ── RAGAS scores ──
        ragas_results = [r for r in results if r.get("ragas_scores")]
        if ragas_results:
            st.markdown("#### RAGAS Metric Scores")
            metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            rc1, rc2, rc3, rc4 = st.columns(4)
            for col, metric in zip([rc1, rc2, rc3, rc4], metrics):
                avg = sum(r["ragas_scores"].get(metric, 0.0) for r in ragas_results) / len(ragas_results)
                col.metric(metric.replace("_", " ").title(), f"{avg:.2f}")
            st.markdown("---")

        # ── Latency chart ──
        st.markdown("#### Response Latency per Test")
        lat_html = "<div style='display:flex;flex-wrap:wrap;gap:0.4rem;'>"
        for r in results:
            lat = r.get("latency", 0)
            color = "#2ecc71" if lat < 5 else "#e67e22" if lat < 15 else "#e74c3c"
            q_short = r.get("question", "")[:30] + "…"
            lat_html += f"<div title='{q_short}' style='background:{color};color:#fff;font-size:0.7rem;padding:0.2rem 0.5rem;border-radius:4px;'>{lat:.1f}s</div>"
        lat_html += "</div>"
        st.markdown(lat_html, unsafe_allow_html=True)

        st.markdown("---")

        # ── Pass/Fail cards ──
        st.markdown("#### Test Case Results")
        for r in results:
            status = "✅" if r.get("passed") else "❌"
            color = "rgba(46,204,113,0.1)" if r.get("passed") else "rgba(231,76,60,0.1)"
            border = "rgba(46,204,113,0.3)" if r.get("passed") else "rgba(231,76,60,0.3)"
            q = r.get("question", "(empty)") or "(empty)"
            dim = r.get("dimension", "")
            lat = r.get("latency", 0)
            st.markdown(
                f"<div style='background:{color};border:1px solid {border};border-radius:8px;"
                f"padding:0.5rem 0.8rem;margin-bottom:0.4rem;'>"
                f"<b>{status} [{dim}]</b> {q[:80]} "
                f"<span style='color:#d4956a;font-size:0.75rem;'>⏱ {lat:.1f}s</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Full text report ──
        with st.expander("📄 Full Evaluation Report"):
            st.code(st.session_state.eval_report, language="")
    else:
        st.info("Click ▶️ Run Evaluation Suite to start. Results will appear here.")
