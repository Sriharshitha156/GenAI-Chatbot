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
            <div class="label">Status</div>
            <div class="value-green">● Ready to answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:0.8rem 0;'>", unsafe_allow_html=True)

    top_k = 5
    section_filter = None
    use_tools = st.checkbox("🔧 Enable tools (fee calc, date checker)", value=True, 
                            help="When enabled, the chatbot can use fee_calculator and date_checker tools for computations.")

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.18);font-size:0.62rem;"
        "margin-top:1.5rem;'>BVRITH · GenAI Lab · Day 4</p>",
        unsafe_allow_html=True,
    )

# ─── Chat Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="avatar">🎓</div>
    <div class="info">
        <h2>Zia — BVRITH FAQ</h2>
        <p><span class="status-dot"></span>RAG-powered · Answers grounded in the official knowledge base</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Chat History ────────────────────────────────────────────────────────────────
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

    # ── Suggested questions ──
    SUGGESTIONS = [
        "🏫 How do I get admission to BVRIT?",
        "💰 What are the fees at BVRIT?",
        "🏆 What are the placement stats?",
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
query = st.chat_input("Ask about BVRITH…")

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
            answer = "Here's a photo of BVRITH - College of Engineering for Women! 📸"
            citations = ["[Section: Campus & Facilities]"]
            refused = False
            latency = 0.0
        else:
            with st.spinner("Thinking…"):
                t0 = time.time()
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
                    # Show routing info in sidebar
                    routing = result["routing"]
                    if routing.startswith("tool:"):
                        st.caption(f"🔧 Used: {routing[5:]}")
                    elif routing == "RAG":
                        st.caption("📄 Used: RAG (knowledge base)")
                else:
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
