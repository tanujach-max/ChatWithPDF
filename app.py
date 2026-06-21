import streamlit as st
import os
from pdf_processor import extract_text_from_pdf, split_into_chunks
from embeddings import create_vector_store, search_similar_chunks
from groq_handler import get_groq_answer

# ── Hardcoded API Key ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS (pastel palette from reference image) ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --pink:       #F2A8B8;
    --peach:      #F7C9A8;
    --lavender:   #C9A8D4;
    --mint:       #A8D4C2;
    --sky:        #A8C9E8;
    --cream:      #FDF6F0;
    --soft-white: #FEFCFA;
    --text-dark:  #3D2E2E;
    --text-mid:   #6B5050;
    --text-light: #9E7E7E;
    --card-bg:    rgba(255,255,255,0.72);
    --glass:      rgba(255,255,255,0.55);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FDE8E8 0%, #FCF0E8 25%, #EEE8F8 55%, #E8F4EE 80%, #E8F0FC 100%) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-dark);
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(242,168,184,0.25) 0%, rgba(201,168,212,0.2) 50%, rgba(168,201,232,0.25) 100%) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(242,168,184,0.3);
}
[data-testid="stSidebar"] * { color: var(--text-dark) !important; }

/* ── Main hero title ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #C97A8A, #9A6CB4, #6A9FC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    color: var(--text-mid);
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
}

/* ── Feature cards ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.feature-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 1.4rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 4px 24px rgba(180,140,160,0.12);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(180,140,160,0.2);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.feature-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 0.3rem;
}
.feature-desc { font-size: 0.8rem; color: var(--text-light); line-height: 1.4; }

/* ── Steps banner ── */
.steps-banner {
    background: var(--glass);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    border: 1px solid rgba(255,255,255,0.75);
    box-shadow: 0 2px 16px rgba(180,140,160,0.1);
    margin: 1.5rem 0;
}
.steps-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--text-dark);
    margin-bottom: 1rem;
    font-weight: 600;
}
.steps-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.step-pill {
    background: linear-gradient(135deg, rgba(242,168,184,0.4), rgba(201,168,212,0.4));
    border-radius: 50px;
    padding: 0.45rem 1rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-dark);
    border: 1px solid rgba(255,255,255,0.7);
    white-space: nowrap;
}
.step-arrow { color: var(--text-light); font-size: 1rem; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: var(--card-bg) !important;
    backdrop-filter: blur(12px);
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.8) !important;
    box-shadow: 0 2px 12px rgba(180,140,160,0.1) !important;
    margin-bottom: 0.8rem !important;
    padding: 0.2rem 0.5rem !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessageContent"] {
    color: var(--text-dark) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px solid rgba(201,168,212,0.5) !important;
    border-radius: 16px !important;
    color: var(--text-dark) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--lavender) !important;
    box-shadow: 0 0 0 3px rgba(201,168,212,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--pink), var(--lavender)) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.5rem !important;
    box-shadow: 0 3px 14px rgba(201,168,212,0.4) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(201,168,212,0.5) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px solid rgba(201,168,212,0.4) !important;
    border-radius: 12px !important;
    color: var(--text-dark) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.65) !important;
    border: 2px dashed rgba(201,168,212,0.6) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-mid) !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    fill: var(--lavender) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, var(--pink), var(--lavender)) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
}
[data-testid="stFileUploaderDropzone"] button p {
    color: white !important;
}
[data-testid="stFileUploaderFile"] {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 12px !important;
    color: var(--text-dark) !important;
}
[data-testid="stFileUploaderFile"] * {
    color: var(--text-dark) !important;
}

/* ── Sidebar headers ── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.sidebar-logo .logo-icon { font-size: 2.8rem; }
.sidebar-logo .logo-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #C97A8A, #9A6CB4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Active PDF badge ── */
.pdf-badge {
    background: linear-gradient(135deg, rgba(168,212,194,0.4), rgba(168,201,232,0.4));
    border-radius: 14px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.75);
    font-size: 0.85rem;
    color: var(--text-dark);
    word-break: break-word;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,168,212,0.4); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "vector_store"  not in st.session_state: st.session_state.vector_store  = None
if "pdf_processed" not in st.session_state: st.session_state.pdf_processed = False
if "pdf_name"      not in st.session_state: st.session_state.pdf_name      = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">📄</div>
        <div class="logo-text">Chat with PDF</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**📂 Upload your PDF**")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Reading your PDF..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    text = extract_text_from_pdf(temp_path)
                    if not text.strip():
                        st.error("Could not extract text. Try a different PDF.")
                    else:
                        chunks = split_into_chunks(text)
                        vector_store = create_vector_store(chunks)
                        st.session_state.vector_store  = vector_store
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name      = uploaded_file.name
                        st.session_state.chat_history  = []
                        os.remove(temp_path)
                        st.success(f"✅ Ready! {len(chunks)} sections indexed.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    if st.session_state.pdf_processed:
        st.markdown(f"""
        <div class="pdf-badge">
            📄 <strong>Active PDF</strong><br>{st.session_state.pdf_name}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑️ Clear & Upload New", use_container_width=True):
            st.session_state.chat_history  = []
            st.session_state.vector_store  = None
            st.session_state.pdf_processed = False
            st.session_state.pdf_name      = ""
            st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-size:0.78rem; color: #9E7E7E; line-height:1.7;">
    💡 <strong>Tips</strong><br>
    • Upload any PDF up to 200MB<br>
    • Ask specific questions<br>
    • Try "Summarize this PDF"<br>
    • Ask "What are the key points?"
    </div>
    """, unsafe_allow_html=True)

# ── Main Area ──────────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:

    # Hero
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 1rem 1rem 1rem;">
        <div class="hero-title">Chat with any PDF ✨</div>
        <div class="hero-sub">Upload a document, ask anything — get instant answers powered by Groq AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Textbooks</div>
            <div class="feature-desc">Chat with your ECE, CS, or any study material instantly</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Research Papers</div>
            <div class="feature-desc">Understand complex research without reading every line</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <div class="feature-title">Project Reports</div>
            <div class="feature-desc">Summarize, extract key points, and review quickly</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Smart Summaries</div>
            <div class="feature-desc">Get concise summaries of any document in seconds</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">❓</div>
            <div class="feature-title">Q&A Mode</div>
            <div class="feature-desc">Ask concept questions and get answers from the doc</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌸</div>
            <div class="feature-title">Study Smarter</div>
            <div class="feature-desc">Prepare for exams by chatting with your notes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How to use
    st.markdown("""
    <div class="steps-banner">
        <div class="steps-title">✦ How to use</div>
        <div class="steps-row">
            <div class="step-pill">📂 Upload PDF</div>
            <div class="step-arrow">→</div>
            <div class="step-pill">🚀 Click Process PDF</div>
            <div class="step-arrow">→</div>
            <div class="step-pill">💬 Ask your question</div>
            <div class="step-arrow">→</div>
            <div class="step-pill">✨ Get instant answers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1rem 0;">
        <span style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700;
                     background:linear-gradient(135deg,#C97A8A,#9A6CB4);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                     background-clip:text;">
            💬 Chatting with: {st.session_state.pdf_name}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🌸"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="✨"):
                st.write(message["content"])

    # Suggested questions (only when no chat yet)
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="margin: 1rem 0 0.5rem 0; font-size:0.85rem; color:#9E7E7E; font-weight:500;">
            ✦ Try asking...
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        suggestions = [
            "📝 Summarize this PDF",
            "🔑 What are the key points?",
            "❓ What is this document about?"
        ]
        for col, suggestion in zip([col1, col2, col3], suggestions):
            with col:
                if st.button(suggestion, use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    relevant_chunks = search_similar_chunks(suggestion, st.session_state.vector_store)
                    try:
                        answer = get_groq_answer(suggestion, relevant_chunks, GROQ_API_KEY, [])
                    except Exception as e:
                        answer = f"⚠️ Error: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

    # Chat input
    user_question = st.chat_input("Ask anything about your PDF... 🌸")

    if user_question:
        with st.chat_message("user", avatar="🌸"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking..."):
                try:
                    relevant_chunks = search_similar_chunks(user_question, st.session_state.vector_store)
                    answer = get_groq_answer(
                        user_question, relevant_chunks, GROQ_API_KEY,
                        st.session_state.chat_history[-6:]
                    )
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.write(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})