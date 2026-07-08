# 🎓 BVRIT Hyderabad FAQ Chatbot

**Zia** — A RAG-powered conversational AI assistant for BVRIT Hyderabad College of Engineering for Women.

🌐 **Live Demo:** [https://genai-chatbot-fwac5ogv6qbjelv6x5axbn.streamlit.app](https://genai-chatbot-fwac5ogv6qbjelv6x5axbn.streamlit.app)

---

## 🌟 Features

- **RAG Architecture** — Retrieval-Augmented Generation with LangChain + ChromaDB
- **Grounded Responses** — Every answer is sourced from the official BVRIT knowledge base with inline citations
- **LLM** — Uses `gpt-4o-mini` via GitHub Models (Azure inference endpoint), with `gpt-4o` as fallback
- **Agentic Tools** — Fee calculator, date checker, and agentic loop via `agentic_chain.py`
- **Memory & History** — Persistent conversation history and user profiles across sessions
- **Observability** — LLM call logging via `src/observability.py`
- **Smart Query Handling** — Recognizes BVRITH/bvrith/BVRIT Hyderabad as the same college, rejects queries about other institutions
- **Image Support** — Shows college entrance photo on relevant queries
- **Dual Theme** — Light and dark mode support with smooth animations
- **Friendly Persona** — Zia 🎓, a warm and helpful guide with conversational responses
- **Safety Guardrails** — Refuses sensitive info requests, prompt injections, and out-of-scope queries
- **8-Dimension Evaluation** — Test suite covering Functional, Quality, Safety, Security, Robustness, Performance, Context, and RAGAS metrics

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ Greeting │  ◄── Intercept "hi", "hello" → instant response
    │ Check    │
    └────┬─────┘
         │ not greeting
    ┌────▼─────────┐
    │ Name         │  ◄── Normalize BVRITH → BVRIT
    │ Normalizer   │      Reject BVRITN, GRIET, etc.
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │ Retriever    │  ◄── Embed query → search ChromaDB
    │ (top-k=5)    │      Optional section filter
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │ Agentic      │  ◄── Tool calls: fee_calculator, date_checker
    │ Chain        │      Falls back to direct RAG if no tools needed
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │ LLM          │  ◄── gpt-4o-mini via GitHub Models
    │ (Zia)        │      System prompt with grounding rules
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │ Response     │  ◄── Strip inline citations
    │ Post-Process │      Save to conversation history
    └──────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Loader** | `Docx2txtLoader` (LangChain) | Load `.docx` knowledge base |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Chunk with size 500, overlap 100 |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Free local embeddings |
| **Vector Store** | ChromaDB | Persistent storage with metadata |
| **LLM** | `gpt-4o-mini` (GitHub Models) | Zero-cost generation with `gpt-4o` fallback |
| **Agentic Chain** | LangChain Tools | Fee calculator, date checker |
| **Memory** | JSON-based store | Persistent conversation history + user profiles |
| **Observability** | JSONL logging | LLM call tracing |
| **UI** | Streamlit | Chat interface with st.chat_input |
| **Evaluation** | RAGAS + LLM-as-judge | 8-dimension test suite |

---

## 📂 Project Structure

```
GenAI-ChatBot--main/
├── app.py                      # Streamlit UI (main entry point)
├── agentic_chain.py            # Tool-enabled agentic loop
├── tools.py                    # Fee calculator, date checker tools
├── src/
│   ├── ingest.py              # Document loading, chunking, embedding, indexing
│   ├── retriever.py           # Vector search with metadata filtering
│   ├── generator.py           # Grounded generation with Zia persona
│   ├── memory.py              # Conversation memory management
│   ├── history_store.py       # Persistent history store
│   ├── observability.py       # LLM call logging
│   └── evaluation.py          # 8-dimension test suite + RAGAS
├── assets/
│   └── college.png            # BVRIT entrance image
├── bvrit_knowledge_base.docx  # Official knowledge base (8 sections)
├── requirements.txt           # Python dependencies
├── run_eval.py                # Run evaluation suite
├── .env                       # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenRouter API key (free at [openrouter.ai](https://openrouter.ai))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sriharshitha156/GenAI-Chatbot.git
cd GenAI-Chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo OPENAI_API_KEY=your_openrouter_api_key_here > .env
echo OPENAI_BASE_URL=https://openrouter.ai/api/v1 >> .env
```

### Run the App

```bash
python -m streamlit run app.py
```

The app will:
1. Load `bvrit_knowledge_base.docx`
2. Index chunks into ChromaDB (first run only — persists after)
3. Launch at `http://localhost:8501`

---

## 🔑 Configuration

### .env file

```env
OPENAI_API_KEY=ghp_your_github_token_here
OPENAI_BASE_URL=https://models.inference.ai.azure.com
```

> Uses GitHub Models (free tier). Get a token at [github.com/settings/tokens](https://github.com/settings/tokens) — enable **Models** access.

### Retrieval Settings (Sidebar)

- **Top-K Results** — slider (3–10, default 5)
- **Section Filter** — dropdown (All, About, Departments, Admissions, Fees, Placements, etc.)

---

## 📚 Knowledge Base

`bvrit_knowledge_base.docx` contains 8 sections:

1. **About BVRITH** — History, vision, mission, accreditations
2. **Departments** — B.Tech, M.Tech, Ph.D programs
3. **Admissions** — Eligibility, entrance exams, process
4. **Fee Structure** — Tuition, hostel, scholarships
5. **Placements** — Top recruiters, packages, statistics
6. **Campus & Facilities** — Library, labs, hostel, sports
7. **Faculty** — Leadership, research areas
8. **Contact** — Address, phone, email, social media

---

## 🧪 Evaluation

```bash
python run_eval.py
```

Generates `evaluation_report.txt` covering 20 test cases across 8 dimensions: Functional, Quality, Safety, Security, Robustness, Performance, Context, and RAGAS metrics.

---

## 🛡️ Safety Features

- **Grounding** — Answers only from retrieved context
- **Privacy** — Refuses personal student/staff info requests
- **Injection Defense** — Rejects system prompt reveal attempts
- **College Filtering** — Only answers BVRIT Hyderabad queries

---

## 🐛 Troubleshooting

### "ChromaDB has 0 chunks"
```bash
# Delete the chroma_db folder and restart
rm -rf chroma_db
python -m streamlit run app.py
```

### "OPENAI_API_KEY not set"
Create a `.env` file with your OpenRouter key (see Configuration above).

### ImportError / chromadb crash on startup
Pin compatible versions:
```bash
pip install chromadb==1.4.0 opentelemetry-sdk==1.27.0 opentelemetry-exporter-otlp-proto-grpc==1.27.0
pip install sentence-transformers
```

### Model returned garbled output
The app auto-falls back through: `gpt-oss-20b:free` → `liquid/lfm-2.5-1.2b-instruct:free` → `llama-3.3-70b:free` → `gemma-4-26b:free`

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push your code to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo, branch `main`, file `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   OPENAI_API_KEY = "ghp_your_github_token_here"
   OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
   ```
5. Click **Deploy** — your app gets a public URL like `https://yourname-chatbot.streamlit.app`

> **Note:** ChromaDB's vector store will be rebuilt on each cold start on Streamlit Cloud since the filesystem is ephemeral. The first message may take ~30 seconds while it re-indexes.

---

## 📝 Example Queries

- "What B.Tech departments are offered at BVRIT?"
- "What is the fee structure for CSE?"
- "When was BVRITH established?"
- "Show me placement statistics"
- "Can you show me an image of the college?"
- "How do I apply for admissions?"

---

## 📜 License

Built for educational purposes as part of the GenAI & Agentic AI Engineering programme.

---

## 🙏 Acknowledgments

- **BVRIT Hyderabad** — For the official knowledge base
- **OpenRouter** — For free LLM access
- **LangChain & ChromaDB** — RAG framework and vector store
- **Streamlit** — Chat UI

---

**Zia 🎓 is ready to help! Ask anything about BVRIT Hyderabad.**
