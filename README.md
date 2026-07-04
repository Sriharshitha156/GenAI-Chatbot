# 🎓 BVRIT Hyderabad FAQ Chatbot

**Veda** — A RAG-powered conversational AI assistant for BVRIT Hyderabad College of Engineering for Women. Built for GenAI & Agentic AI Engineering Day 4 Lab.

---

## 🌟 Features

- **RAG Architecture** — Retrieval-Augmented Generation with LangChain + ChromaDB
- **Grounded Responses** — Every answer is sourced from the official BVRIT knowledge base with inline citations
- **Free LLM** — Uses `openai/gpt-oss-20b:free` via OpenRouter (zero cost, no credits needed)
- **Smart Query Handling** — Recognizes BVRITH/bvrith/BVRIT Hyderabad as the same college, rejects queries about other institutions
- **Image Support** — Shows college entrance photo on relevant queries
- **Dual Theme** — Light and dark mode support with smooth animations
- **Friendly Persona** — Veda 🎓, a warm and helpful guide with conversational responses
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
    │ LLM          │  ◄── openai/gpt-oss-20b:free
    │ (Veda)       │      System prompt with grounding rules
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │ Response     │  ◄── Strip inline citations
    │ Post-Process │      Return (answer, citations, refused)
    └──────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Loader** | `Docx2txtLoader` (LangChain) | Load `.docx` knowledge base |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Chunk with size 500, overlap 100 |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Free local embeddings (1536 dim) |
| **Vector Store** | ChromaDB | Persistent storage with metadata |
| **LLM** | `openai/gpt-oss-20b:free` (OpenRouter) | Zero-cost generation with fallback chain |
| **UI** | Streamlit | Chat interface with st.chat_input |
| **Evaluation** | RAGAS + LLM-as-judge | 8-dimension test suite |

---

## 📂 Project Structure

```
TechVest-4/
├── app.py                      # Streamlit UI (main entry point)
├── src/
│   ├── ingest.py              # Document loading, chunking, embedding, indexing
│   ├── retriever.py           # Vector search with metadata filtering
│   ├── generator.py           # Grounded generation with Veda persona
│   └── evaluation.py          # 8-dimension test suite + RAGAS
├── assets/
│   └── college.png            # BVRIT entrance image
├── bvrit_knowledge_base.docx  # Official knowledge base (8 sections)
├── run_eval.py                # Run evaluation suite
├── .env                       # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenRouter API key (get one free at [openrouter.ai](https://openrouter.ai))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Manvitha-2905/GenAI-ChatBot-.git
cd GenAI-ChatBot-

# 2. Install dependencies
pip install streamlit langchain langchain-chroma langchain-huggingface \
            sentence-transformers openai python-dotenv chromadb

# 3. Create .env file
echo "OPENAI_API_KEY=your_openrouter_api_key_here" > .env
echo "OPENAI_BASE_URL=https://openrouter.ai/api/v1" >> .env
```

### Run the App

```bash
streamlit run app.py
```

The app will:
1. Load `bvrit_knowledge_base.docx`
2. Index 66 chunks into ChromaDB (first run only — persists after)
3. Launch at `http://localhost:8501`

---

## 🔑 Configuration

### API Keys

Add your OpenRouter API key to `.env`:

```env
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

> **Note:** The default model `openai/gpt-oss-20b:free` requires **zero credits** on OpenRouter. No billing needed.

### Retrieval Settings

Adjust in the Streamlit sidebar:
- **Top-K Results** — slider (3–10, default 5)
- **Section Filter** — dropdown (All, About, Departments, Admissions, Fees, Placements, etc.)

---

## 📚 Knowledge Base

The chatbot is grounded in `bvrit_knowledge_base.docx`, which contains 8 sections:

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

Run the 8-dimension test suite:

```bash
python run_eval.py
```

Generates `evaluation_report.txt` with:
- 20 test cases across all dimensions
- Pass/fail per dimension
- RAGAS metrics (faithfulness, answer relevancy, context precision, context recall)
- Weakest dimension + fix recommendation

---

## 🎨 UI Features

- **Dark & Light Mode** — Automatic theme switching based on system/Streamlit settings
- **Smooth Animations** — Message fade-in, input focus glow, button hover effects
- **Suggested Questions** — 3 FAQ chips on welcome screen
- **Citation Tags** — Green pills below each answer showing source sections
- **Image Display** — College entrance photo on "show me the college" queries
- **Conversation History** — Maintained in session state

---

## 🛡️ Safety Features

- **Grounding** — Answers ONLY from retrieved context, never from LLM training data
- **Privacy** — Refuses requests for personal student/staff info, internal security details
- **Injection Defense** — Rejects attempts to reveal system prompt or change persona
- **College Filtering** — Only answers BVRIT Hyderabad queries; rejects BVRITN, GRIET, IIT, etc.
- **No Guarantees** — Never promises individual outcomes like "you will get placed"

---

## 🤖 Veda's Personality

- Warm, encouraging, conversational tone
- Uses light emojis naturally (🎓 😊 🤔)
- Varies openers, ends with helpful follow-ups
- Friendly refusals with guidance to administration

**Example:**
> "Hmm, I don't have that specific detail handy! 🤔 For the most accurate info, you can reach the BVRIT Hyderabad team directly at **+91 40 4241 7773** — they'll be happy to help!"

---

## 📝 Example Queries

- "What B.Tech departments are offered at BVRIT?"
- "What is the fee structure for CSE?"
- "When was BVRITH established?"
- "Show me placement statistics"
- "Can you show me an image of the college?"
- "Where is BVRIT Hyderabad located?"
- "How do I apply for admissions?"

---

## 🐛 Troubleshooting

### "ChromaDB has 0 chunks"
Delete `chroma_db/` folder and restart — it will re-index.

```bash
rm -rf chroma_db
streamlit run app.py
```

### "OPENAI_API_KEY not set"
Create a `.env` file with your OpenRouter key (see Configuration above).

### "Model returned garbled output"
The app auto-falls back to `liquid/lfm-2.5-1.2b-instruct:free` → `llama-3.3-70b:free` → `gemma-4-26b:free` if the primary model fails.

---

## 📜 License

Built for educational purposes as part of GenAI & Agentic AI Engineering programme, Day 4 Lab.

---

## 👥 Contributors

- **Manvitha** — [GitHub](https://github.com/Manvitha-2905)

---

## 🙏 Acknowledgments

- **BVRIT Hyderabad** — For the official knowledge base
- **OpenRouter** — For free LLM access
- **LangChain & ChromaDB** — RAG framework and vector store
- **Streamlit** — Beautiful chat UI

---

**Veda 🎓 is ready to help! Ask anything about BVRIT Hyderabad.**
