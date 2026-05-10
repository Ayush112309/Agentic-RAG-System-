# 🧠 Agentic RAG System

> A production-quality, agentic Retrieval-Augmented Generation (RAG) chatbot that ingests your documents and answers questions grounded strictly in your data — no hallucinations.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green?logo=chainlink)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Project Overview

The **Agentic RAG System** is a lightweight yet powerful AI agent that:

1. **Ingests** documents from PDF, TXT, and CSV sources
2. **Processes** and chunks text using a recursive splitting strategy
3. **Embeds** chunks using OpenAI or local sentence-transformers
4. **Stores** embeddings persistently in ChromaDB
5. **Retrieves** the most relevant context using semantic similarity search
6. **Generates** accurate, grounded answers using GPT-4o-mini (or local Llama3)
7. **Cites** the source documents for every answer
8. **Remembers** conversation history across turns

If an answer cannot be found in the documents, the system explicitly says:
> *"I could not find relevant information in the provided documents."*

---

## ✨ Features

### Core Features
- ✅ Multi-format document ingestion (PDF, TXT, CSV)
- ✅ Recursive character text chunking with configurable size and overlap
- ✅ OpenAI Embeddings + local sentence-transformers fallback
- ✅ ChromaDB persistent vector storage
- ✅ Semantic retrieval with relevance scoring
- ✅ Strict hallucination prevention via context-grounding
- ✅ Source attribution on every answer
- ✅ Graceful "I don't know" responses when context is unavailable

### Bonus / Agentic Features
- ✅ LangChain OpenAI Functions agent with explicit tool calling
- ✅ Conversation memory (windowed buffer)
- ✅ Calculator tool for mathematical queries
- ✅ Web search placeholder (ready for SerpAPI/Tavily integration)
- ✅ Beautiful Streamlit web UI
- ✅ CLI interface for terminal use
- ✅ Modular, OOP, PEP8 codebase with full type hints
- ✅ Comprehensive test suite (pytest)
- ✅ Full logging to file and console

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    AGENTIC RAG SYSTEM                          │
│                                                                │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐  │
│  │  Document    │    │   Text      │    │   Embedding      │  │
│  │  Loader      │───▶│   Chunker   │───▶│   Manager        │  │
│  │  (PDF/TXT/   │    │  (Recursive │    │  (OpenAI /       │  │
│  │   CSV)       │    │   Split)    │    │   SentTrans)     │  │
│  └──────────────┘    └─────────────┘    └────────┬─────────┘  │
│                                                  │            │
│                                         ┌────────▼─────────┐  │
│                                         │   VectorStore    │  │
│                                         │   (ChromaDB)     │  │
│                                         │   Persistent     │  │
│                                         └────────┬─────────┘  │
│                                                  │            │
│  ┌──────────────┐    ┌─────────────┐    ┌────────▼─────────┐  │
│  │  Memory      │    │    RAG      │    │   RAG Retriever  │  │
│  │  Manager     │◀──▶│   Agent     │◀───│   (Similarity    │  │
│  │  (Buffer     │    │  (LangChain │    │    Search +      │  │
│  │   Window)    │    │   OpenAI    │    │    Scoring)      │  │
│  └──────────────┘    │   Functions)│    └──────────────────┘  │
│                      └──────┬──────┘                          │
│                             │ Tools                           │
│              ┌──────────────┼──────────────────┐             │
│              ▼              ▼                  ▼             │
│     ┌────────────┐  ┌─────────────┐  ┌───────────────┐      │
│     │ Document   │  │ Calculator  │  │ Web Search    │      │
│     │ Retrieval  │  │ Tool        │  │ (Placeholder) │      │
│     └────────────┘  └─────────────┘  └───────────────┘      │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Streamlit UI / CLI                          │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User uploads PDF/TXT/CSV
         │
         ▼
   DocumentLoader.load_file()
         │
         ▼
   TextChunker.split_documents()   ← chunk_size=800, overlap=150
         │
         ▼
   EmbeddingManager.get_embeddings()  ← text-embedding-3-small
         │
         ▼
   VectorStoreManager.add_documents()  ← ChromaDB persist
         │
         ▼
   [User asks question]
         │
         ▼
   RAGRetriever.retrieve()  ← cosine similarity, top-k=4
         │
         ▼
   RAGAgent.query()  ← inject context → GPT-4o-mini
         │
         ▼
   AgentResponse(answer, sources, chunks, score)
         │
         ▼
   Streamlit UI displays answer + citations + scores
```

---

## 📁 Project Structure

```
agentic-rag-system/
│
├── app/                          # Core application code
│   ├── ingestion/
│   │   ├── document_loader.py   # PDF, TXT, CSV loading
│   │   └── chunker.py           # Recursive text chunking
│   │
│   ├── embeddings/
│   │   └── embedding_manager.py # OpenAI / SentenceTransformers
│   │
│   ├── retrieval/
│   │   ├── vector_store.py      # ChromaDB management
│   │   └── retriever.py         # Similarity search + scoring
│   │
│   ├── agents/
│   │   ├── rag_agent.py         # Main LangChain agent
│   │   └── tools.py             # Retrieval, calculator, web search
│   │
│   ├── memory/
│   │   └── memory_manager.py    # Conversation buffer window
│   │
│   └── utils/
│       ├── config.py            # Environment-aware configuration
│       ├── logging_config.py    # Structured logging setup
│       └── pipeline.py          # High-level orchestration facade
│
├── data/
│   └── samples/                 # Sample documents for testing
│       ├── company_kb.txt
│       ├── rag_guide.txt
│       └── product_pricing.csv
│
├── vectorstore/                 # ChromaDB persistence (auto-created)
│   └── chroma_db/
│
├── logs/                        # Log files (auto-created)
│
├── tests/
│   └── test_pipeline.py         # Comprehensive pytest test suite
│
├── main.py                      # CLI entry point
├── streamlit_app.py             # Streamlit web UI
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM Framework | LangChain 0.2 |
| LLM | GPT-4o-mini (OpenAI) / Llama3 (Ollama) |
| Embeddings | OpenAI text-embedding-3-small / sentence-transformers |
| Vector DB | ChromaDB (local persistent) |
| Frontend | Streamlit |
| Document Parsing | PyPDFLoader, TextLoader, CSVLoader |
| Memory | ConversationBufferWindowMemory |
| Agent | LangChain OpenAI Functions Agent |
| Testing | pytest + pytest-cov |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key (or Ollama with llama3 for free local use)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-rag-system.git
cd agentic-rag-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Minimum required:
```env
OPENAI_API_KEY=sk-your-key-here
```

### 5. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ CLI Usage

```bash
# Ingest all documents in a directory
python main.py --ingest data/samples/

# Ingest a single file
python main.py --ingest data/samples/company_kb.txt

# Ask a single question
python main.py --query "What is the refund policy?"

# Start interactive chat session
python main.py --interactive

# Check system status
python main.py --status

# Reset the vector store
python main.py --reset

# Set log level
python main.py --log-level DEBUG --interactive
```

---

## 💬 Example Queries

After ingesting the sample documents:

| Query | Expected Behavior |
|-------|------------------|
| "What is the refund policy?" | Returns 30-day policy from company_kb.txt |
| "What does CloudAI Pro cost?" | Returns $2,000/month from product_pricing.csv |
| "Explain how RAG works" | Retrieves step-by-step from rag_guide.txt |
| "What is 2500 * 12?" | Uses calculator tool, returns 30000 |
| "What is the weather today?" | Returns "I could not find..." (not in docs) |

---

## 🔧 Configuration Reference

All settings can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI LLM + embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | LLM to use |
| `LLM_TEMPERATURE` | `0.1` | Creativity (0=factual, 1=creative) |
| `EMBEDDING_BACKEND` | `openai` | `openai` or `sentence_transformers` |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between chunks |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve |
| `RETRIEVAL_THRESHOLD` | `0.30` | Min similarity score (0–1) |
| `MEMORY_WINDOW` | `10` | Conversation turns to remember |
| `USE_AGENT_MODE` | `true` | Full agent vs simple chain |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=html

# Run specific test class
pytest tests/test_pipeline.py::TestDocumentLoader -v
```

---

## 🚢 Deployment

### Option A: Local / Development
```bash
streamlit run streamlit_app.py
```

### Option B: Docker

```bash
# Build image
docker build -t agentic-rag .

# Run
docker run -p 8501:8501 --env-file .env agentic-rag
```

### Option C: Streamlit Community Cloud
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set `OPENAI_API_KEY` in the Streamlit secrets manager
5. Deploy!

---

## ⚠️ Limitations

1. **Context window**: Very long documents may exceed LLM context limits
2. **No table understanding**: Embedded CSVs lose structural context
3. **English-optimized**: Non-English documents may have lower retrieval quality
4. **No image support**: Image content in PDFs is not extracted
5. **Web search stub**: Real-time web search is not yet implemented
6. **No access control**: All uploaded documents are shared in one collection

---

## 🔮 Future Improvements

- [ ] Add Pinecone/Weaviate for cloud-scale vector storage
- [ ] Implement real web search (Tavily, SerpAPI)
- [ ] Add re-ranking with cross-encoder models
- [ ] Support multi-modal documents (images, tables)
- [ ] Add user authentication and per-user document isolation
- [ ] Implement streaming responses in Streamlit
- [ ] Add RAGAS evaluation pipeline
- [ ] Support Google Drive / Notion ingestion
- [ ] Add response confidence scoring via self-consistency
- [ ] Deploy with auto-scaling on AWS/GCP

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Write tests for new functionality
4. Ensure `pytest` passes
5. Submit a PR with a clear description

---

*Built as part of the AI Internship Assignment — Agentic RAG System*