# RAG Assistant — Retrieval-Augmented Generation over PDFs

A full-stack **document Q&A system** that lets you upload any PDF, embed it locally with a sentence-transformer, store vectors in ChromaDB, and ask questions answered by **Groq's Llama 3.1** — all grounded strictly in the uploaded document.

> **Stack**: LangChain · HuggingFace (bge-small-en-v1.5) · ChromaDB · Groq (Llama 3.1 8B Instant) · Streamlit

---

## Table of Contents

- [Demo](#demo)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Ingestion Pipeline](#1-ingestion-pipeline)
  - [Retrieval Pipeline](#2-retrieval-pipeline)
  - [Generation Pipeline](#3-generation-pipeline)
- [Retrieval Strategies (Lab)](#retrieval-strategies-lab)
- [Document Loaders & Splitters (Lab)](#document-loaders--splitters-lab)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the App](#running-the-app)
- [Running the CLI](#running-the-cli)
- [Dependencies](#dependencies)

---

## Demo

```
Upload PDF  →  Embed (local)  →  Ask questions  →  Grounded LLM answer
```

The Streamlit UI has:
- A left sidebar for uploading and processing PDFs
- A main chat window with persistent conversation history
- A status panel showing chunk count and vector store state

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION (one-time)                      │
│                                                                  │
│  PDF Upload ──► PyPDFLoader ──► RecursiveCharacterTextSplitter   │
│                  (page-by-page)     chunk_size=1000, overlap=200 │
│                                          │                       │
│                               HuggingFaceEmbeddings              │
│                               (BAAI/bge-small-en-v1.5, CPU)      │
│                                          │                       │
│                               ChromaDB  (persist: ./chroma-db)   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     QUERY (every question)                       │
│                                                                  │
│  User Question                                                   │
│       │                                                          │
│       ▼                                                          │
│  MMR Retriever  ──►  ChromaDB  (k=10, fetch_k=30, λ=0.5)        │
│  (Max Marginal Relevance — relevant + diverse chunks)            │
│       │                                                          │
│       ▼                                                          │
│  Context Assembly  (join top-10 page_content strings)            │
│       │                                                          │
│       ▼                                                          │
│  ChatPromptTemplate  (system: answer only from context)          │
│       │                                                          │
│       ▼                                                          │
│  ChatGroq — llama-3.1-8b-instant  (temperature=0.7)             │
│       │                                                          │
│       ▼                                                          │
│  Answer  ──►  Streamlit Chat UI  /  Terminal                     │
└──────────────────────────────────────────────────────────────────┘
```

**Key design choice**: embeddings run **entirely locally** on CPU (no API needed) while generation is offloaded to **Groq's ultra-fast inference API** — giving low-cost, low-latency answers.

---

## Project Structure

```
RAG/
│
├── app.py                    # ★ Main Streamlit web app (upload + chat)
├── RAG.py                    # ★ CLI version (assumes DB already exists)
├── create_database.py        # One-shot indexer for a fixed local PDF
│
├── Document_loader/          # Loader & splitter experiments
│   ├── pdf_loader.py         #   Load PDF → summarize with Groq
│   ├── web_loader.py         #   Scrape URL with WebBaseLoader
│   ├── Token_splittting.py   #   TokenTextSplitter demo
│   └── Character_splittting.py  # CharacterTextSplitter demo
│
├── Retriever/                # Retrieval strategy experiments
│   ├── datasource.py         #   WikipediaRetriever (external knowledge)
│   └── retrieval_strategy/
│       ├── similarity_search.py          # Cosine similarity retrieval demo
│       ├── mmr_max_marginal_relevancy.py # MMR retrieval demo
│       └── multi_query.py               # Multi-query LLM expansion demo
│
├── chroma-db/                # ← auto-generated; persisted vector store
├── requirements.txt          # pip dependencies
├── environment.yml           # conda environment (Python 3.11, CPU PyTorch)
└── .env                      # API keys (never committed)
```

---

## How It Works

### 1. Ingestion Pipeline

**In `app.py`** (dynamic, per upload):

```python
# 1. Load PDF pages
loader = PyPDFLoader(tmp_path)
data = loader.load()

# 2. Split into overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(data)

# 3. Embed each chunk locally
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)

# 4. Store in ChromaDB on disk
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)
```

**`RecursiveCharacterTextSplitter`** splits on `\n\n`, `\n`, ` `, `` in order — keeping semantically related text together. The `chunk_overlap=200` ensures context is not cut off at boundaries.

**`bge-small-en-v1.5`** (BAAI) is a compact but high-quality English embedding model (~33M parameters), well-suited for CPU inference.

---

### 2. Retrieval Pipeline

**MMR (Max Marginal Relevance)** retriever:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
)
```

- `fetch_k=30`: first fetches 30 candidates by similarity
- `k=10`: then picks 10 that are both **relevant** and **diverse** from each other
- `lambda_mult=0.5`: balances similarity vs diversity (0 = max diversity, 1 = pure similarity)

This prevents the context window from being filled with near-duplicate chunks when a topic is repeated across a document.

---

### 3. Generation Pipeline

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
     Use ONLY the provided context to answer the question.
     If the answer is not present in the context,
     say: I could not find the answer in the document."""),
    ("human", "Question: {question}\nContext: {context}")
])

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])
final_prompt = prompt_template.invoke({"context": context, "question": query})
response = llm.invoke(final_prompt)
```

The **system prompt strictly grounds** the LLM — it will not fabricate information outside the document. If the answer is not in the retrieved chunks, it says so explicitly.

---

## Retrieval Strategies (Lab)

Three strategies are demonstrated side-by-side in `Retriever/retrieval_strategy/`:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| **Similarity Search** | Embeds query → cosine distance → top-k | Simple, specific questions |
| **MMR** | Similarity + diversity penalty on results | Documents with repeated/overlapping content |
| **Multi-Query** | LLM generates N query variants → merge results → deduplicate | Broad, vague, or ambiguous questions |

**Main app uses MMR** — the best general-purpose choice for arbitrary PDF content.

---

## Document Loaders & Splitters (Lab)

`Document_loader/` contains standalone demos:

| File | Demonstrates |
|------|-------------|
| `pdf_loader.py` | `PyPDFLoader` + Groq summarization (no vector DB) |
| `web_loader.py` | `WebBaseLoader` to scrape a URL |
| `Token_splittting.py` | `TokenTextSplitter` — splits by token count |
| `Character_splittting.py` | `CharacterTextSplitter` — splits on `\n` separator |

---

## Setup & Installation

### Option A — pip (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/AfnanAjmal/RAG.git
cd RAG

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install streamlit langchain-chroma langchain-groq
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate rag-env
pip install streamlit langchain-chroma langchain-groq
```

> **Python 3.11** is required (specified in `environment.yml`).

---

## Environment Variables

Create a `.env` file in the project root (never commit this):

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

- **`GROQ_API_KEY`**: get free at [console.groq.com](https://console.groq.com)
- **`HUGGINGFACEHUB_API_TOKEN`**: get at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (needed only for HuggingFace Hub API calls; local embedding inference does not require it)

---

## Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501):

1. Click **"Choose a PDF file"** in the sidebar and upload a PDF.
2. Click **"⚡ Process & Embed"** — the PDF is chunked, embedded, and stored in `chroma-db/`.
3. Type any question in the chat box.
4. The assistant retrieves relevant chunks and answers using only the document content.
5. Use **"🗑️ Clear Chat"** to reset conversation history (vector store is preserved).

---

## Running the CLI

First, build the database (or use the Streamlit app to create `chroma-db/`):

```bash
# Optional: index the hardcoded PDF
python create_database.py
```

Then run the interactive terminal Q&A:

```bash
python RAG.py
# Type your question and press Enter
# Type 0 to exit
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 0.3.7 | Core RAG orchestration |
| `langchain-community` | 0.3.7 | Document loaders, retrievers |
| `langchain-huggingface` | 0.1.2 | HuggingFace embeddings integration |
| `langchain-chroma` | latest | Chroma vector store integration |
| `langchain-groq` | latest | Groq LLM integration |
| `chromadb` | 0.5.18 | Local vector database |
| `sentence-transformers` | 3.3.1 | bge-small-en-v1.5 embedding model |
| `torch` | 2.2.2 | PyTorch backend for embeddings |
| `streamlit` | latest | Web UI |
| `python-dotenv` | 1.0.1 | `.env` file loading |
| `huggingface-hub` | 0.26.2 | Model downloading |

---

## License

MIT
