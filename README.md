<div align="center">

# 📄 Document Intelligence — AI-Powered PDF Analysis

> A production-ready RAG system that transforms static PDF documents into interactive, conversational knowledge bases using agentic AI workflows.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🎯 Overview

**Document Intelligence** is an end-to-end system that allows users to upload PDF documents and interact with them through natural language conversation. It combines **Retrieval-Augmented Generation (RAG)**, **agentic tool orchestration**, and **streaming responses** to deliver accurate, context-aware answers grounded in the user's own documents.

Unlike simple "chat with your PDF" demos, this project implements a **multi-step agentic pipeline** where the AI autonomously decides when to discover document structure, extract specific data points, and summarize conversations — mimicking how a human analyst would approach an unfamiliar document.

### ✨ Key Capabilities

- **Intelligent Document Discovery** — The agent autonomously analyzes uploaded documents to identify their purpose, key entities, trackable data points, and tone
- **Structured Data Extraction** — Extracts precise values, dates, and clauses based on discovery insights using targeted vector search
- **Streaming Conversation** — Real-time token-by-token response streaming via Server-Sent Events for a responsive chat experience
- **Persistent Memory** — Conversation history is checkpointed in PostgreSQL, enabling multi-turn context across sessions
- **Automatic Summarization** — Long conversations are condensed into concise summaries to maintain relevant context within the LLM's window
- **Multi-Provider LLM Support** — Seamlessly switch between Google Gemini, OpenAI, and Ollama (local models) via configuration
- **Multi-User Isolation** — Documents and conversations are scoped per user, ensuring data privacy

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (:8501)                            │
│                   Chat Interface + File Upload                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  HTTP / Streaming
┌───────────────────────────────▼─────────────────────────────────────────┐
│                        FastAPI Backend (:8000)                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐                             │
│  │ /chat    │  │ /upload  │  │ /health    │                             │
│  └────┬─────┘  └────┬─────┘  └────────────┘                             │
│       │             │                                                   │
│  ┌────▼─────────────▼──────-────────────────────────────────────┐       │  
│  │                   LangGraph Agent                            │       │
│  │                                                              │       │
│  │   START → Chatbot ──(tool call?)──→ Tools → Chatbot          │       │
│  │               │                                    │         │       │
│  │               └──(no tools)──→ Summarizer → END  <─          │       │
│  │                                                              │       │
│  │   Tools:                                                     │       │
│  │   ├── 🔍 Discovery Tool (document understanding)             │       │
│  │   └── 📊 Extraction Tool (structured data retrieval)         │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────────┐
│                    PostgreSQL + pgvector (:5432)                        │
│  ┌─────────────────────┐  ┌──────────────────────────┐                  │
│  │   Vector Store      │  │   LangGraph Checkpoints  │                  │
│  │   (document chunks) │  │   (conversation memory)  │                  │
│  └─────────────────────┘  └──────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Workflow (LangGraph State Machine)

The core of the system is a **stateful, cyclic agent graph** built with LangGraph:

1. **Chatbot Node** — Invokes the LLM with conversation history and bound tools. The model autonomously decides whether to call a tool or respond directly.
2. **Tool Node** — Executes the selected tool (discovery or extraction) with user-scoped document retrieval.
3. **Summarizer Node** — When conversation history exceeds a threshold, compresses it into a summary to manage the context window efficiently.

This design enables the agent to perform **multi-hop reasoning**: discover a document's structure first, then extract specific data points in a follow-up tool call — all within a single user query.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM Orchestration** | LangGraph 1.0 | Stateful agent graph with conditional edges and cyclic tool loops |
| **RAG Framework** | LangChain 1.2 | Document loading, text splitting, retriever abstraction |
| **LLM Providers** | Google Gemini · OpenAI · Ollama | Multi-provider support with hot-swappable configuration |
| **Vector Database** | PostgreSQL + pgvector | Embedding storage with JSONB metadata filtering |
| **Checkpointing** | LangGraph Checkpoint (Postgres) | Persistent conversation memory across sessions |
| **Backend API** | FastAPI + Uvicorn | Async REST API with streaming response support |
| **Frontend** | Streamlit | Interactive chat interface with real-time streaming |
| **Containerization** | Docker Compose | Single-command deployment of all services |
| **Package Manager** | uv | Fast, reproducible dependency management with lockfile |
| **Code Quality** | Ruff | Linting and formatting (PEP 8, import sorting) |

---

## 📁 Project Structure

```
pdf-rag/
├── src/                            # Backend application
│   ├── main.py                     # FastAPI entry point + lifecycle management
│   ├── api/
│   │   ├── routers/
│   │   │   ├── chat.py             # Streaming chat endpoint
│   │   │   ├── upload.py           # PDF upload + ingestion endpoint
│   │   │   └── admin.py            # Ops: prompt reload, config status
│   │   └── schemas/
│   │       └── query.py            # Request/response models
│   ├── core/
│   │   └── settings.py             # Environment-based configuration
│   ├── infrastructure/
│   │   ├── database.py             # PostgreSQL + checkpointer initialization
│   │   └── retrievers.py           # User-scoped vector retriever factory
│   ├── services/
│   │   ├── agents/
│   │   │   └── agent_service.py    # Query analysis + event streaming
│   │   ├── graphs/
│   │   │   └── document_graph.py   # LangGraph state machine definition
│   │   ├── ingestion/
│   │   │   └── ingestion_service.py # PDF → chunks → embeddings pipeline
│   │   ├── nodes/
│   │   │   ├── chabot_node.py      # LLM invocation with tool binding
│   │   │   └── summarizer.py       # Conversation compression
│   │   ├── prompts/
│   │   │   ├── registry.py         # YAML-based prompt template loader
│   │   │   ├── nodes/              # Node prompt templates
│   │   │   └── tools/              # Tool prompt templates
│   │   ├── states/
│   │   │   └── graph_state.py      # Pydantic state schema
│   │   └── tools/
│   │       ├── discovery.py        # Document structure analysis tool
│   │       └── extraction.py       # Targeted data extraction tool
│   ├── shared/
│   │   └── schemas/
│   │       └── universal_discovery.py  # Structured output schema
│   └── utils/
│       ├── embedding_factory.py    # Multi-provider embedding resolver
│       ├── llm_factory.py          # Multi-provider LLM resolver
│       ├── log_wrapper.py          # Execution timing decorator
│       └── logging_config.py       # Centralized logging setup
├── ui/                             # Frontend application
│   ├── app.py                      # Streamlit entry point
│   ├── components/
│   │   ├── chat_interface.py       # Chat history + streaming renderer
│   │   └── sidebar.py             # Upload + status panel
│   ├── services/
│   │   └── api_client.py           # HTTP client for the backend API
│   └── styles/
│       └── custom.css              # UI customization
├── Dockerfile                      # Multi-stage build with uv
├── docker-compose.yaml             # Full stack: DB + API + UI
├── pyproject.toml                  # Project metadata + dependencies
└── uv.lock                        # Reproducible dependency lockfile
```

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- An LLM provider: **Ollama** (local, free) or an **API key** for Google Gemini / OpenAI

### 1. Clone the Repository

```bash
git clone https://github.com/Antonio-Jr/pdf-rag.git
cd pdf-rag
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your preferred LLM provider:

<details>
<summary><b>Ollama (Local — Free)</b></summary>

```env
LLM_API_KEY="ollama"
LLM_BASE_URL="http://host.docker.internal:11434"
LLM_PROVIDER="ollama"
LLM_MODEL_NAME="qwen3.5"
LLM_TEMPERATURE="0.3"
LLM_EMBEDDING_PROVIDER="ollama"
LLM_EMBEDDING_MODEL="qwen3-embedding"
```

> Make sure Ollama is running locally with the required models pulled.
</details>

<details>
<summary><b>Google Gemini (Cloud)</b></summary>

```env
LLM_API_KEY="your-google-api-key"
LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
LLM_PROVIDER="google_genai"
LLM_MODEL_NAME="gemini-2.5-flash"
LLM_TEMPERATURE="0.3"
LLM_EMBEDDING_PROVIDER="google_genai"
LLM_EMBEDDING_MODEL="text-embedding-004"
```
</details>

<details>
<summary><b>OpenAI (Cloud)</b></summary>

```env
LLM_API_KEY="sk-your-openai-key"
LLM_BASE_URL="https://api.openai.com/v1"
LLM_PROVIDER="openai"
LLM_MODEL_NAME="gpt-4o-mini"
LLM_TEMPERATURE="0.3"
LLM_EMBEDDING_PROVIDER="openai"
LLM_EMBEDDING_MODEL="text-embedding-3-small"
```
</details>

### 3. Launch

```bash
docker compose up --build
```

Once running:
- **API** → [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **Chat UI** → [http://localhost:8501](http://localhost:8501)
- **Health** → [http://localhost:8000/health](http://localhost:8000/health)

### 4. Usage

1. Open the **Chat UI** at `localhost:8501`
2. Upload one or more **PDF documents** via the sidebar
3. Ask questions — the agent will autonomously discover, extract, and answer

---

## 🔧 Development

### Local Setup (without Docker)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start a local PostgreSQL with pgvector
docker compose up db -d

# Run the API
uv run uvicorn src.main:app --reload --port 8000

# Run the UI (in another terminal)
uv run streamlit run ui/app.py
```

### Code Quality

```bash
# Lint
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format
uv run ruff format .
```

---

## 🧠 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph over LangChain agents** | Provides explicit, debuggable control flow with conditional edges and cycles, instead of opaque ReAct loops |
| **Two-phase tool strategy** | Discovery → Extraction mirrors how a human analyst approaches documents progressively |
| **YAML prompt templates** | Externalizes prompts from code, enabling hot-reload in production without redeployment |
| **Async-first architecture** | Every database call, LLM invocation, and API endpoint is fully asynchronous for maximum throughput |
| **User-scoped retrieval** | JSONB metadata filtering ensures multi-tenant document isolation at the vector store level |
| **Psycopg v3 pool** | Direct connection pooling for the checkpointer avoids SQLAlchemy overhead on high-frequency state writes |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built by [Antonio Jr](https://github.com/Antonio-Jr)** · Exploring the intersection of AI engineering, agentic systems, and production-grade software design.

</div>
