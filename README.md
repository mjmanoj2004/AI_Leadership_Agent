# 🚀 AI Leadership Insight & Autonomous Decision Agent

Modular AI system with two distinct agents:

1. **Insight Agent** – RAG-based agent that answers factual questions from internal company documents (Chroma + sentence-transformers).
2. **Strategic Decision Agent** – LangGraph workflow:
   - Question Analysis  
   - Internal Research  
   - Knowledge Gap Detection  
   - Strategic Options Generation  
   - Risk Assessment  
   - Decision Synthesis (recommendations + confidence)

---

# 🧱 Tech Stack

- **Python 3.11
- **Backend:** FastAPI
- **UI:** Streamlit
- **LLM Orchestration:** LangChain + LangGraph
- **LLM:** HuggingFace (configurable free model)
- **Embeddings:** sentence-transformers
- **Vector DB:** Chroma (Hybrid Search: Semantic + BM25 + RRF)
- **Document Monitoring:** watchdog

---

# 📁 Project Structure

```
AI_Leadership_Agent/
│
├── config/
│   └── settings.py
│
├── src/
│   ├── api/
│   ├── agents/
│   ├── graph/
│   ├── prompts/
│   ├── retrieval/
│   ├── ingestion/
│   ├── llm/
│   └── models/
│
├── ui/
│   └── app.py
│
├── scripts/
│   └── ingest_documents.py
│
├── data/
│   ├── documents/
│   └── chroma_db/
│
├── requirements.txt
├── run_api.py
├── run_ui.py
└── README.md
```

---

# ⚙️ Setup

## 1️⃣ Create Virtual Environment (Python 3.11)

### Windows

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### Mac / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Configure Environment

Copy example file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```
HUGGINGFACE_HUB_TOKEN=your_token_here
```

Get your token from:
https://huggingface.co/settings/tokens

Enable:
- Read access
- Inference API / Inference Providers

Without a valid token:
- AI summaries will fail
- You may see “model unavailable” errors

---

## 4️⃣ Ingest Company Documents (Recommended)

Place PDF / DOCX / TXT files in:

```
data/documents/
```

Run:

```bash
python scripts/ingest_documents.py
```

If you change chunking strategy, re-run ingestion.

### Upload via UI

You can also upload documents through the Streamlit sidebar.  
Files are automatically saved and ingested.

---

# ▶️ Running the Application

## Start Backend (FastAPI)

```bash
uvicorn src.api.main:app --reload
```

API:
```
http://localhost:8000
```

Docs:
```
http://localhost:8000/docs
```

---

## Start UI (Streamlit)

```bash
streamlit run ui/app.py
```

UI:
```
http://localhost:8502
```

---

# 🧠 Using the System

### Modes

- **Auto** – Classifier chooses agent
- **Insight** – RAG-based factual answers
- **Strategic** – Full Decision Agent workflow

### Example Questions

Insight:
```
What is our revenue trend over the last 3 years?
```

Strategic:
```
Should we expand into Southeast Asia?
```

Outputs include:
- Answer
- Sources
- Reasoning trace (expandable)
- Risk chart (strategic mode)

---

# 🔌 API

## POST `/ask`

### Request

```json
{
  "question": "Your question here",
  "mode": "auto | insight | strategic"
}
```

### Response

```json
{
  "agent_type": "insight | strategic",
  "answer": "...",
  "sources": [
    { "content": "...", "metadata": {}, "score": 0.9 }
  ],
  "reasoning_trace": [
    { "node": "...", "summary": "..." }
  ],
  "risk_summary": {
    "options": [],
    "scores": {}
  }
}
```

---

# 🧩 Strategic Decision Agent (LangGraph)

Workflow:

1. Question Analyzer  
2. Internal Research  
3. Knowledge Gap Detection  
4. Strategic Reasoning  
5. Risk Assessment  
6. Decision Synthesis  

If context is insufficient, the graph loops back to Internal Research (configurable max iterations).

---

# 🔎 Hybrid Retrieval System

Retrieval uses **Hybrid Search**:

### 1️⃣ Semantic Search
- Chroma similarity search  
- Top 5 embedding matches  

### 2️⃣ Keyword Search
- BM25 over stored chunks  
- Top 5 keyword matches  

### 3️⃣ Fusion
- Reciprocal Rank Fusion (RRF, k=60)  
- Final Top 5 combined chunks  

Hybrid can be disabled:

```python
query_documents(use_hybrid=False)
```

---

# 🔧 Extensibility - For Production Grade

You can extend the system by:

- Adding new tools (web search, forecasting, financial analysis)
- Introducing memory or multi-agent collaboration
- AWS - S3 bucket for documents, API gateway, Rate limiting, 
- LLM - any better/latest/Fine tuned/ vLLM can be used
- Vector DB - Any other managed vector db is better option for production grade applications
- Token usage monitoring can be done
- Observability can be included e.g. langsmith
- Access - JWT token, Role based access to upload documents
- Gaurdrails
- Redis cache
- Async processing
- connection pooling
- Docker, K8s
---




