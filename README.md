# Nexus RAG — Hybrid Retrieval-Augmented Generation Q&A System

A hybrid RAG pipeline that combines keyword and semantic search to answer questions over a document set, wrapped in a Streamlit app with a live pipeline-stage visualizer and analytics dashboard.

## How it works

```
Query → BM25 keyword search ─┐
                              ├─→ Hybrid fusion → Cross-encoder re-ranking → Flan-T5 answer generation
Query → Dense vector search ─┘        (ms-marco-MiniLM)
```

1. **BM25 retrieval** — classic keyword search over the corpus
2. **Dense retrieval** — semantic search via vector embeddings
3. **Hybrid fusion** — combines both result sets
4. **Cross-encoder re-ranking** — `ms-marco-MiniLM` re-scores candidates for precision
5. **Generation** — a local Flan-T5 model produces the final answer

## Features

- Single-file Streamlit app (`nexus_rag.py`)
- Pipeline-stage visualization — see each retrieval/ranking step as it runs
- Analytics dashboard for retrieval and generation performance
- Runs locally, no external LLM API required

## Tech stack

Python · Streamlit · BM25 · sentence embeddings · `ms-marco-MiniLM` cross-encoder · Flan-T5

## Getting started

```bash
git clone https://github.com/your-username/nexus-rag.git
cd nexus-rag
pip install -r requirements.txt
streamlit run nexus_rag.py
```

## Notes

Built and debugged for cross-platform stability, including resolving dependency-version conflicts (Python 3.11 vs. 3.14) and tuning generation parameters to fix truncated answers.

---
Part of Zeyad Fouad Talaat's AI engineering portfolio.
