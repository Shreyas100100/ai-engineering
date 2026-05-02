# RAG (Retrieval-Augmented Generation) from Scratch

A minimal RAG pipeline built from scratch using Voyage AI embeddings, 
cosine similarity, and Gemini — no LangChain, no vector database, just NumPy.

## What is this

An extension of semantic search that doesn't just *find* the most relevant document — 
it feeds that context into an LLM and generates a natural language answer.

RAG = Retrieval (semantic search) + Augmented Generation (LLM with context).

## How it works

1. Documents are embedded using Voyage AI and cached to disk (`embeddings.npy`)
2. At query time, the query is embedded using the same model
3. Cosine similarity ranks documents by relevance
4. The top-k most relevant documents are retrieved
5. A prompt is built: `context (retrieved docs) + question (query)`
6. Gemini generates a grounded answer based only on the retrieved context

## Key concepts

- **Retrieval** — same semantic search from `01_semantic_search`: embed, compare, rank
- **Augmented Generation** — instead of returning raw documents, pass them as context to an LLM
- **Prompt engineering** — the prompt explicitly instructs the LLM to answer based on the provided context, reducing hallucination
- **top_k** — controls how many documents are retrieved and injected into the prompt

## What I learned building this

- RAG grounds the LLM in real data — the model can't hallucinate facts that aren't in the context
- The quality of retrieval directly affects the quality of the answer — bad search = bad answer
- Prompt structure matters: telling the model "answer based on this context" keeps it focused
- This is the foundation of most production AI assistants — the vector DB is just a scaled-up version of this NumPy approach

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env        # add VOYAGE_API_KEY and GEMINI_API_KEY
python rag.py
```

## Stack

- [Voyage AI](https://voyageai.com) — embedding model (`voyage-4`, `voyage-4-lite`)
- [Google Gemini](https://ai.google.dev) — LLM for generation (`gemini-3-flash-preview`)
- NumPy — cosine similarity + vector storage
- python-dotenv — environment variable management

## Difference from 01_semantic_search

| | Semantic Search | RAG |
|---|---|---|
| Output | Ranked document list | Natural language answer |
| LLM | None | Gemini |
| Use case | Find relevant docs | Answer questions from docs |
