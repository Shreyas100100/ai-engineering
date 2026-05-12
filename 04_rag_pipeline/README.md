# Full RAG Pipeline from Scratch

A complete, end-to-end RAG pipeline using Cohere — chunking a real document, embedding in batches, caching to disk, and generating grounded answers. No LangChain, no vector database, just NumPy and the Cohere SDK.

## What is this

This module brings everything together: chunking (from `03_chunking`) + RAG (from `02_rag`) into a single, production-shaped pipeline. It takes a full literary text (`julius_ceaser.txt` — Shakespeare's *Julius Caesar*), splits it into chunks, embeds them with Cohere, and answers natural language questions using Cohere's grounded chat API.

Unlike `02_rag` which used a small set of pre-split documents, this pipeline starts from a raw long-form document and handles the full ingestion flow.

## How it works

1. `julius_ceaser.txt` is loaded — the full text of Julius Caesar (~full play)
2. The text is split into chunks using **sentence boundary chunking** (max 500 chars per chunk)
3. Chunks are embedded in batches of 90 using Cohere's `embed-english-v3.0` model
4. Embeddings are cached to `embeddings.npy` so the document is only embedded once
5. At query time, the query is embedded using the same model (`input_type="search_query"`)
6. Cosine similarity ranks chunks by relevance to the query
7. Top-k chunks are passed as `documents` to Cohere's `co.chat()` — the model generates a grounded answer

## Key concepts

- **Sentence boundary chunking** — accumulates sentences until a size limit, then starts a new chunk. Preserves sentence meaning and produces coherent, embeddable units.
- **Batched embedding** — Cohere's API has limits per request. Chunks are sent in batches of 90, with a 2-second delay between batches to avoid rate limits.
- **Embedding cache** — embeddings are saved to `embeddings.npy` after the first run. Subsequent runs load from disk, skipping the API call entirely.
- **`input_type` distinction** — Cohere requires you to specify whether you're embedding a document (`search_document`) or a query (`search_query`). Using the wrong type degrades retrieval quality.
- **Cohere native RAG** — instead of manually building a prompt with context, `co.chat(documents=[...])` passes retrieved chunks directly to the model. Cohere handles grounding internally and the model cites its sources.
- **Cosine similarity** — same as previous modules: measures directional closeness between two embedding vectors. Score near 1 = highly relevant.

## What I learned building this

- Batching + rate limiting is a real concern at scale — even a medium-length document can exceed per-request limits
- The `input_type` parameter in Cohere embeddings is not optional — `search_document` vs `search_query` meaningfully changes the embedding space
- Cohere's `documents=` parameter in `co.chat()` is a clean abstraction over manual prompt engineering — the model knows to stay grounded in what's passed
- Caching embeddings is essential: re-embedding on every run wastes API calls and adds latency to a pipeline that only needs to do it once
- This is the closest thing in this series to how a real production RAG system works — just swap NumPy for a vector DB

## How to run

```bash
pip install -r requirements.txt
# create a .env file with your COHERE_API_KEY
python rag_pipeline.py
```

**Expected output:** document load size, chunk count, embedding progress (batches), then the query and Cohere's grounded answer.

## Stack

- [Cohere](https://cohere.com) — embedding model (`embed-english-v3.0`) + generation (`command-a-03-2025`)
- NumPy — cosine similarity + embedding storage
- python-dotenv — environment variable management

## How this fits in the series

| | 01 Semantic Search | 02 RAG | 03 Chunking | 04 RAG Pipeline |
|---|---|---|---|---|
| Input | Pre-split docs | Pre-split docs | One long doc | One long doc (play) |
| Chunking | Manual | Manual | 3 strategies compared | Sentence boundary |
| Embedding model | Voyage AI | Voyage AI | Gemini Embedding | Cohere |
| Generation | None | Gemini | None | Cohere (`co.chat`) |
| Batching | No | No | No | Yes (90 per batch) |
| Caching | Yes | Yes | No | Yes |
| Key question | Which doc matches? | What is the answer? | How to split? | Full pipeline end-to-end |
