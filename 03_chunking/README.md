# Chunking Strategies for Semantic Search

A hands-on exploration of how different text chunking strategies affect semantic search quality — using Gemini embeddings and cosine similarity, no vector database, no LangChain.

## What is this

Before you can do semantic search (like in `01_semantic_search`) or RAG (like in `02_rag`), you need to split your documents into chunks. But *how* you split matters — a bad chunking strategy can bury the answer in irrelevant text, or worse, cut a key sentence in half.

This module takes a single long document (`document.txt` — a backend engineering reference covering databases, indexing, caching, and more) and runs the same semantic search query across three different chunking strategies to compare which one surfaces the most relevant content.

## How it works

1. `document.txt` is loaded — a multi-chapter technical document (11,321 characters)
2. The document is split using three different strategies (see below)
3. Each set of chunks is embedded using **Gemini Embedding** (`gemini-embedding-2`)
4. A query is embedded the same way at search time
5. Cosine similarity is computed between the query and every chunk embedding
6. Top-k most similar chunks are ranked and printed for each strategy

## Key concepts

- **Chunking** — splitting a long document into smaller pieces so each piece can be embedded and compared independently. Embedding a 10-page document as one vector loses granularity; you want each chunk to represent one idea.
- **Fixed-size chunking** — splits text every N characters regardless of sentence boundaries. Fast and simple, but can cut sentences mid-way, losing context.
- **Sentence boundary chunking** — accumulates sentences until a size limit, then starts a new chunk. Preserves readability and coherence within each chunk.
- **Overlapping chunking** — like fixed-size, but consecutive chunks share a region of overlap. Prevents key information from being lost at chunk boundaries.
- **Chunk size tradeoff** — smaller chunks are more precise but noisier; larger chunks carry more context but may dilute the relevant signal.
- **Cosine similarity** — same as `01_semantic_search`: measures directional closeness between two embedding vectors. Score near 1 = highly relevant.

## What I learned

- Chunking strategy directly affects retrieval quality — the same query can return completely different results depending on how the document was split
- Fixed-size chunking is fast but fragile — it often cuts a sentence mid-way, and the embedding for that chunk is semantically weaker
- Sentence boundary chunking produces cleaner, more coherent chunks — embeddings better capture the intended meaning
- Overlapping chunks act as a safety net for information near boundaries — but increase the total number of chunks (and API calls)
- This is why real RAG pipelines invest heavily in chunking logic before anything else — garbage in, garbage out

## How to run

```bash
pip install -r requirements.txt
# create a .env file with your GEMINI_API_KEY
python chunking.py
```

**Expected output:** for each strategy, you'll see the total chunk count, the query, and the top-3 chunks ranked by similarity score.

## Stack

- [Google Gemini](https://ai.google.dev) — embedding model (`gemini-embedding-2`)
- NumPy — cosine similarity computation
- python-dotenv — environment variable management

## Difference from previous modules

| | 01 Semantic Search | 02 RAG | 03 Chunking |
|---|---|---|---|
| Input | Pre-split documents | Pre-split documents | One long document |
| Focus | Search + ranking | Search + LLM answer | How to split before search |
| LLM | None | Gemini (generation) | None |
| Key question | Which doc matches? | What is the answer? | How to split for best retrieval? |
| Embedding model | Voyage AI | Voyage AI | Gemini Embedding |
