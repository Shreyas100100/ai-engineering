# Semantic Search from Scratch

A semantic search engine built from scratch using Voyage AI embeddings 
and cosine similarity — no vector database, no LangChain, just NumPy.

## What is this

A search system that finds the most semantically relevant document 
for a given query. Unlike keyword search which matches exact words, 
this understands *meaning* — so "fast car" can match "quick automobile".

## How it works

1. Each document is converted into a vector (embedding) using Voyage AI
2. The search query is embedded the same way at query time
3. Cosine similarity is calculated between the query vector and every document vector
4. Documents are ranked — similarity closest to 1 = most relevant

## Key concepts

- **Embedding** — a list of numbers representing the meaning of a sentence, 
  produced by an embedding model. Similar meanings produce similar vectors.
- **Cosine similarity** — measures the angle between two vectors. 
  Score of 1 = identical meaning, 0 = unrelated.
- **Unit vectors** — Voyage AI returns normalized vectors (magnitude = 1), 
  so dot product alone equals cosine similarity.
- **Embedding cache** — embeddings are saved to disk (`embeddings.npy`) 
  so documents are only embedded once, not on every run.

## What I learned building this

- Keyword search fails for synonyms — semantic search solves this by 
  working in vector space, not word space
- The API is stateless — you send text, get back numbers. 
  The "understanding" is baked into the model weights.
- Cosine similarity is about direction, not distance — 
  two vectors pointing the same way = same meaning
- Re-embedding on every run wastes API calls — always cache

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env        # add your VOYAGE_API_KEY
python search.py
```

## Stack

- [Voyage AI](https://voyageai.com) — embedding model
- NumPy — cosine similarity + vector storage
- python-dotenv — environment variable management