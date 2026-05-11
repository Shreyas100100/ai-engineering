import cohere
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY)

def sentence_boundary_chunks(text, chunk_size=500):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    chunks.append(current_chunk.strip())
    return chunks

def embed_chunks(chunks):
    embeddings = []
    batch_size = 90

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")

        response = co.embed(
            model="embed-english-v3.0",
            texts=batch,
            input_type="search_document",
            truncate="END"
        )
        embeddings.extend(response.embeddings)

        if i + batch_size < len(chunks):
            time.sleep(2)

    return embeddings


def embed_query(query):
    response = co.embed(
        model="embed-english-v3.0",
        texts=[query],
        input_type="search_query",
        truncate="END"
    )
    return response.embeddings[0]


def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))


def search(query, chunks, chunk_embeddings, top_k=3):
    query_embedding = embed_query(query)
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return ranked_indices

def generate(query, chunks, chunk_embeddings, top_k=3):
    ranked_indices = search(query, chunks, chunk_embeddings, top_k=top_k)
    
    documents = [{"text": chunks[i]} for i in ranked_indices]

    response = co.chat(
        model="command-a-03-2025",
        message=query,
        documents=documents,
        temperature=0.3
    )
    return response.text


with open("julius_ceaser.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded document: {len(text)} characters")

chunks = sentence_boundary_chunks(text, chunk_size=500)
print(f"Total chunks: {len(chunks)}")

EMBEDDINGS_FILE = "embeddings.npy"

if os.path.exists(EMBEDDINGS_FILE):
    print("Loading embeddings from cache...")
    chunk_embeddings = np.load(EMBEDDINGS_FILE)
else:
    print("Embedding chunks with Cohere...")
    chunk_embeddings = embed_chunks(chunks)
    np.save(EMBEDDINGS_FILE, chunk_embeddings)
    print("Embeddings saved to cache.")

query = "Why was Julius Caesar said et tu brute and to whom was he referring to?"
print(f"\nQuery: {query}")
answer = generate(query, chunks, chunk_embeddings, top_k=3)
print(f"Answer: {answer}")