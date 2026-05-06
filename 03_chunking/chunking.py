from google import genai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        chunk_embedding = gemini_client.models.embed_content(model="gemini-embedding-2", contents=chunk).embeddings[0].values
        embeddings.append(chunk_embedding)
    return embeddings


def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))


def search(query, chunks, chunk_embeddings, top_k=3):
    query_embedding = gemini_client.models.embed_content(model="gemini-embedding-2", contents=query).embeddings[0].values
    similarities = [cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
    similarities = np.array(similarities)

    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return ranked_indices, similarities[ranked_indices]


def compare_strategy(strategy_name, chunks, query, top_k=3):
    print(f"Total chunks: {len(chunks)}")
    print(f"\nQuery: {query}")
    ranked_indices, similarities = search(query, chunks, embed_chunks(chunks), top_k=top_k)
    print(f"\nTop Ranked Chunks using {strategy_name}:")
    for idx, sim in zip(ranked_indices, similarities):
        print(f"Chunk Index: {idx}, Similarity Score: {sim:.4f}")


with open("document.txt", "r") as f:
    text = f.read()

print(f"Document loaded: {len(text)} characters")


def fixed_size_chunks(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks


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


def overlapping_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start+chunk_size]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


query = "which data structure is used in indexing ?"

# Fixed Size Chunking
fixed_chunks = fixed_size_chunks(text, chunk_size=500)
compare_strategy("Fixed Size Chunking", fixed_chunks, query)

# Overlapping Chunking
overlapped_chunks = overlapping_chunks(text, chunk_size=500, overlap=100)
compare_strategy("Overlapping Chunking", overlapped_chunks, query)

# Sentence Boundary Chunking
sentence_chunks = sentence_boundary_chunks(text, chunk_size=500)
compare_strategy("Sentence Boundary Chunking", sentence_chunks, query)