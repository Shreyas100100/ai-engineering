from google import genai
import numpy as np
import os
from dotenv import load_dotenv
 
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key = GEMINI_API_KEY)

def emebed_chunks(fixed_chunks):
    embeddings = []
    for chunk in fixed_chunks:
        chunk_embds = gemini_client.models.embed_content(model="gemini-embedding-2", contents=chunk).embeddings[0].values
        embeddings.append(chunk_embds)
    return embeddings

def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

def search(query, chunk, chunk_embds, top_k = 3):
    query_embd = gemini_client.models.embed_content(model="gemini-embedding-2", contents=query).embeddings[0].values
    similarities = [cosine_similarity(query_embd, chunk_embd) for chunk_embd in chunk_embds]
    similarities = np.array(similarities)
 
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return ranked_indices

with open("document.txt", "r") as f:
    text = f.read()

print(f"Document loaded: {len(text)} characters")

def fixed_size_chunks(text, chunk_size = 500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def sentence_boundary_chunk(text, chunk_size = 500):
    sentence = text.split(". ")
    chunks = []
    current_chunk = ""
    for i in sentence:
        if len(current_chunk) + len(i) <= chunk_size:
            current_chunk += i + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = i + ". "
    chunks.append(current_chunk.strip())
    return chunks

def overlapping_chunks(text, chunk_size = 500, overlap = 100):
    chunks = []
    start = 0
    current_chunk = ""
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# fixed_chunks = overlapping_chunks(text, chunk_size=500, overlap=100)
fixed_chunks = sentence_boundary_chunk(text, chunk_size=500)

print(f"Total chunks: {len(fixed_chunks)}")

query = "which data structure is used in indexing ?"
print(f"\nQuery: {query}")
print(search(query, fixed_chunks, emebed_chunks(fixed_chunks), top_k=3))