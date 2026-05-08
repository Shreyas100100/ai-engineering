from google import genai
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# --- Chunking ---

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


# --- Embedding ---

def embed_chunks(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = gemini_client.models.embed_content(
            model="gemini-embedding-2", contents=chunk
        ).embeddings[0].values
        embeddings.append(chunk_embedding)
        if (i + 1) % 90 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks, pausing to avoid rate limit...")
            time.sleep(62)
    return embeddings


# --- Search ---

def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))


def search(query, chunks, chunk_embeddings, top_k=3):
    query_embedding = gemini_client.models.embed_content(
        model="gemini-embedding-2", contents=query
    ).embeddings[0].values
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return ranked_indices


# --- Generate ---

def build_prompt(query, context):
    return f"""You are a helpful assistant. Answer the question based on the context below.

Context:
{context}

Question: {query}"""


def generate(query, chunks, chunk_embeddings, top_k=1):
    ranked_indices = search(query, chunks, chunk_embeddings, top_k=top_k)
    context = "\n".join([chunks[i] for i in ranked_indices])
    prompt = build_prompt(query, context)
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )
    return response.text


# --- Main ---

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
    print("Embedding chunks...")
    chunk_embeddings = embed_chunks(chunks)
    np.save(EMBEDDINGS_FILE, chunk_embeddings)
    print("Embeddings saved to cache.")

query = "Why was Julius Caesar assassinated?"
print(f"\nQuery: {query}")
answer = generate(query, chunks, chunk_embeddings, top_k=1)
print(f"Answer: {answer}")
