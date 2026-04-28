import voyageai
import numpy as np
import os
from dotenv import load_dotenv
 
load_dotenv()
 
# Read API key from .env
API_KEY = os.getenv("VOYAGE_API_KEY")
 
vo = voyageai.Client(api_key=API_KEY)
 
documents = [
    "Database indexing can drastically improve query performance on large tables",
    "Connection pooling in Spring Boot prevents your app from running out of database connections",
    "Always use pagination when returning large datasets from your REST API",
    "Redis caching reduced our API response time from 800ms to 50ms in production",
    "Database normalization helps reduce data redundancy but can hurt read performance",
    "We migrated from monolith to microservices and our deployment frequency increased 10x",
    "Virat Kohli's cover drive is one of the most elegant shots in modern cricket",
    "Hyderabadi biryani tastes best when the rice is cooked in dum style with slow heat",
    "The 2011 World Cup final six by Dhoni is still the greatest moment in Indian cricket",
    "Nothing beats a hot filter coffee on a rainy Monday morning in Bangalore",
]
 
EMBEDDINGS_FILE = "embeddings.npy"
 
if os.path.exists(EMBEDDINGS_FILE):
    print("loading embeddings from cache")
    doc_embds = np.load(EMBEDDINGS_FILE)
else:
    print("generating embeddings for documents")
    doc_embds = vo.embed(documents, model="voyage-4", input_type="document").embeddings
    np.save(EMBEDDINGS_FILE, doc_embds)
    print("embeddings saved to cache")
 
def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
 
def search(query, top_k = 5):
    query_embd = vo.embed([query], model="voyage-4", input_type="query").embeddings[0]
    similarities = [cosine_similarity(query_embd, doc_embd) for doc_embd in doc_embds]
    similarities = np.array(similarities)
 
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
 
    print(f"Search results for query: '{query}'")
    for rank, idx in enumerate(ranked_indices):
        print(f"rank {rank+1} = {documents[idx]} (similarity = {similarities[idx]:.4f})")
    print()
 
 
 
search("What are the best practices for database performance?")
search("What is the best way to cook biryani?")