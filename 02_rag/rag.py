from google import genai
from typer import prompt
import voyageai
import numpy as np
import os
from dotenv import load_dotenv
 
load_dotenv()
 
# Read API key from .env
API_KEY = os.getenv("VOYAGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
 
vo = voyageai.Client(api_key=API_KEY)
gemini_client = genai.Client(api_key = GEMINI_API_KEY)


 
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
  
def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
 
def search(query, top_k = 5):
    query_embd = vo.embed([query], model="voyage-4", input_type="query").embeddings[0]
    doc_embds = vo.embed(documents, model="voyage-4-lite", input_type="document").embeddings

    similarities = [cosine_similarity(query_embd, doc_embd) for doc_embd in doc_embds]
    similarities = np.array(similarities)
 
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return ranked_indices

def build_prompt(query, retrieved_docs):

    return f"""You are a helpful assistant that answers the query provided based on the following context: {retrieved_docs}

Answer the following question: {query}"""


def generate(query,top_k=1):
    ranked_indices = search(query, top_k=top_k)
    retrieved_docs = "\n".join([documents[idx] for idx in ranked_indices[:top_k]])
    prompt = build_prompt(query, retrieved_docs)
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )    
    return response.text

result = generate("Which is the best Biryani and why?")
# print query and result
print(f"Query: Which is the best Biryani and why?")
print(f"Answer: {result}")