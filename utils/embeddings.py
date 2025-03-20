import os
import json
import numpy as np
from utils.env_setup import client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
embeddings_path = os.path.join(BASE_DIR, "..", "Embeddings JSON", "embeddings.json")
metadata_path = os.path.join(BASE_DIR, "..", "Embeddings JSON", "metadata.json")

# Load Embeddings and Metadata
with open(embeddings_path, "r", encoding="utf-8") as f:
    embeddings_store = json.load(f)

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata_store = json.load(f)

# Merge metadata
for chunk_id, data in embeddings_store.items():
    if chunk_id in metadata_store:
        data["metadata"] = metadata_store[chunk_id]

def get_embedding(text: str):
    """Get embedding for a single text using the same model as embed_json.py."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve(query: str, top_k: int = 20):
    """Retrieve top_k chunks based on cosine similarity."""
    query_embedding = get_embedding(query)
    results = []
    for chunk_id, data in embeddings_store.items():
        embedding = data["embedding"]
        sim = cosine_similarity(query_embedding, embedding)
        results.append((chunk_id, sim, data))
    # Sort results by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
