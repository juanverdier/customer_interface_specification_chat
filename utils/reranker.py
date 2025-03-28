from utils.env_setup import co

def rerank_with_cohere(query: str, initial_results):
    candidate_docs = [data["content"] for _, _, data in initial_results]
    response = co.rerank(query=query, documents=candidate_docs, model="rerank-v3.5")
    
    reranked = response.results
    reranked_results = []
    for item in reranked:
        idx = item.index
        confidence = item.relevance_score
        chunk_id, sim, data = initial_results[idx]
        header = data.get("title", data["content"][:100])
        
        # Return (chunk_id, confidence, header, data)
        reranked_results.append((chunk_id, confidence, header, data))
    
    # Sort by confidence descending
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return reranked_results