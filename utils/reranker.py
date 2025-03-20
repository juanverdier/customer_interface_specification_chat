from utils.env_setup import co

def rerank_with_cohere(query: str, initial_results):
    """Rerank initial retrieval results using Cohere."""
    candidate_docs = [data["content"] for _, _, data in initial_results]

    response = co.rerank(query=query, documents=candidate_docs, model="rerank-v3.5")
    reranked = response.results

    reranked_results = [
        (initial_results[item.index][0], item.relevance_score, initial_results[item.index][2])
        for item in reranked
    ]
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return reranked_results
