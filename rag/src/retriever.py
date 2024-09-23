from indexing import get_query_embedding, index
from preprocess import all_splits

def retrieve(query, top_k):
    query_embedding = get_query_embedding(query)
    distances, indices = index.search(query_embedding, top_k)
    threshold = 1.4
    results = [
        {
            "snippet": all_splits[i].page_content 
        }
        for i, dist in zip(indices[0], distances[0])
        if dist <= threshold
    ]
    return results, distances[0]