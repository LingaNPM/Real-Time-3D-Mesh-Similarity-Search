import faiss
import numpy as np
import pickle

def query_index(index_path, query_vector, top_k=5):
    index = faiss.read_index(index_path)
    D, I = index.search(query_vector.reshape(1, -1), top_k)
    with open(index_path + ".ids.pkl", "rb") as f:
        ids = pickle.load(f)
    results = [(ids[i], D[0][j]) for j, i in enumerate(I[0])]
    return results
