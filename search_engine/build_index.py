import faiss
import numpy as np
import pickle
import os

# faiss can support cpy & Gpu
# Use FAISS-GPU on Linux for large-scale search
# Use Annoy or HNSWlib for memory-efficient CPU search on both platforms ( I want to include apple Metal GPU)
# Quantize vectors for faster retrieval (FAISS IVF/PQ)

def build_index(feature_matrix, ids, output_path):
    dim = feature_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(feature_matrix)
    faiss.write_index(index, output_path)
    with open(output_path + ".ids.pkl", "wb") as f:
        pickle.dump(ids, f)

def batch_query_index(index_path, feature_matrix, k=5):
    index = faiss.read_index(index_path)
    distances, indices = index.search(feature_matrix, k)
    return distances, indices
