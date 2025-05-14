import numpy as np
import faiss

vecs = np.array([
    np.random.rand(1024),
    np.random.rand(1024),
    np.ones(1024),
    np.zeros(1024)
]).astype('float32')

faiss.normalize_L2(vecs)
index = faiss.IndexFlatIP(1024)
index.add(vecs)

query = vecs[0].reshape(1, -1)
D, I = index.search(query, 3)
print("Results:", I, D)
