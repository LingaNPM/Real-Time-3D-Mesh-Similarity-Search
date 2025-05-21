import torch
import numpy as np
import faiss
from descriptors.deep_feature_extractor import SimplePointNet, extract_deep_feature
from utils.mesh_loader import load_and_preprocess_mesh

class MeshSearchBackend:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = SimplePointNet().to(self.device)
        self.model.eval()
        self.features = None
        self.ids = []
        self.index = None

    def set_device(self, device_str):
        if device_str not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unsupported device: {device_str}")
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if device_str == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        self.device = torch.device(device_str)
        self.model.to(self.device)
        print(f"[INFO] Using device: {self.device}")

    def index_meshes(self, mesh_paths):
        features = []
        ids = []
        for path in mesh_paths:
            try:
                mesh = load_and_preprocess_mesh(path)
                feat = extract_deep_feature(mesh, self.model, self.device)
                features.append(feat)
                ids.append(path)
                print(f"[INFO] Indexed {path}")
            except Exception as e:
                print(f"[ERROR] Processing {path}: {e}")

        if not features:
            raise RuntimeError("No features extracted. Aborting indexing.")

        self.features = np.vstack(features).astype('float32')
        self.ids = ids
        self.index = faiss.IndexFlatL2(self.features.shape[1])
        self.index.add(self.features)
        print(f"[INFO] Indexed {len(ids)} meshes.")

    def query(self, query_mesh_path, top_k=5):
        if self.index is None:
            raise RuntimeError("Index not built yet.")
        query_mesh = load_and_preprocess_mesh(query_mesh_path)
        query_feat = extract_deep_feature(query_mesh, self.model, self.device).astype('float32')
        D, I = self.index.search(np.expand_dims(query_feat, 0), top_k)
        results = [(self.ids[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
        return results


def test():
    import glob
    backend = MeshSearchBackend(device="cuda" if torch.cuda.is_available() else "cpu")
    mesh_files = glob.glob("data/meshes/*.obj")
    backend.index_meshes(mesh_files)
    results = backend.query("data/meshes/query.obj", top_k=5)
    print("Top results:")
    for mesh_id, dist in results:
        print(f"{mesh_id} (distance: {dist:.4f})")


if __name__ == "__main__":
    test()