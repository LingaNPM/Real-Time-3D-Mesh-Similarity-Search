from mesh_to_pointcloud import *
import torch

def query_similar(input_mesh_path, model, index, model_paths, k=5):
    pc = mesh_to_pointcloud(input_mesh_path)
    pc_tensor = torch.from_numpy(pc.T).unsqueeze(0)
    with torch.no_grad():
        feat = model(pc_tensor).numpy().astype('float32')
    D, I = index.search(feat, k)
    return [model_paths[i] for i in I[0]]
    faiss.normalize_L2(query)
