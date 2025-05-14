from mesh_to_pointcloud import *
import torch
import faiss

def build_index(model_paths, model, device='cpu'):
    features = []
    for path in model_paths:
        pc = mesh_to_pointcloud(path)
        pc_tensor = torch.from_numpy(pc.T).unsqueeze(0).to(device)  # [1, 3, N]
        with torch.no_grad():
            feat = model(pc_tensor).cpu().numpy()
        features.append(feat[0])
    features = np.stack(features).astype('float32')


    faiss.normalize_L2(features)

    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    print(index.ntotal)
    print(index.reconstruct(0))
    return index, features
