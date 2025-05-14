
from pointnet_encoder import *
from faiss_helper import build_index
from query import *

device = 'cpu'
model = SimplePointNet().to(device)
model.eval()  # load trained weights in a real scenario

# List of known 3D model mesh paths
model_paths = ["models/chair-lowpoly/chair.obj", "models/chair-highdense/chair1.obj", "models/chair-modern/modern_chair_11.obj", "models/chair-simple/chair-t.obj"]
index, _ = build_index(model_paths, model, device=device)

# Query
results = query_similar("models/chair.obj", model, index, model_paths)
print("Top suggestions:")
for path in results:
    print(path)
