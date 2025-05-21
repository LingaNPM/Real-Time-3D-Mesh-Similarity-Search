import os, sys
import glob
import torch
import numpy as np
from utils.mesh_loader import load_and_preprocess_mesh
from descriptors.deep_feature_extractor import SimplePointNet, HeavyPointNet, extract_deep_feature
from search_engine.build_index import build_index
from search_engine.build_index import batch_query_index
import argparse
from visualizer.threejs import visualize_results

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def index_meshes(data_dir, model, device):
    features = []
    ids = []

    mesh_files = glob.glob(os.path.join(data_dir, "**/*.obj"), recursive=True)

    for file in mesh_files:
        fname = os.path.basename(file)
        print(f"Processing {file}")
        try:
            mesh = load_and_preprocess_mesh(file)
            feat = extract_deep_feature(mesh, model, device)
            features.append(feat)
            ids.append(fname)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not features:
        raise RuntimeError("No features extracted. Aborting indexing.")

    features = np.vstack(features)
    build_index(features, ids, "index.faiss")


def run_query(model, device, query_path="/Users/lingaraja/work/Real-Time-3D-Mesh-Similarity-Search/data/meshes/cone.obj"):
    try:
        query_mesh = load_and_preprocess_mesh(query_path)
        query_vec = extract_deep_feature(query_mesh, model, device)
        results = query_index("index.faiss", query_vec)

        print("\nTop Results:")
        for mesh_id, score in results:
            print(f"{mesh_id} (score: {score:.4f})")

            result_files = [os.path.join("data/meshes", mesh_id) for mesh_id, _ in results]
            visualize_results("data/meshes/query.obj", result_files[:4])  # top 4 results


    except Exception as e:
        print(f"Query failed: {e}")



import argparse

def main():
    parser = argparse.ArgumentParser(description="3D Mesh Similarity Search")
    parser.add_argument(
        "--query", type=str, required=True,
        help="Path to the query .obj file"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to directory with .obj files to index"
    )
    parser.add_argument(
        "--device", type=str, required=True,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on: cpu, cuda, or mps"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        print("\n❌ Missing or invalid arguments. Usage:")
        parser.print_help()
        exit(0)

    try:
        print(f"Using device: {args.device}")
       # model = SimplePointNet().to(args.device)
        model = HeavyPointNet().to(args.device)
        model.eval()

        print("=== Building Mesh Index ===")
        index_meshes(args.data_dir, model, args.device)

        print("\n=== Running Query ===")
        run_query(model, args.device, args.query)

    except FileNotFoundError as fnf_error:
        print(f"📂 File error: {fnf_error}")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    #from backend.model_search import test
    #test()
    from GUI.qt_ms_app import *
    run()    

