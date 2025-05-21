import sys
import os
import numpy as np
import torch
import trimesh
import time

import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QComboBox, QVBoxLayout, QWidget, QLabel
)

from utils.mesh_loader import load_and_preprocess_mesh
from descriptors.deep_feature_extractor import SimplePointNet, extract_deep_feature
from search_engine.build_index import build_index
from search_engine.query import query_index



class MeshSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Mesh Similarity Viewer")
        self.resize(1000, 800)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=3)

        self.device_selector = QComboBox()
        self.device_selector.addItems(["cpu", "cuda", "mps"])
        self.device_selector.setCurrentText(self.get_available_device())

        self.query_button = QPushButton("Select Query Mesh")
        self.query_button.clicked.connect(self.load_query_mesh)

        self.search_button = QPushButton("Run Similarity Search")
        self.search_button.clicked.connect(self.run_similarity_search)
        self.search_time_label = QLabel("Search time: -- ms")

        layout = QVBoxLayout()
        layout.addWidget(self.device_selector)
        layout.addWidget(self.query_button)
        layout.addWidget(self.search_button)
        layout.addWidget(self.view)

        layout.addWidget(self.search_time_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.query_path = None
        self.model = None
        self.device = None
        self.load_model()

        if self.query_path:
            # rerun similarity search on new device automatically
            self.run_similarity_search()


    def get_available_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self):
        selected_device = self.device_selector.currentText()
        self.device_selector.currentTextChanged.connect(self.on_device_change)
        self.device = torch.device(selected_device)
        self.model = SimplePointNet().to(self.device)
        self.model.eval()

    def load_query_mesh(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Mesh", "", "OBJ Files (*.obj)")
        if file_path:
            self.query_path = file_path
            self.display_mesh(file_path, color=[0.0, 0.5, 1.0, 1.0], offset=0)

    def run_similarity_search(self):
        self.view.clear()
        self.load_model()

        if not self.query_path:
            return

        start = time.time()

        mesh_dir = "data/meshes"
        features = []
        ids = []
        files = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj")]

        for file in files:
            try:
                mesh = load_and_preprocess_mesh(file)
                feat = extract_deep_feature(mesh, self.model, self.device)
                features.append(feat)
                ids.append(file)
            except Exception as e:
                print(f"Skipping {file}: {e}")

        if not features:
            print("No features extracted. Aborting search.")
            return

        features = np.vstack(features)
        build_index(features, ids, "index.faiss")

        query_mesh = load_and_preprocess_mesh(self.query_path)
        query_vec = extract_deep_feature(query_mesh, self.model, self.device)
        results = query_index("index.faiss", query_vec, top_k=5)

        end = time.time()
        elapsed_ms = (end - start) * 1000
        self.search_time_label.setText(f"Search time: {elapsed_ms:.2f} ms")

        # Show query mesh
        self.display_mesh(self.query_path, color=[0.0, 0.5, 1.0, 1.0], offset=-1.0)

        for i, (mesh_path, score) in enumerate(results):
            color = [0.2, 0.8, 0.2, 1.0] if i == 0 else [0.6, 0.6, 0.6, 1.0]
            self.display_mesh(mesh_path, color=color, offset=i + 1)

    def display_mesh(self, mesh_path, color=[0.6, 0.6, 0.6, 1.0], offset=0):
        mesh = trimesh.load_mesh(mesh_path)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        vertices += offset  # Shift position
        colors = np.tile(color, (faces.shape[0], 1))

        item = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, smooth=True, drawEdges=True)
        self.view.addItem(item)

    def on_device_change(self, device_str):
        print(f"Switching device to: {device_str}")
        self.device = torch.device(device_str)
        self.model = SimplePointNet().to(self.device)
        self.model.eval()




def run():
    app = QApplication(sys.argv)
    window = MeshSimilarityApp()
    window.show()
    sys.exit(app.exec())