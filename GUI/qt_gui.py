import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget
from backend.model_search import MeshSearchBackend
import torch

class MeshSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Mesh Similarity Search Demo")
        self.backend = MeshSearchBackend(device="cuda" if torch.cuda.is_available() else "cpu")

        self.layout = QVBoxLayout()
        self.index_btn = QPushButton("Index Meshes")
        self.query_btn = QPushButton("Select Query Mesh")
        self.device_btn = QPushButton("Toggle Device (CPU/CUDA)")

        self.status_label = QLabel("Status: Ready")
        self.results_list = QListWidget()

        self.layout.addWidget(self.index_btn)
        self.layout.addWidget(self.device_btn)
        self.layout.addWidget(self.query_btn)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.results_list)
        self.setLayout(self.layout)

        self.index_btn.clicked.connect(self.index_meshes)
        self.query_btn.clicked.connect(self.select_query_mesh)
        self.device_btn.clicked.connect(self.toggle_device)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.indexed = False

    def index_meshes(self):
        from glob import glob
        try:
            self.status_label.setText("Status: Indexing meshes...")
            mesh_files = glob("data/meshes/*.obj")
            self.backend.index_meshes(mesh_files)
            self.indexed = True
            self.status_label.setText(f"Status: Indexed {len(mesh_files)} meshes")
        except Exception as e:
            self.status_label.setText(f"Error indexing: {e}")

    def select_query_mesh(self):
        if not self.indexed:
            self.status_label.setText("Please index meshes first.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Query Mesh", "", "OBJ Files (*.obj)")
        if file_path:
            try:
                self.status_label.setText("Running query...")
                results = self.backend.query(file_path, top_k=5)
                self.results_list.clear()
                for mesh_id, dist in results:
                    self.results_list.addItem(f"{mesh_id} (distance: {dist:.4f})")
                self.status_label.setText("Query complete.")
            except Exception as e:
                self.status_label.setText(f"Query error: {e}")

    def toggle_device(self):
        try:
            new_device = "cpu" if self.device == "cuda" else "cuda"
            self.backend.set_device(new_device)
            self.device = new_device
            self.status_label.setText(f"Switched to device: {self.device}")
        except Exception as e:
            self.status_label.setText(f"Device switch error: {e}")

def start():
    app = QApplication([])
    window = MeshSearchApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    start()
