import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

class HeavyPointNet(nn.Module):
    def __init__(self, out_dim=256):
        super(HeavyPointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.permute(0, 2, 1)  # (B, 3, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.max(x, 2)[0]  # Global max pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Actually this extractor can be implemented many ways.
# Intended to expand with many Open3D and DNN. But I start with PointNet-style model now. (Todo)

class SimplePointNet(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return torch.max(self.model(x), dim=0)[0]

def sample_points_from_mesh(mesh, n_points=1024):
    pcd = mesh.sample_points_uniformly(n_points)
    return np.asarray(pcd.points).astype(np.float32)


def sample_points_trimesh(mesh, n_points=1024):
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def extract_batch_features(pointclouds, model, device):
    pointclouds = pointclouds.to(device)
    with torch.no_grad():
        features = model(pointclouds)  # (B, D)
    return features.cpu().numpy()


def extract_deep_feature(mesh, model, device):
    points = sample_points_trimesh(mesh)
    x = torch.from_numpy(points).float().to(device)
    feature = model(x).detach().cpu().numpy()
    return feature / np.linalg.norm(feature)  # Normalize
