import open3d as o3d
import numpy as np

def mesh_to_pointcloud(path, num_points=1024):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points).astype(np.float32)
