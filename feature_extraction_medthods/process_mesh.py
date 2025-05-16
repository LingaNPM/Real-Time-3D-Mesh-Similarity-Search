import open3d as o3d

mesh = o3d.io.read_triangle_mesh("models/chair.obj")

if not mesh.has_triangles():
    print("Mesh has no triangles. Attempting reconstruction...")
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]
    print("Reconstruction complete.")
