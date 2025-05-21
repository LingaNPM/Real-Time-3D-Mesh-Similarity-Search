import trimesh
import numpy as np

# this function will do neccessary pre-processing on input meshes.
# Todo: add Tringulation as well if not already.
# I expect meaningful obj here. The obj should contain vertices, faces and normals.


def load_and_preprocess_mesh(path, n_points=1024):

    try:
        mesh = trimesh.load_mesh(path, file_type='obj')

        if not isinstance(mesh, trimesh.Trimesh):
            # Some .obj files load as a Scene; sum() merges geometries
            mesh = mesh.dump().sum()

        # Normalize the mesh
        mesh.vertices -= mesh.center_mass
        scale = np.max(mesh.extents)
        mesh.vertices /= scale

        # Ensure vertices are in float32 format
        mesh.vertices = mesh.vertices.astype(np.double)

        return mesh

    except Exception as e:
        print(f"[ERROR] Could not load mesh {path}: {e}")
        raise
