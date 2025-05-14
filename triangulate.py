import argparse
import os
import openmesh as om
import open3d as o3d


def triangulate_with_openmesh(input_file, output_file):
    mesh = om.read_trimesh(input_file)
    mesh.request_face_normals()
    mesh.update_face_normals()
    om.write_mesh(output_file, mesh)

def triangulate_with_open3d(input_file, output_file):
    mesh = o3d.io.read_triangle_mesh(input_file)
    if mesh.has_vertices() and mesh.has_triangles():
        pass
    else:
        mesh = mesh.triangulate()
    o3d.io.write_triangle_mesh(output_file, mesh)

def main():
    parser = argparse.ArgumentParser(description="Triangulate an .obj file.")
    parser.add_argument("input", help="Path to the input .obj file")
    parser.add_argument("output", help="Path to save the triangulated .obj file")
    parser.add_argument("--method", choices=["pywavefront", "openmesh", "open3d"], default="openmesh",
                        help="Method to use for triangulation (default: openmesh)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: The file {args.input} does not exist.")
        return

    elif args.method == "openmesh":
        triangulate_with_openmesh(args.input, args.output)
    elif args.method == "open3d":
        triangulate_with_open3d(args.input, args.output)

    print(f"Triangulated .obj file saved to {args.output}")

if __name__ == "__main__":
    main()
