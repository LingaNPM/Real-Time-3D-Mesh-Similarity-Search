import trimesh
from trimesh.scene import Scene
import os

def visualize_results(query_path, result_paths, output_path="result_scene.html"):
    print("🔍 Visualizing results...")
    scene = Scene()

    # Load and color the query mesh
    query_mesh = trimesh.load(query_path)
    query_mesh.visual.face_colors = [255, 0, 0, 200]  # red, semi-transparent
    query_mesh.apply_translation([0, 0, 0])
    scene.add_geometry(query_mesh, node_name="QueryMesh")

    # Add result meshes side by side
    spacing = 3.0
    for i, result in enumerate(result_paths):
        mesh = trimesh.load(result)
        mesh.visual.face_colors = [0, 255, 0, 180]  # green, semi-transparent
        mesh.apply_translation([spacing * (i+1), 0, 0])
        scene.add_geometry(mesh, node_name=f"Result_{i}")

    # Save as HTML
    try:
        scene.show(viewer='threejs')  # Launch in browser (interactive)
        # Optional: export HTML
        # scene.export(output_path)
        try:
            png_bytes = scene.save_image(resolution=(1024, 768))
            with open("scene_snapshot.png", "wb") as f:
                f.write(png_bytes)
                print("📸 Saved static image as scene_snapshot.png")
        except Exception as e:
            print("❌ Failed to render image snapshot:", e)

        print(f"✅ Visualization done")
    except Exception as e:
        print(f"[Failed] Visualization failed: {e}")
