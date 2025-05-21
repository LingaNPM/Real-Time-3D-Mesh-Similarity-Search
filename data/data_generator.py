import trimesh
import os 

# Remove the invalid 'torus' shape and regenerate the valid ones
shapes = {
    "cube": trimesh.creation.box(),
    "sphere": trimesh.creation.icosphere(),
    "cylinder": trimesh.creation.cylinder(radius=0.5, height=1.0),
    "cone": trimesh.creation.cone(radius=0.5, height=1.0)
}

# Export each shape to .obj format
exported_files = []
for name, mesh in shapes.items():
    path = os.path.join(os.getcwd(), f"{name}.obj")
    mesh.process(validate=True)
    mesh.export(path)
    exported_files.append(path)

exported_files
