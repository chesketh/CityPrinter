"""Quick test: regenerate GLB for Mount Rainier and check for wall faces."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from citybuilder.database import CityDatabase
from citybuilder import glb as glb_mod
import trimesh

CITY_ID = 216
OUT = "output/wall_test.glb"

print("Regenerating GLB...")
db = CityDatabase()
path = glb_mod.generate_glb(db, CITY_ID, OUT)
print(f"GLB written to: {path}")

print("\nChecking for visible wall faces at bbox boundary...")
scene = trimesh.load(path)

all_x = np.concatenate([g.vertices[:, 0] for g in scene.geometry.values()])
all_z = np.concatenate([g.vertices[:, 2] for g in scene.geometry.values()])
gx_min, gx_max = all_x.min(), all_x.max()
gz_min, gz_max = all_z.min(), all_z.max()

tol = 1.0
any_issues = False

for name, geom in scene.geometry.items():
    verts = geom.vertices
    faces = geom.faces
    normals = geom.face_normals

    x, z = verts[:, 0], verts[:, 2]
    on_bound = ((np.abs(x - gx_min) < tol) | (np.abs(x - gx_max) < tol) |
                (np.abs(z - gz_min) < tol) | (np.abs(z - gz_max) < tol))
    bound_count = np.array([on_bound[f].sum() for f in faces])
    has_boundary = (bound_count >= 2)

    # A visible wall: steep (Y-normal < 0.5) AND tall (Y-span > 10m)
    steep = np.abs(normals[:, 1]) < 0.5
    face_y = verts[faces, 1]
    y_span = face_y.max(axis=1) - face_y.min(axis=1)
    tall = y_span > 10.0
    visible_wall = has_boundary & steep & tall
    n = int(visible_wall.sum())

    if n > 0:
        any_issues = True
        print(f"  {name}: {n} visible wall faces!")
        for fi in np.where(visible_wall)[0][:3]:
            fv = verts[faces[fi]]
            print(f"    Y-normal={normals[fi,1]:.3f}, "
                  f"Y-span={fv[:,1].max()-fv[:,1].min():.1f}m")

if any_issues:
    print("\nFAILED")
else:
    print("\nPASSED — no visible wall faces at bbox boundary")
