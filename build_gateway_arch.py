"""Fast standalone Gateway Arch GLB builder for rapid iteration."""
import math, sys, time
import numpy as np
import trimesh

sys.path.insert(0, '.')
from citybuilder.glb import _add_prism_to_group


def catenary_arch(n_pts=80):
    """Generate points along the Gateway Arch weighted catenary curve.

    Real arch: height = span = 192m (630 ft).
    Uses Saarinen's weighted catenary: more vertical legs, smooth apex.
    Cross-section is equilateral triangle: ~16.5m at base, ~5.2m at apex.
    """
    H = 192.0   # height in meters
    W = 192.0   # span in meters

    # Weighted catenary with two parameters for inner/outer curves
    # Real arch centerline: y = fc * (1 - cosh(x/c) / cosh(L/c))
    # where fc=190.5m, L=W/2=96m, c varies
    # c=50 for smooth catenary — balanced between sharp peak and round
    c = 50.0
    half_w = W / 2
    A = H / (math.cosh(half_w / c) - 1.0)

    # Cross-section radius (half-width) — exaggerated for visibility
    cs_base = 28.0   # very wide to make triangular shape unmistakable
    cs_top = 5.0     # thin at apex

    points = []
    for i in range(n_pts):
        t = -1.0 + 2.0 * i / (n_pts - 1)
        x = t * W / 2

        # Inverted catenary: apex at top
        y = A * (math.cosh(W / (2 * c)) - math.cosh(x / c))

        # Tangent: dy/dx = A * sinh(x/c) / c (note sign)
        dy_dx = A * math.sinh(x / c) / c
        angle = math.atan2(dy_dx, 1.0)

        # Non-linear taper: stays wide at base, narrows near top
        # power > 1 means cross-section stays wide longer before tapering
        frac = y / H
        cs = cs_base + (cs_top - cs_base) * (frac ** 1.8)

        points.append((x, y, angle, cs))

    return points


def build_arch_mesh(points, group):
    """Build arch by extruding equilateral triangle cross-sections along the curve.

    The arch lies in the XY plane (X = span, Y = height).
    Cross-section normal is along the curve tangent.
    One triangle axis is Z (depth), the other rotates with the tangent.
    """
    n_pts = len(points)
    n_sides = 3  # triangular — the arch's signature cross-section
    verts = []
    faces = []

    for i, (x, y, angle, cs) in enumerate(points):
        # Tangent direction in XY plane
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Normal to tangent in XY: (-sin_a, cos_a)
        # Cross-section lives in the plane perpendicular to tangent
        # Two axes of cross-section:
        #   axis1: normal in XY plane = (-sin_a, cos_a, 0)
        #   axis2: Z axis = (0, 0, 1)

        r = cs / 2.0  # half-width of cross-section

        for j in range(n_sides):
            # One vertex points outward (along normal), flat face faces depth (Z)
            a = 2 * math.pi * j / n_sides
            # Local cross-section offsets
            d_norm = r * math.cos(a)   # along normal in XY
            d_z = r * math.sin(a)      # along Z (depth)

            vx = x + d_norm * (-sin_a)
            vy = y + d_norm * cos_a
            vz = d_z

            verts.append([vx, vy, vz])

    # Stitch adjacent cross-sections
    for i in range(n_pts - 1):
        base0 = i * n_sides
        base1 = (i + 1) * n_sides
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            faces.append([base0 + j, base0 + j_next, base1 + j_next])
            faces.append([base0 + j, base1 + j_next, base1 + j])

    # End caps
    # Left base
    cb = len(verts)
    x0, y0 = points[0][0], points[0][1]
    verts.append([x0, y0, 0])
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        faces.append([cb, j_next, j])

    # Right base
    ct = len(verts)
    last_base = (n_pts - 1) * n_sides
    xn, yn = points[-1][0], points[-1][1]
    verts.append([xn, yn, 0])
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        faces.append([ct, last_base + j, last_base + j_next])

    _add_prism_to_group(group, verts, faces)


# ── BUILD ──
groups = {}
groups['arch'] = {'verts': [], 'faces': [], 'offset': 0}

arch_points = catenary_arch(n_pts=48)
build_arch_mesh(arch_points, groups['arch'])

# ── EXPORT ──
t0 = time.perf_counter()
scene = trimesh.Scene()
for name, g in groups.items():
    if not g['verts']:
        continue
    mesh = trimesh.Trimesh(
        vertices=np.array(g['verts'], dtype=np.float64),
        faces=np.array(g['faces'], dtype=np.int32),
        process=False)
    mesh.fix_normals()
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0.85, 0.85, 0.88, 1.0],  # stainless steel
        roughnessFactor=0.2, metallicFactor=0.8, doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(mesh, geom_name=name)

import glob as _glob
existing = _glob.glob('output/gateway-arch-v*.glb')
nums = [int(f.split('-v')[-1].split('.')[0]) for f in existing if f.split('-v')[-1].split('.')[0].isdigit()]
ver = max(nums, default=0) + 1
out = 'output/gateway-arch.glb'
vout = f'output/gateway-arch-v{ver}.glb'
scene.export(out)
import shutil; shutil.copy2(out, vout)
dt = time.perf_counter() - t0
print(f"v{ver} exported in {dt:.2f}s -> {vout}")
