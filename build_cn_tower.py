"""Fast standalone CN Tower GLB builder for rapid iteration."""
import math, sys, time
import numpy as np
import trimesh

sys.path.insert(0, '.')
from citybuilder.glb import (
    _make_tapered_prism, _add_prism_to_group, _build_profile_slices,
)


def make_y_section(r_base, fin_extent, fin_half_width, n_smooth=8):
    """Generate a Y-shaped cross-section that's mostly circular with 3 bumps.

    r_base: radius of the circular base shape
    fin_extent: how far each fin extends beyond the base circle
    fin_half_width: angular half-width of each fin bump (radians)
    n_smooth: points per segment between fins
    """
    pts = []
    fin_angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
    n_total = n_smooth * 6  # 6 segments: 3 fin peaks + 3 valleys

    for i in range(n_total):
        angle = 2 * math.pi * i / n_total
        # Calculate radius at this angle: base + bump near fin angles
        r = r_base
        for fa in fin_angles:
            # Angular distance to this fin
            da = abs(math.atan2(math.sin(angle - fa), math.cos(angle - fa)))
            if da < fin_half_width:
                # Cosine bump: smooth transition from base to fin tip
                bump = fin_extent * math.cos(da * math.pi / (2 * fin_half_width))
                r = max(r, r_base + bump)
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return pts


def extrude_profile(heights, sections, group):
    """Extrude a series of 2D cross-sections at given heights."""
    n_levels = len(heights)
    n_pts = len(sections[0])

    verts = []
    faces = []

    for i in range(n_levels):
        h = heights[i]
        for (x, z) in sections[i]:
            verts.append([x, h, z])

    for i in range(n_levels - 1):
        base0 = i * n_pts
        base1 = (i + 1) * n_pts
        for j in range(n_pts):
            j_next = (j + 1) % n_pts
            faces.append([base0 + j, base0 + j_next, base1 + j_next])
            faces.append([base0 + j, base1 + j_next, base1 + j])

    # Bottom cap
    cb = len(verts)
    verts.append([0, heights[0], 0])
    for j in range(n_pts):
        faces.append([cb, (j + 1) % n_pts, j])

    # Top cap
    ct = len(verts)
    tb = (n_levels - 1) * n_pts
    verts.append([0, heights[-1], 0])
    for j in range(n_pts):
        faces.append([ct, tb + j, tb + (j + 1) % n_pts])

    _add_prism_to_group(group, verts, faces)


groups = {}
groups['tower'] = {'verts': [], 'faces': [], 'offset': 0}
groups['pod']   = {'verts': [], 'faces': [], 'offset': 0}

# ── MAIN SHAFT (0–335 m) with Y-shaped cross-section ──
# Smooth concave taper + 3 fin bumps that fade out by ~280m
n_levels = 24
heights = []
sections = []
base_r = 16.0    # base circular radius (wider for better flare)
top_r = 4.5      # top circular radius (at pod)

for i in range(n_levels):
    frac = i / (n_levels - 1)
    h = frac * 335.0
    heights.append(h)

    # Smooth taper with slight concavity
    r = top_r + (base_r - top_r) * (1.0 - frac) ** 0.65

    # Fin parameters: prominent at base, fade to zero by 280m
    fin_frac = max(0, 1.0 - (h / 280.0) ** 1.5)
    fin_extent = 10.0 * fin_frac     # broader, smoother fins
    fin_hw = 0.45                      # wider angular spread (~26 degrees)

    sections.append(make_y_section(r, fin_extent, fin_hw, n_smooth=12))

extrude_profile(heights, sections, groups['tower'])

# ── MAIN POD (observation deck + restaurant, 335–353 m) ──
# Prominent flat disc with strong overhang and visible banding
pod_profile = [
    (335.0,  4.5),
    (335.5,  7.0),
    (336.5, 14.0),
    (337.5, 19.0),
    (338.5, 22.0),
    (339.5, 23.5),
    (340.5, 24.0),     # widest
    (342.0, 24.0),     # flat plateau
    (343.5, 23.5),
    (345.0, 23.0),
    (347.0, 22.0),
    (349.0, 19.0),
    (350.5, 14.0),
    (351.5,  9.0),
    (353.0,  4.5),
]
_build_profile_slices(0, 0, 0, pod_profile, groups['pod'], nsides=36)

# ── UPPER SHAFT (353–447 m) ──
# Thinner to increase slenderness above pod
upper_profile = [
    (353.0, 3.5),
    (370.0, 3.3),
    (390.0, 3.1),
    (410.0, 2.9),
    (430.0, 2.7),
    (447.0, 2.5),
]
_build_profile_slices(0, 0, 0, upper_profile, groups['tower'], nsides=24)

# ── SKY POD (447–457 m) ──
skypod_profile = [
    (447.0, 2.5),
    (448.5, 4.5),
    (450.0, 6.0),
    (451.5, 6.5),
    (453.0, 6.0),
    (455.0, 4.5),
    (457.0, 2.2),
]
_build_profile_slices(0, 0, 0, skypod_profile, groups['pod'], nsides=24)

# ── ANTENNA (457–553.3 m) ──
antenna_profile = [
    (457.0, 2.2),
    (475.0, 2.0),
    (490.0, 1.7),
    (510.0, 1.3),
    (530.0, 0.8),
    (545.0, 0.4),
    (553.3, 0.15),
]
_build_profile_slices(0, 0, 0, antenna_profile, groups['tower'], nsides=12)

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
    if name == 'tower':
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.75, 0.75, 0.75, 1.0],
            roughnessFactor=0.6, metallicFactor=0.1, doubleSided=True)
    else:
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.88, 0.88, 0.90, 1.0],
            roughnessFactor=0.3, metallicFactor=0.4, doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(mesh, geom_name=name)

import glob as _glob
existing = _glob.glob('output/cn-tower-v*.glb')
nums = [int(f.split('-v')[-1].split('.')[0]) for f in existing if f.split('-v')[-1].split('.')[0].isdigit()]
ver = max(nums, default=0) + 1
out = 'output/cn-tower.glb'
vout = f'output/cn-tower-v{ver}.glb'
scene.export(out)
import shutil; shutil.copy2(out, vout)
dt = time.perf_counter() - t0
print(f"v{ver} exported in {dt:.2f}s -> {vout}")
