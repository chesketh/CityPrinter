"""Fast standalone Brandenburg Gate GLB builder for rapid iteration."""
import math, sys, time
import numpy as np
import trimesh

sys.path.insert(0, '.')
from citybuilder.glb import (
    _make_tapered_prism, _add_prism_to_group, _build_profile_slices,
)

groups = {}
groups['columns'] = {'verts': [], 'faces': [], 'offset': 0}
groups['structure'] = {'verts': [], 'faces': [], 'offset': 0}

# ── BRANDENBURG GATE DIMENSIONS ──
# Real dimensions: ~65.5m wide, ~26m tall (columns), ~11m deep
# 12 columns (6 per side), 5 passageways
# Columns: ~1.1m diameter, ~15m tall Doric columns (with base/capital)
# Entablature + attic: ~11m above columns
# Quadriga on top: ~5m tall

GATE_DEPTH = 11.0    # depth (front to back)
COL_HEIGHT = 15.0    # correct column height
COL_RADIUS = 1.2     # slimmer Doric columns
COL_NSIDES = 10      # decagonal for smoother columns
ENTAB_HEIGHT = 1.5   # slim entablature
ATTIC_HEIGHT = 4.0   # reduced attic
QUADRIGA_HEIGHT = 3.0  # compact quadriga
BASE_HEIGHT = 1.5    # stepped base platform

# 6 columns per row, evenly spaced
# Center-to-center spacing ~10m across 5 gaps = 50m total span
COL_SPACING = 10.0
col_positions_x = [COL_SPACING * (i - 2.5) for i in range(6)]
# Result: [-27.5, -16.5, -5.5, 5.5, 16.5, 27.5]

# Two rows of columns (front and back)
col_rows_z = [-GATE_DEPTH / 2, GATE_DEPTH / 2]

# ── BASE PLATFORM ──
half_w = max(col_positions_x) + COL_RADIUS * 3
half_d = GATE_DEPTH / 2 + COL_RADIUS * 2
base_verts = [
    [-half_w, 0, -half_d], [half_w, 0, -half_d],
    [half_w, 0, half_d], [-half_w, 0, half_d],
    [-half_w, BASE_HEIGHT, -half_d], [half_w, BASE_HEIGHT, -half_d],
    [half_w, BASE_HEIGHT, half_d], [-half_w, BASE_HEIGHT, half_d],
]
base_faces = [
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
]
_add_prism_to_group(groups['structure'], base_verts, base_faces)

# ── COLUMNS ──
col_base = BASE_HEIGHT  # columns start on the platform
col_top = col_base + COL_HEIGHT
for cx in col_positions_x:
    for cz in col_rows_z:
        # Column base (wider)
        v, f = _make_tapered_prism(
            cx, cz, col_base, col_base + 0.8, COL_RADIUS * 1.3, COL_RADIUS, COL_NSIDES)
        _add_prism_to_group(groups['columns'], v, f)
        # Column shaft
        v, f = _make_tapered_prism(
            cx, cz, col_base + 0.8, col_top - 0.8, COL_RADIUS, COL_RADIUS * 0.95, COL_NSIDES)
        _add_prism_to_group(groups['columns'], v, f)
        # Column capital (wider)
        v, f = _make_tapered_prism(
            cx, cz, col_top - 0.8, col_top, COL_RADIUS * 0.95, COL_RADIUS * 1.5, COL_NSIDES)
        _add_prism_to_group(groups['columns'], v, f)

# ── ENTABLATURE (beam across top of columns) ──
entab_bottom = col_top
entab_top = col_top + ENTAB_HEIGHT

# Main horizontal beam (rectangular prism)
v, f = _make_tapered_prism(
    0, 0, entab_bottom, entab_top, half_w, half_w, 4, rotation=math.pi/4)
# Scale the Z dimension — need to make it rectangular, not square
# Since _make_tapered_prism makes a regular polygon, we need a different approach
# Use direct vertex construction for the rectangular beam
beam_verts = []
beam_faces = []
bx1, bx2 = -half_w, half_w
bz1, bz2 = -half_d, half_d
by1, by2 = entab_bottom, entab_top

# 8 corners of the box
beam_verts = [
    [bx1, by1, bz1], [bx2, by1, bz1], [bx2, by1, bz2], [bx1, by1, bz2],  # bottom
    [bx1, by2, bz1], [bx2, by2, bz1], [bx2, by2, bz2], [bx1, by2, bz2],  # top
]
beam_faces = [
    [0, 1, 5], [0, 5, 4],  # front
    [2, 3, 7], [2, 7, 6],  # back
    [0, 3, 2], [0, 2, 1],  # bottom
    [4, 5, 6], [4, 6, 7],  # top
    [0, 4, 7], [0, 7, 3],  # left
    [1, 2, 6], [1, 6, 5],  # right
]
_add_prism_to_group(groups['structure'], beam_verts, beam_faces)

# ── ATTIC (raised central section above entablature) ──
attic_bottom = entab_top
attic_top = entab_top + ATTIC_HEIGHT
attic_hw = half_w * 0.85  # slightly narrower than entablature
attic_hd = half_d * 0.9

attic_verts = [
    [-attic_hw, attic_bottom, -attic_hd], [attic_hw, attic_bottom, -attic_hd],
    [attic_hw, attic_bottom, attic_hd], [-attic_hw, attic_bottom, attic_hd],
    [-attic_hw, attic_top, -attic_hd], [attic_hw, attic_top, -attic_hd],
    [attic_hw, attic_top, attic_hd], [-attic_hw, attic_top, attic_hd],
]
attic_faces = [
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [0, 3, 2], [0, 2, 1],
    [4, 5, 6], [4, 6, 7],
    [0, 4, 7], [0, 7, 3],
    [1, 2, 6], [1, 6, 5],
]
_add_prism_to_group(groups['structure'], attic_verts, attic_faces)

# ── QUADRIGA (wider, flatter block on top center) ──
q_bottom = attic_top
q_top = attic_top + QUADRIGA_HEIGHT
q_hw = 6.0  # wider quadriga
q_hd = 3.0  # quadriga depth

q_verts = [
    [-q_hw, q_bottom, -q_hd], [q_hw, q_bottom, -q_hd],
    [q_hw, q_bottom, q_hd], [-q_hw, q_bottom, q_hd],
    [-q_hw, q_top, -q_hd], [q_hw, q_top, -q_hd],
    [q_hw, q_top, q_hd], [-q_hw, q_top, q_hd],
]
q_faces = [
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [0, 3, 2], [0, 2, 1],
    [4, 5, 6], [4, 6, 7],
    [0, 4, 7], [0, 7, 3],
    [1, 2, 6], [1, 6, 5],
]
_add_prism_to_group(groups['structure'], q_verts, q_faces)

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
    if name == 'columns':
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.85, 0.82, 0.75, 1.0],  # sandstone
            roughnessFactor=0.7, metallicFactor=0.0, doubleSided=True)
    else:
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.80, 0.78, 0.72, 1.0],  # slightly darker sandstone
            roughnessFactor=0.6, metallicFactor=0.05, doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(mesh, geom_name=name)

import glob as _glob
existing = _glob.glob('output/brandenburg-gate-v*.glb')
nums = [int(f.split('-v')[-1].split('.')[0]) for f in existing if f.split('-v')[-1].split('.')[0].isdigit()]
ver = max(nums, default=0) + 1
out = 'output/brandenburg-gate.glb'
vout = f'output/brandenburg-gate-v{ver}.glb'
scene.export(out)
import shutil; shutil.copy2(out, vout)
dt = time.perf_counter() - t0
print(f"v{ver} exported in {dt:.2f}s -> {vout}")
