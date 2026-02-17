"""Fast standalone Opera House GLB builder for rapid iteration."""
import math, sys, time
import numpy as np
import trimesh

sys.path.insert(0, '.')
from citybuilder.glb import (
    OPERA_R, _make_spherical_shell, _add_prism_to_group,
)

R = OPERA_R
PODIUM_TOP = 6.7
HEADING = math.radians(-22.0)

along_x = math.sin(HEADING)
along_z = math.cos(HEADING)
perp_x  = math.cos(HEADING)
perp_z  = -math.sin(HEADING)

groups = {}
groups['shells'] = {'verts': [], 'faces': [], 'offset': 0}
groups['podium'] = {'verts': [], 'faces': [], 'offset': 0}


def place_pair(along_off, perp_off, height, phi_deg,
               n_theta=32, n_phi=20, thickness=20.0):
    sphere_cy = PODIUM_TOP + height - R
    theta_base = math.acos(max(-1.0, min(1.0, 1.0 - height / R)))
    phi_max = math.radians(phi_deg)
    px = along_off * along_x + perp_off * perp_x
    pz = along_off * along_z + perp_off * perp_z
    for s in (+1, -1):
        sv, sf = _make_spherical_shell(
            px, pz, sphere_cy, R,
            theta_base, phi_max, HEADING, s,
            podium_y=PODIUM_TOP, thickness=thickness,
            n_theta=n_theta, n_phi=n_phi)
        _add_prism_to_group(groups['shells'], sv, sf)


def place_single(along_off, perp_off, height, phi_deg, heading_override,
                 side=+1, n_theta=16, n_phi=12, thickness=20.0):
    sphere_cy = PODIUM_TOP + height - R
    theta_base = math.acos(max(-1.0, min(1.0, 1.0 - height / R)))
    phi_max = math.radians(phi_deg)
    px = along_off * along_x + perp_off * perp_x
    pz = along_off * along_z + perp_off * perp_z
    sv, sf = _make_spherical_shell(
        px, pz, sphere_cy, R,
        theta_base, phi_max, heading_override, side,
        podium_y=PODIUM_TOP, thickness=thickness,
        n_theta=n_theta, n_phi=n_phi)
    _add_prism_to_group(groups['shells'], sv, sf)


# ── v60 restored: 3 broad pairs per group, original heading ─────

# Concert Hall (western group) — 3 shell pairs, broad
ch_base = -20.0
ch_perp = -26.0
for along, h, phi in [
    (  0.0,  67.0,  65.0),    # A: tallest, broadest
    ( 18.0,  50.0,  55.0),    # B: mid
    ( 38.0,  34.0,  45.0),    # C: shortest
]:
    place_pair(ch_base + along, ch_perp, h, phi)

# Joan Sutherland Theatre (eastern group) — 3 shell pairs
jst_base = -15.0
jst_perp = 26.0
for along, h, phi in [
    (  0.0,  58.0,  60.0),    # A: tallest
    ( 18.0,  42.0,  50.0),    # B: mid
    ( 38.0,  28.0,  40.0),    # C: shortest
]:
    place_pair(jst_base + along, jst_perp, h, phi)

# Restaurant / Bennelong (at tip, faces opposite direction)
place_single(80.0, 0.0, 18.0, 20.0, HEADING + math.pi, side=+1)

# ── PODIUM ────────────────────────────────────────────────────────
podium_outline = [
    (-55, -48), (-55, 48),
    ( 20, 48), (55, 28), (80, 10),
    (92, 0),
    (80, -10), (55, -28), (20, -48),
]
n_pts = len(podium_outline)
pv = []
for al, pr in podium_outline:
    px = al * along_x + pr * perp_x
    pz = al * along_z + pr * perp_z
    pv.append([px, 0.0, pz])
    pv.append([px, PODIUM_TOP, pz])
pf = []
for i in range(n_pts):
    j = (i + 1) % n_pts
    b0, t0 = 2*i, 2*i+1
    b1, t1 = 2*j, 2*j+1
    pf.append([b0, b1, t1])
    pf.append([b0, t1, t0])
for i in range(1, n_pts - 1):
    pf.append([1, 2*i+1, 2*(i+1)+1])
for i in range(1, n_pts - 1):
    pf.append([0, 2*(i+1), 2*i])
_add_prism_to_group(groups['podium'], pv, pf)

# ── EXPORT ────────────────────────────────────────────────────────
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
    if name == 'shells':
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.97, 0.94, 0.88, 1.0],
            roughnessFactor=0.25, metallicFactor=0.0, doubleSided=True)
    else:
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.75, 0.73, 0.70, 1.0], doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(mesh, geom_name=name)

import glob as _glob
existing = _glob.glob('output/sydney-opera-house-v*.glb')
nums = [int(f.split('-v')[-1].split('.')[0]) for f in existing if f.split('-v')[-1].split('.')[0].isdigit()]
ver = max(nums, default=0) + 1
out = 'output/sydney-opera-house.glb'
vout = f'output/sydney-opera-house-v{ver}.glb'
scene.export(out)
import shutil; shutil.copy2(out, vout)
dt = time.perf_counter() - t0
print(f"v{ver} exported in {dt:.2f}s  ({len(groups['shells']['verts'])} shell verts) -> {vout}")
