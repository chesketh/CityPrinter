"""GLB and STL file generation from stored city features."""

import json
import math
import logging
import random
import time

import numpy as np
import trimesh
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.wkb import loads as wkb_loads

from .constants import FEATURE_CATEGORIES
from .geometry import (extrude_watertight, generate_roof_mesh,
                       generate_window_quads, ROOF_SHAPES_SUPPORTED)
from .models import BoundingBox, PathManager
from .database import CityDatabase
from . import terrain as terrain_mod

logger = logging.getLogger(__name__)


def generate_stl(db, city_id: int, output_path: str) -> None:
    """Generate STL file from stored city data."""
    from shapely.wkb import loads as wkb_loads

    output_path = PathManager.get_output_path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    logger.info("Generating STL file...")

    # Get all features for the city
    features = db.get_city_features(city_id)

    # Create base mesh
    vertices = []
    faces = []
    face_colors = []
    current_vertex_offset = 0

    for feature_type, osm_id, geometry_blob, properties_json, source, confidence in features:
        try:
            geometry = wkb_loads(geometry_blob)
            properties = json.loads(properties_json)

            if feature_type == 'building':
                height = float(properties.get('height', 10))
                if math.isnan(height) or height <= 0:
                    height = 10.0
                new_verts, new_faces, colors = _geometry_to_mesh(geometry, height)
            elif feature_type == 'water':
                new_verts, new_faces, colors = _geometry_to_mesh(geometry, -1)  # Slight depression for water
            elif feature_type == 'road':
                new_verts, new_faces, colors = _geometry_to_mesh(geometry, 0.5)  # Slight elevation for roads
            else:
                continue

            if new_verts and new_faces:
                # Ensure all vertices are float arrays
                new_verts = [[float(x), float(y), float(z)] for x, y, z in new_verts]
                # Ensure all faces are integer arrays with consistent size (triangulate if needed)
                triangulated_faces = []
                triangulated_colors = []
                for face, color in zip(new_faces, colors):
                    if len(face) == 3:
                        triangulated_faces.append([int(i + current_vertex_offset) for i in face])
                        triangulated_colors.append(color)
                    elif len(face) == 4:
                        # Split quad into two triangles
                        triangulated_faces.append([
                            int(face[0] + current_vertex_offset),
                            int(face[1] + current_vertex_offset),
                            int(face[2] + current_vertex_offset)
                        ])
                        triangulated_faces.append([
                            int(face[0] + current_vertex_offset),
                            int(face[2] + current_vertex_offset),
                            int(face[3] + current_vertex_offset)
                        ])
                        triangulated_colors.extend([color, color])

                vertices.extend(new_verts)
                faces.extend(triangulated_faces)
                face_colors.extend(triangulated_colors)
                current_vertex_offset += len(new_verts)
        except Exception as e:
            logger.error(f"Error processing feature: {e}")
            continue

    if not vertices or not faces:
        logger.error("No valid geometry to generate STL file")
        return

    try:
        # Convert to numpy arrays with explicit types
        vertices_array = np.array(vertices, dtype=np.float64)
        faces_array = np.array(faces, dtype=np.int32)
        colors_array = np.array(face_colors, dtype=np.float64)

        # Create and save the mesh
        mesh = trimesh.Trimesh(
            vertices=vertices_array,
            faces=faces_array,
            face_colors=colors_array
        )
        mesh.export(str(output_path))
        logger.info(f"STL file generated successfully: {output_path}")
    except Exception as e:
        logger.error(f"Error creating mesh: {e}")
        raise


def _geometry_to_mesh(geometry, height: float):
    """Convert a geometry to a mesh with the given height.

    Args:
        geometry: The geometry to convert
        height: The height in meters

    Returns:
        Tuple of (vertices, faces, colors)
    """
    # Buffer line geometries into thin polygons (road surface)
    if geometry.geom_type in ('LineString', 'MultiLineString'):
        geometry = geometry.buffer(3.0, cap_style=2)
        if geometry.is_empty:
            return [], [], []

    # Get the coordinates of the geometry
    if geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    else:
        polygons = [geometry]

    vertices = []
    faces = []
    colors = []

    for polygon in polygons:
        # Get exterior coordinates
        coords = list(polygon.exterior.coords)[:-1]  # Remove last point (same as first)

        # Create base vertices
        base_vertices = [[x, y, 0] for x, y in coords]

        # Create top vertices - scale height by 3.28084 to compensate for STL viewer interpreting as feet
        top_vertices = [[x, y, height * 3.28084] for x, y in coords]

        # Add all vertices
        start_idx = len(vertices)
        vertices.extend(base_vertices)
        vertices.extend(top_vertices)

        # Create faces
        n_points = len(coords)

        # Add side faces (two triangles per side)
        for i in range(n_points):
            j = (i + 1) % n_points
            # First triangle
            faces.append([
                start_idx + i,
                start_idx + j,
                start_idx + i + n_points
            ])
            colors.append([0.8, 0.8, 0.8])  # Light gray

            # Second triangle
            faces.append([
                start_idx + j,
                start_idx + j + n_points,
                start_idx + i + n_points
            ])
            colors.append([0.8, 0.8, 0.8])  # Light gray

        # Add base face (subdivide into triangles)
        for i in range(1, n_points - 1):
            faces.append([
                start_idx,
                start_idx + i,
                start_idx + i + 1
            ])
            colors.append([0.7, 0.7, 0.7])  # Slightly darker gray for base

        # Add top face (subdivide into triangles)
        for i in range(1, n_points - 1):
            faces.append([
                start_idx + n_points,
                start_idx + n_points + i,
                start_idx + n_points + i + 1
            ])
            colors.append([0.9, 0.9, 0.9])  # Slightly lighter gray for top

    return vertices, faces, colors


def _scatter_on_polygon(polygon, density_sqm, min_count=0, max_count=500,
                        seed=42):
    """Return random points inside *polygon* via bbox rejection sampling.

    Parameters
    ----------
    polygon : shapely Polygon/MultiPolygon
        Area to scatter points within.
    density_sqm : float
        Average area (sq m) per object — larger = fewer objects.
    min_count, max_count : int
        Hard limits on the number of returned points.
    seed : int
        RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    area = polygon.area
    target = int(area / density_sqm)
    target = max(min_count, min(target, max_count))
    if target == 0:
        return []

    prepared = prep(polygon)
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    attempts = 0
    max_attempts = target * 20
    while len(points) < target and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        pt = Point(x, y)
        if prepared.contains(pt):
            points.append((x, y))
        attempts += 1
    return points


def _build_skirt_walls(terrain_verts, terrain_faces, base_y=-1.0):
    """Build vertical walls from terrain boundary edges down to *base_y*.

    Any edge in the terrain mesh that belongs to exactly one face is a
    boundary edge (bbox perimeter, water holes, building holes, etc.).
    For each such edge we drop a quad from the terrain surface down to
    *base_y*, creating a solid skirt suitable for 3-D printing.

    Vertices are shared via a cache keyed by (terrain_vi, is_top) so
    adjacent boundary edges reuse the same skirt vertex.  Each quad
    emits 4 faces (double-sided) for robust visibility from any angle.

    Returns (verts_list, faces_list) ready for the groups dict.
    """
    from collections import defaultdict

    # Build edge -> face-index mapping
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(terrain_faces):
        for a, b in ((face[0], face[1]),
                     (face[1], face[2]),
                     (face[2], face[0])):
            key = (min(a, b), max(a, b))
            edge_faces[key].append(fi)

    # Boundary edges: appear in exactly one face
    boundary_edges = [(e, fis[0]) for e, fis in edge_faces.items()
                      if len(fis) == 1]
    if not boundary_edges:
        return [], []

    verts: list[list[float]] = []
    faces: list[list[int]] = []

    # Vertex cache: (terrain_vertex_index, is_top) -> skirt vertex index
    v_cache: dict[tuple[int, bool], int] = {}

    def _get_vi(terrain_vi: int, top: bool) -> int:
        key = (terrain_vi, top)
        if key in v_cache:
            return v_cache[key]
        p = terrain_verts[terrain_vi]
        idx = len(verts)
        if top:
            verts.append([p[0], p[1], p[2]])
        else:
            verts.append([p[0], base_y, p[2]])
        v_cache[key] = idx
        return idx

    for (vi0, vi1), face_idx in boundary_edges:
        t0 = _get_vi(vi0, True)
        t1 = _get_vi(vi1, True)
        b0 = _get_vi(vi0, False)
        b1 = _get_vi(vi1, False)

        # Single-sided quad (2 faces) — glTF doubleSided handles back-faces
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])

    return verts, faces


# ── Window dimple helpers ─────────────────────────────────────────

_WINDOW_ELIGIBLE_TYPES = frozenset({
    'commercial', 'retail', 'shop', 'store', 'mall', 'supermarket',
    'office', 'industrial', 'warehouse', 'factory', 'manufacturing',
    'government', 'hospital', 'school', 'university', 'library',
    'museum', 'theatre', 'cinema', 'arts_centre', 'concert_hall',
    'hotel', 'apartments', 'civic',
})


_WINDOW_EXCLUDE_TYPES = frozenset({
    'house', 'detached', 'semidetached_house', 'terrace', 'residential',
    'garage', 'garages', 'carport', 'parking', 'shed', 'hut', 'cabin',
    'barn', 'farm_auxiliary', 'greenhouse', 'church', 'chapel', 'mosque',
    'temple', 'synagogue', 'shrine', 'cathedral', 'stadium',
    'roof', 'construction', 'ruins',
})


def _building_wants_windows(properties: dict) -> bool:
    """Return True if a building should get facade window quads.

    Uses an exclude-list approach: everything qualifies UNLESS it's a
    known small-residential, religious, or non-windowed type.  The caller
    already gates on extrude_height >= 12 and area >= 100, so generic
    ``building=yes`` that pass those thresholds are large enough.
    """
    btype = properties.get('building_type', 'yes').lower()
    return btype not in _WINDOW_EXCLUDE_TYPES


# ── Landmark geometry ──────────────────────────────────────────────
# Known landmarks that need procedural geometry beyond OSM data.
_LANDMARKS = {
    'statue_of_liberty': {'lat': 40.6892, 'lon': -74.0445, 'exclude_radius': 30},
    'sydney_opera_house': {'lat': -33.8568, 'lon': 151.2153, 'exclude_radius': 120},
    'cn_tower': {'lat': 43.6426, 'lon': -79.3871, 'exclude_radius': 40},
    'gateway_arch': {'lat': 38.6247, 'lon': -90.1848, 'exclude_radius': 120},
    'brandenburg_gate': {'lat': 52.5163, 'lon': 13.3777, 'exclude_radius': 50},
}


def _make_tapered_prism(cx, cz, y_bot, y_top, r_bot, r_top,
                        nsides=8, rotation=0.0):
    """Create a tapered prism (frustum) with *nsides* sides.

    Returns (verts, faces) in Y-up [easting, elevation, northing] format.
    *rotation* offsets the starting angle in radians (use pi/4 to align
    a square prism's flat faces with the axes).
    """
    verts = []
    faces = []

    # Bottom ring then top ring: 2*nsides vertices
    for i in range(nsides):
        angle = 2.0 * math.pi * i / nsides + rotation
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        verts.append([cx + r_bot * cos_a, y_bot, cz + r_bot * sin_a])
    for i in range(nsides):
        angle = 2.0 * math.pi * i / nsides + rotation
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        verts.append([cx + r_top * cos_a, y_top, cz + r_top * sin_a])

    # Side quads (two triangles each)
    for i in range(nsides):
        j = (i + 1) % nsides
        b0, b1 = i, j                        # bottom ring
        t0, t1 = nsides + i, nsides + j      # top ring
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    # Bottom cap (fan from centre)
    cbot = len(verts)
    verts.append([cx, y_bot, cz])
    for i in range(nsides):
        j = (i + 1) % nsides
        faces.append([cbot, j, i])

    # Top cap (fan from centre)
    ctop = len(verts)
    verts.append([cx, y_top, cz])
    for i in range(nsides):
        j = (i + 1) % nsides
        faces.append([ctop, nsides + i, nsides + j])

    return verts, faces


def _add_prism_to_group(group, verts, faces):
    """Append prism geometry into a mesh *group* dict."""
    off = group['offset']
    for f in faces:
        group['faces'].append([f[0] + off, f[1] + off, f[2] + off])
    group['verts'].extend(verts)
    group['offset'] += len(verts)


def _build_profile_slices(cx, cz, base_y, profile, group, nsides=12):
    """Build stacked thin plates from a (height, radius) profile list."""
    for i in range(len(profile) - 1):
        h0, r0 = profile[i]
        h1, r1 = profile[i + 1]
        if r0 <= 0 and r1 <= 0:
            continue
        v, f = _make_tapered_prism(
            cx, cz, base_y + h0, base_y + h1,
            r_bot=max(r0, 0.05), r_top=max(r1, 0.05), nsides=nsides)
        _add_prism_to_group(group, v, f)


def _build_offset_profile_slices(cx, cz, base_y, profile, group, nsides=8):
    """Build slices from a profile with per-slice center offsets.

    Each entry is (height, dx, dz, radius).
    """
    for i in range(len(profile) - 1):
        h0, dx0, dz0, r0 = profile[i]
        h1, dx1, dz1, r1 = profile[i + 1]
        if r0 <= 0 and r1 <= 0:
            continue
        mid_dx = (dx0 + dx1) / 2.0
        mid_dz = (dz0 + dz1) / 2.0
        v, f = _make_tapered_prism(
            cx + mid_dx, cz + mid_dz,
            base_y + h0, base_y + h1,
            r_bot=max(r0, 0.05), r_top=max(r1, 0.05), nsides=nsides)
        _add_prism_to_group(group, v, f)


def _generate_statue_of_liberty(cx_utm, cz_utm, terrain_elev,
                                groups, type_colors):
    """Generate Statue of Liberty geometry using stacked profile slices.

    The star-shaped Fort Wood base (~19 m) is already rendered from OSM
    building data.  This adds the granite pedestal and a detailed
    height–radius profile of the statue figure, arm, and torch.
    """
    fort_top = terrain_elev + 19.0        # top of Fort Wood
    pedestal_top = fort_top + 28.0        # top of granite pedestal (~47 m)

    # ── Pedestal (white stone → 'building' group) ──
    ped_v, ped_f = _make_tapered_prism(
        cx_utm, cz_utm, fort_top, pedestal_top,
        r_bot=9.0, r_top=7.0, nsides=4, rotation=math.pi / 4)
    _add_prism_to_group(groups['building'], ped_v, ped_f)

    # ── Statue body — profile-traced thin plates (patina green) ──
    if 'statue' not in groups:
        groups['statue'] = {'verts': [], 'faces': [], 'offset': 0}
    type_colors['statue'] = [0.41, 0.58, 0.49, 1.0]   # patina green

    # (height above pedestal top, radius in metres)
    body_profile = [
        ( 0.0, 6.5),    # feet / robe base
        ( 1.5, 7.0),    # robe hem flare
        ( 3.5, 6.8),
        ( 5.5, 6.2),
        ( 7.5, 5.6),
        ( 9.5, 5.1),
        (11.5, 4.6),
        (13.5, 4.2),
        (15.5, 3.8),
        (17.5, 3.5),
        (19.0, 3.2),    # waist
        (21.0, 3.0),
        (23.0, 3.3),    # torso widens
        (25.0, 3.8),    # chest
        (26.5, 4.0),    # shoulders
        (27.5, 3.5),
        (28.5, 2.0),    # neck
        (29.5, 2.3),    # lower face
        (30.5, 2.6),    # head
        (31.5, 3.0),    # crown base
        (33.0, 3.8),    # crown widest (rays)
        (34.0, 3.3),    # crown narrows
        (35.0, 2.0),    # crown tip
        (35.5, 0.3),    # spike
    ]
    _build_profile_slices(cx_utm, cz_utm, pedestal_top,
                          body_profile, groups['statue'], nsides=12)

    # ── Right arm — angled outward from shoulder to torch ──
    # (height, dx_offset, dz_offset, radius)
    arm_profile = [
        (26.0, 3.0, -0.5, 2.0),    # shoulder junction
        (28.0, 3.5, -0.5, 1.7),
        (30.0, 4.0, -0.5, 1.5),
        (32.0, 4.5, -0.5, 1.4),
        (34.0, 4.8, -0.5, 1.3),
        (36.0, 5.0, -0.5, 1.2),
        (38.0, 5.0, -0.5, 1.2),
        (40.0, 5.0, -0.5, 1.2),    # wrist
    ]
    _build_offset_profile_slices(cx_utm, cz_utm, pedestal_top,
                                 arm_profile, groups['statue'], nsides=8)

    # Torch platform (at arm tip)
    torch_cx = cx_utm + 5.0
    torch_cz = cz_utm - 0.5
    torch_profile = [
        (40.0, 2.0),    # handle
        (41.0, 2.8),    # platform widens
        (42.0, 2.5),    # platform rim
    ]
    _build_profile_slices(torch_cx, torch_cz, pedestal_top,
                          torch_profile, groups['statue'], nsides=8)

    # ── Torch flame (gold) ──
    if 'torch' not in groups:
        groups['torch'] = {'verts': [], 'faces': [], 'offset': 0}
    type_colors['torch'] = [0.85, 0.70, 0.20, 1.0]    # gold

    flame_profile = [
        (42.0, 2.0),
        (43.0, 1.8),
        (44.0, 1.3),
        (45.0, 0.7),
        (46.0, 0.2),    # flame tip
    ]
    _build_profile_slices(torch_cx, torch_cz, pedestal_top,
                          flame_profile, groups['torch'], nsides=8)

    logger.info("Generated Statue of Liberty landmark geometry "
                "(pedestal_top=%.0f m, torch_tip=%.0f m)",
                pedestal_top, pedestal_top + 46.0)


OPERA_R = 75.2  # metres — Utzon's universal sphere radius


def _make_spherical_shell(cx, cz, sphere_cy, R,
                          theta_base, phi_max,
                          heading_rad, side, podium_y,
                          thickness=10.0,
                          n_theta=16, n_phi=12):
    """Generate a thick shell with base legs extending to the podium.

    Creates outer surface (radius R), inner surface (radius R-thickness),
    edge caps, and vertical legs from the base down to *podium_y*.

    Returns (verts, faces) in Y-up model coordinates.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    def _sp(theta, phi, r):
        """Spherical → rotated Y-up Cartesian."""
        sx = r * math.sin(theta) * math.sin(phi)
        sy = r * math.cos(theta)
        sz = -(r * math.sin(theta) * math.cos(phi))
        rx = sx * cos_h + sz * sin_h
        rz = -sx * sin_h + sz * cos_h
        return [cx + rx, sphere_cy + sy, cz + rz]

    cols = n_phi + 1

    sin_tb = math.sin(theta_base) if theta_base > 1e-6 else 1e-6
    MIN_PHI_FRAC = 0.0  # natural taper

    # ── Outer surface (radius R) ──
    for j in range(n_theta + 1):
        t = j / n_theta
        theta = t * theta_base
        frac = math.sin(theta) / sin_tb if theta > 1e-6 else 0.0
        phi_ext = phi_max * max(MIN_PHI_FRAC, frac)  # minimum width at ridge
        for i in range(cols):
            phi = side * phi_ext * (i / n_phi)
            verts.append(_sp(theta, phi, R))
    outer_n = len(verts)

    for j in range(n_theta):
        for i in range(n_phi):
            a = j * cols + i
            b = a + 1
            c = (j + 1) * cols + i
            d = c + 1
            faces.extend([[a, c, b], [b, c, d]])

    # ── Inner surface (radius varies: thin at ridge, thick at base) ──
    for j in range(n_theta + 1):
        t = j / n_theta
        theta = t * theta_base
        # Taper thickness: 10% at ridge, 100% at base
        t_thick = max(0.1, t)
        R_in = R - thickness * t_thick
        frac = math.sin(theta) / sin_tb if theta > 1e-6 else 0.0
        phi_ext = phi_max * max(MIN_PHI_FRAC, frac)  # minimum width: match outer
        for i in range(cols):
            phi = side * phi_ext * (i / n_phi)
            verts.append(_sp(theta, phi, R_in))

    # Inner faces (reversed winding)
    for j in range(n_theta):
        for i in range(n_phi):
            a = outer_n + j * cols + i
            b = a + 1
            c = outer_n + (j + 1) * cols + i
            d = c + 1
            faces.extend([[a, b, c], [b, d, c]])

    # ── Edge caps: ridge (i=0) and outer edge (i=n_phi) ──
    for j in range(n_theta):
        # Ridge seam
        a = j * cols
        b = (j + 1) * cols
        c = outer_n + j * cols
        d = outer_n + (j + 1) * cols
        faces.extend([[a, c, b], [b, c, d]])
        # Outer edge seam
        a = j * cols + n_phi
        b = (j + 1) * cols + n_phi
        c = outer_n + j * cols + n_phi
        d = outer_n + (j + 1) * cols + n_phi
        faces.extend([[a, b, c], [b, d, c]])

    # ── Base skirt: connect base row down to podium_y ──
    base_out = n_theta * cols           # first outer base vertex index
    base_in = outer_n + n_theta * cols  # first inner base vertex index

    # Close outer-to-inner at base
    for i in range(n_phi):
        a = base_out + i
        b = base_out + i + 1
        c = base_in + i
        d = base_in + i + 1
        faces.extend([[a, b, c], [b, d, c]])

    # Vertical legs: outer base → ground, inner base → ground, bottom cap
    leg_out = len(verts)
    for i in range(cols):
        v = verts[base_out + i]
        verts.append([v[0], podium_y, v[2]])
    leg_in = len(verts)
    for i in range(cols):
        v = verts[base_in + i]
        verts.append([v[0], podium_y, v[2]])

    # Outer leg face
    for i in range(n_phi):
        a = base_out + i;  b = base_out + i + 1
        c = leg_out + i;   d = leg_out + i + 1
        faces.extend([[a, c, b], [b, c, d]])
    # Inner leg face
    for i in range(n_phi):
        a = base_in + i;   b = base_in + i + 1
        c = leg_in + i;    d = leg_in + i + 1
        faces.extend([[a, b, c], [b, d, c]])
    # Bottom cap (podium level)
    for i in range(n_phi):
        a = leg_out + i;   b = leg_out + i + 1
        c = leg_in + i;    d = leg_in + i + 1
        faces.extend([[a, b, c], [b, d, c]])

    return verts, faces


def _generate_sydney_opera_house(cx_utm, cz_utm, terrain_elev,
                                  groups, type_colors):
    """Generate Sydney Opera House roof shells using spherical geometry.

    All shells are sections of a sphere of radius 75.2 m — Utzon's
    breakthrough "spherical solution" (1961).  Each shell pair consists
    of two mirror-image half-shells opening in opposite lateral
    directions from a shared ridge line.

    Geometry
    --------
    For a shell of height *h* above the podium:
        sphere_cy  = podium_top + h − R     (sphere centre Y)
        theta_base = arccos(1 − h/R)        (polar angle at the base)
    The apex is at Y = sphere_cy + R = podium_top + h.

    Real dimensions: building 183 m long × 120 m wide, tallest shell
    ~67 m above sea level, podium ~8 m above water.
    """
    R = OPERA_R

    if 'shells' not in groups:
        groups['shells'] = {'verts': [], 'faces': [], 'offset': 0}
    type_colors['shells'] = [0.97, 0.94, 0.88, 1.0]   # warm ivory ceramic

    podium_top = terrain_elev + 6.7   # real podium height

    # Building axis heading: Bennelong Point runs ~338° compass = −22°.
    heading = math.radians(-22.0)

    # Unit vectors: along = toward harbour tip (NNW), perp = to the right
    along_x = math.sin(heading)
    along_z = math.cos(heading)
    perp_x = math.cos(heading)
    perp_z = -math.sin(heading)

    def _place_pair(along_off, perp_off, height, phi_deg,
                    n_theta=32, n_phi=20, thickness=20.0):
        """Place a shell pair at the given offset from building centre."""
        sphere_cy = podium_top + height - R
        theta_base = math.acos(max(-1.0, min(1.0, 1.0 - height / R)))
        phi_max = math.radians(phi_deg)

        px = cx_utm + along_off * along_x + perp_off * perp_x
        pz = cz_utm + along_off * along_z + perp_off * perp_z

        for s in (+1, -1):
            sv, sf = _make_spherical_shell(
                px, pz, sphere_cy, R,
                theta_base, phi_max, heading, s,
                podium_y=podium_top, thickness=thickness,
                n_theta=n_theta, n_phi=n_phi)
            _add_prism_to_group(groups['shells'], sv, sf)

    def _place_single(along_off, perp_off, height, phi_deg, heading_override,
                      side=+1, n_theta=16, n_phi=12, thickness=20.0):
        """Place a single shell (not a mirror pair)."""
        sphere_cy = podium_top + height - R
        theta_base = math.acos(max(-1.0, min(1.0, 1.0 - height / R)))
        phi_max = math.radians(phi_deg)
        px = cx_utm + along_off * along_x + perp_off * perp_x
        pz = cz_utm + along_off * along_z + perp_off * perp_z
        sv, sf = _make_spherical_shell(
            px, pz, sphere_cy, R,
            theta_base, phi_max, heading_override, side,
            podium_y=podium_top, thickness=thickness,
            n_theta=n_theta, n_phi=n_phi)
        _add_prism_to_group(groups['shells'], sv, sf)

    # ── Concert Hall (western group) — 3 broad shell pairs ──
    ch_base = -20.0
    ch_perp = -26.0
    concert = [
        # (along_offset, height_m, phi_max_deg)
        (  0.0,  67.0,  65.0),    # A: tallest, broadest
        ( 18.0,  50.0,  55.0),    # B: mid
        ( 38.0,  34.0,  45.0),    # C: shortest
    ]
    for along, h, phi in concert:
        _place_pair(ch_base + along, ch_perp, h, phi)

    # ── Joan Sutherland Theatre (eastern group) — 3 broad shell pairs ──
    jst_base = -15.0
    jst_perp = 26.0
    theatre = [
        (  0.0,  58.0,  60.0),    # A: tallest
        ( 18.0,  42.0,  50.0),    # B: mid
        ( 38.0,  28.0,  40.0),    # C: shortest
    ]
    for along, h, phi in theatre:
        _place_pair(jst_base + along, jst_perp, h, phi)

    # ── Restaurant / Bennelong (at NNW tip, faces opposite) ──
    _place_single(80.0, 0.0, 18.0, 20.0, heading + math.pi, side=+1)

    total = len(groups['shells']['verts'])
    logger.info("Generated Sydney Opera House: %d shell verts "
                "(spherical R=%.1f m, podium=%.1f m, "
                "tallest apex=%.0f m)",
                total, R, podium_top, podium_top + 67.0)


def _cn_tower_y_section(r_base, fin_extent, fin_half_width, n_smooth=12):
    """Generate a Y-shaped 2D cross-section (mostly circular with 3 bumps)."""
    pts = []
    fin_angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
    n_total = n_smooth * 6

    for i in range(n_total):
        angle = 2 * math.pi * i / n_total
        r = r_base
        for fa in fin_angles:
            da = abs(math.atan2(math.sin(angle - fa), math.cos(angle - fa)))
            if da < fin_half_width:
                bump = fin_extent * math.cos(da * math.pi / (2 * fin_half_width))
                r = max(r, r_base + bump)
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return pts


def _cn_tower_extrude(cx, cz, heights, sections, group):
    """Extrude a Y-shaped cross-section profile at (cx, cz) in UTM."""
    n_levels = len(heights)
    n_pts = len(sections[0])

    verts = []
    faces = []

    for i in range(n_levels):
        h = heights[i]
        for (x, z) in sections[i]:
            verts.append([cx + x, h, cz + z])

    for i in range(n_levels - 1):
        b0 = i * n_pts
        b1 = (i + 1) * n_pts
        for j in range(n_pts):
            jn = (j + 1) % n_pts
            faces.append([b0 + j, b0 + jn, b1 + jn])
            faces.append([b0 + j, b1 + jn, b1 + j])

    # Bottom cap
    cb = len(verts)
    verts.append([cx, heights[0], cz])
    for j in range(n_pts):
        faces.append([cb, (j + 1) % n_pts, j])

    # Top cap
    ct = len(verts)
    tb = (n_levels - 1) * n_pts
    verts.append([cx, heights[-1], cz])
    for j in range(n_pts):
        faces.append([ct, tb + j, tb + (j + 1) % n_pts])

    _add_prism_to_group(group, verts, faces)


def _generate_cn_tower(cx_utm, cz_utm, terrain_elev,
                       groups, type_colors):
    """Generate CN Tower geometry using Y-shaped shaft + disc pod + antenna.

    Total height: 553.3 m.  The base has a Y-shaped cross-section that
    transitions to circular.  Observation pod at ~335 m, SkyPod at ~447 m.
    """
    base_y = terrain_elev

    # ── Tower shaft & antenna → 'cn_shaft' group (concrete grey) ──
    if 'cn_shaft' not in groups:
        groups['cn_shaft'] = {'verts': [], 'faces': [], 'offset': 0}
    type_colors['cn_shaft'] = [0.75, 0.75, 0.75, 1.0]

    # Main shaft (0–335 m): Y-shaped cross-section
    n_levels = 24
    heights = []
    sections = []
    base_r = 16.0
    top_r = 4.5

    for i in range(n_levels):
        frac = i / (n_levels - 1)
        h = base_y + frac * 335.0
        heights.append(h)
        r = top_r + (base_r - top_r) * (1.0 - frac) ** 0.65
        fin_frac = max(0, 1.0 - ((frac * 335.0) / 280.0) ** 1.5)
        fin_extent = 10.0 * fin_frac
        fin_hw = 0.45
        sections.append(_cn_tower_y_section(r, fin_extent, fin_hw))

    _cn_tower_extrude(cx_utm, cz_utm, heights, sections,
                      groups['cn_shaft'])

    # Upper shaft (353–447 m)
    upper_profile = [
        (353.0, 3.5), (370.0, 3.3), (390.0, 3.1),
        (410.0, 2.9), (430.0, 2.7), (447.0, 2.5),
    ]
    _build_profile_slices(cx_utm, cz_utm, base_y,
                          upper_profile, groups['cn_shaft'], nsides=24)

    # Antenna (457–553.3 m)
    antenna_profile = [
        (457.0, 2.2), (475.0, 2.0), (490.0, 1.7),
        (510.0, 1.3), (530.0, 0.8), (545.0, 0.4), (553.3, 0.15),
    ]
    _build_profile_slices(cx_utm, cz_utm, base_y,
                          antenna_profile, groups['cn_shaft'], nsides=12)

    # ── Pod & SkyPod → 'cn_pod' group (lighter metallic grey) ──
    if 'cn_pod' not in groups:
        groups['cn_pod'] = {'verts': [], 'faces': [], 'offset': 0}
    type_colors['cn_pod'] = [0.88, 0.88, 0.90, 1.0]

    # Main observation pod (335–353 m)
    pod_profile = [
        (335.0, 4.5), (335.5, 7.0), (336.5, 14.0), (337.5, 19.0),
        (338.5, 22.0), (339.5, 23.5), (340.5, 24.0), (342.0, 24.0),
        (343.5, 23.5), (345.0, 23.0), (347.0, 22.0), (349.0, 19.0),
        (350.5, 14.0), (351.5, 9.0), (353.0, 4.5),
    ]
    _build_profile_slices(cx_utm, cz_utm, base_y,
                          pod_profile, groups['cn_pod'], nsides=36)

    # SkyPod (447–457 m)
    skypod_profile = [
        (447.0, 2.5), (448.5, 4.5), (450.0, 6.0), (451.5, 6.5),
        (453.0, 6.0), (455.0, 4.5), (457.0, 2.2),
    ]
    _build_profile_slices(cx_utm, cz_utm, base_y,
                          skypod_profile, groups['cn_pod'], nsides=24)

    logger.info("Generated CN Tower landmark geometry "
                "(base=%.0f m, tip=%.0f m, total=553.3 m)",
                base_y, base_y + 553.3)


def _gateway_arch_catenary(cx_utm, cz_utm, base_y, n_pts=64):
    """Generate points along the Gateway Arch weighted catenary curve.

    Returns list of (world_x, world_y, tangent_angle, cross_section_half_width).
    The arch spans 192m along the X (easting) axis, centered at cx_utm.
    """
    H = 192.0   # height in meters
    W = 192.0   # span in meters
    c = 50.0    # catenary shape parameter
    half_w = W / 2
    A = H / (math.cosh(half_w / c) - 1.0)

    cs_base = 20.0  # cross-section half-width at base
    cs_top = 4.5     # cross-section half-width at apex

    points = []
    for i in range(n_pts):
        t = -1.0 + 2.0 * i / (n_pts - 1)
        x_local = t * half_w

        # Catenary height (apex at top)
        y_local = A * (math.cosh(half_w / c) - math.cosh(x_local / c))

        # Tangent angle
        dy_dx = A * math.sinh(x_local / c) / c
        angle = math.atan2(dy_dx, 1.0)

        # Cross-section taper (non-linear: stays wide at base)
        frac = y_local / H
        cs = cs_base + (cs_top - cs_base) * (frac ** 1.8)

        points.append((cx_utm + x_local, base_y + y_local, angle, cs))

    return points


def _gateway_arch_extrude(cz_utm, points, group):
    """Extrude hexagonal cross-sections along the arch catenary curve.

    The arch lies in the XY plane (easting/height), with depth along Z
    (northing) centered at cz_utm.
    """
    n_pts = len(points)
    n_sides = 6  # hexagonal for solid volume
    verts = []
    faces = []

    for i, (wx, wy, angle, cs) in enumerate(points):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        r = cs / 2.0

        for j in range(n_sides):
            a = 2 * math.pi * j / n_sides
            d_norm = r * math.cos(a)   # along normal in XY
            d_z = r * math.sin(a)      # along Z (depth)

            vx = wx + d_norm * (-sin_a)
            vy = wy + d_norm * cos_a
            vz = cz_utm + d_z

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
    cb = len(verts)
    verts.append([points[0][0], points[0][1], cz_utm])
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        faces.append([cb, j_next, j])

    ct = len(verts)
    last_base = (n_pts - 1) * n_sides
    verts.append([points[-1][0], points[-1][1], cz_utm])
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        faces.append([ct, last_base + j, last_base + j_next])

    _add_prism_to_group(group, verts, faces)


def _generate_gateway_arch(cx_utm, cz_utm, terrain_elev,
                           groups, type_colors):
    """Generate Gateway Arch geometry — weighted catenary, 192m tall/wide."""
    base_y = terrain_elev

    if 'gateway_arch' not in groups:
        groups['gateway_arch'] = {'verts': [], 'faces': [], 'offset': 0}
    type_colors['gateway_arch'] = [0.85, 0.85, 0.88, 1.0]  # stainless steel

    points = _gateway_arch_catenary(cx_utm, cz_utm, base_y)
    _gateway_arch_extrude(cz_utm, points, groups['gateway_arch'])

    logger.info("Generated Gateway Arch landmark geometry "
                "(base=%.0f m, apex=%.0f m, span=192 m)",
                base_y, base_y + 192.0)


def _generate_brandenburg_gate(cx_utm, cz_utm, terrain_elev,
                               groups, type_colors):
    """Generate Brandenburg Gate geometry — 12 Doric columns, entablature, attic, quadriga."""
    base_y = terrain_elev

    # Mesh groups for distinct colors
    if 'bg_columns' not in groups:
        groups['bg_columns'] = {'verts': [], 'faces': [], 'offset': 0}
    if 'bg_structure' not in groups:
        groups['bg_structure'] = {'verts': [], 'faces': [], 'offset': 0}
    type_colors['bg_columns'] = [0.85, 0.82, 0.75, 1.0]   # sandstone
    type_colors['bg_structure'] = [0.80, 0.78, 0.72, 1.0]  # darker sandstone

    col_g = groups['bg_columns']
    str_g = groups['bg_structure']

    # ── Dimensions (from v6 standalone builder) ──
    GATE_DEPTH = 11.0
    COL_HEIGHT = 15.0
    COL_RADIUS = 1.2
    COL_NSIDES = 10
    ENTAB_HEIGHT = 1.5
    ATTIC_HEIGHT = 4.0
    QUADRIGA_HEIGHT = 3.0
    BASE_HEIGHT = 1.5
    COL_SPACING = 10.0

    col_positions_x = [COL_SPACING * (i - 2.5) for i in range(6)]
    col_rows_z = [-GATE_DEPTH / 2, GATE_DEPTH / 2]

    # ── Base platform ──
    half_w = max(col_positions_x) + COL_RADIUS * 3
    half_d = GATE_DEPTH / 2 + COL_RADIUS * 2
    by0 = base_y
    by1 = base_y + BASE_HEIGHT
    bv = [
        [cx_utm - half_w, by0, cz_utm - half_d],
        [cx_utm + half_w, by0, cz_utm - half_d],
        [cx_utm + half_w, by0, cz_utm + half_d],
        [cx_utm - half_w, by0, cz_utm + half_d],
        [cx_utm - half_w, by1, cz_utm - half_d],
        [cx_utm + half_w, by1, cz_utm - half_d],
        [cx_utm + half_w, by1, cz_utm + half_d],
        [cx_utm - half_w, by1, cz_utm + half_d],
    ]
    bf = [
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
    _add_prism_to_group(str_g, bv, bf)

    # ── Columns (12 total: 6 per row × 2 rows) ──
    col_base = base_y + BASE_HEIGHT
    col_top = col_base + COL_HEIGHT
    for cx_off in col_positions_x:
        for cz_off in col_rows_z:
            wx = cx_utm + cx_off
            wz = cz_utm + cz_off
            # Column base (wider)
            v, f = _make_tapered_prism(
                wx, wz, col_base, col_base + 0.8,
                COL_RADIUS * 1.3, COL_RADIUS, COL_NSIDES)
            _add_prism_to_group(col_g, v, f)
            # Column shaft
            v, f = _make_tapered_prism(
                wx, wz, col_base + 0.8, col_top - 0.8,
                COL_RADIUS, COL_RADIUS * 0.95, COL_NSIDES)
            _add_prism_to_group(col_g, v, f)
            # Column capital (wider)
            v, f = _make_tapered_prism(
                wx, wz, col_top - 0.8, col_top,
                COL_RADIUS * 0.95, COL_RADIUS * 1.5, COL_NSIDES)
            _add_prism_to_group(col_g, v, f)

    # ── Entablature beam ──
    ey0 = col_top
    ey1 = col_top + ENTAB_HEIGHT
    ev = [
        [cx_utm - half_w, ey0, cz_utm - half_d],
        [cx_utm + half_w, ey0, cz_utm - half_d],
        [cx_utm + half_w, ey0, cz_utm + half_d],
        [cx_utm - half_w, ey0, cz_utm + half_d],
        [cx_utm - half_w, ey1, cz_utm - half_d],
        [cx_utm + half_w, ey1, cz_utm - half_d],
        [cx_utm + half_w, ey1, cz_utm + half_d],
        [cx_utm - half_w, ey1, cz_utm + half_d],
    ]
    ef = [
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
    _add_prism_to_group(str_g, ev, ef)

    # ── Attic (narrower block above entablature) ──
    attic_hw = half_w * 0.85
    attic_hd = half_d * 0.9
    ay0 = ey1
    ay1 = ey1 + ATTIC_HEIGHT
    av = [
        [cx_utm - attic_hw, ay0, cz_utm - attic_hd],
        [cx_utm + attic_hw, ay0, cz_utm - attic_hd],
        [cx_utm + attic_hw, ay0, cz_utm + attic_hd],
        [cx_utm - attic_hw, ay0, cz_utm + attic_hd],
        [cx_utm - attic_hw, ay1, cz_utm - attic_hd],
        [cx_utm + attic_hw, ay1, cz_utm - attic_hd],
        [cx_utm + attic_hw, ay1, cz_utm + attic_hd],
        [cx_utm - attic_hw, ay1, cz_utm + attic_hd],
    ]
    af = [
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
    _add_prism_to_group(str_g, av, af)

    # ── Quadriga (compact block on top center) ──
    q_hw = 6.0
    q_hd = 3.0
    qy0 = ay1
    qy1 = ay1 + QUADRIGA_HEIGHT
    qv = [
        [cx_utm - q_hw, qy0, cz_utm - q_hd],
        [cx_utm + q_hw, qy0, cz_utm - q_hd],
        [cx_utm + q_hw, qy0, cz_utm + q_hd],
        [cx_utm - q_hw, qy0, cz_utm + q_hd],
        [cx_utm - q_hw, qy1, cz_utm - q_hd],
        [cx_utm + q_hw, qy1, cz_utm - q_hd],
        [cx_utm + q_hw, qy1, cz_utm + q_hd],
        [cx_utm - q_hw, qy1, cz_utm + q_hd],
    ]
    qf = [
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
    _add_prism_to_group(str_g, qv, qf)

    total_h = BASE_HEIGHT + COL_HEIGHT + ENTAB_HEIGHT + ATTIC_HEIGHT + QUADRIGA_HEIGHT
    logger.info("Generated Brandenburg Gate landmark geometry "
                "(base=%.0f m, top=%.0f m, total=%.1f m)",
                base_y, base_y + total_h, total_h)


def _get_landmark_exclusion_zones(bbox_transformer, area_poly):
    """Return list of (Point, radius) exclusion zones for landmarks in area."""
    zones = []
    for _name, info in _LANDMARKS.items():
        x_utm, y_utm = bbox_transformer.transform(info['lon'], info['lat'])
        if area_poly and area_poly.contains(Point(x_utm, y_utm)):
            r = info.get('exclude_radius', 0)
            if r > 0:
                zones.append(Point(x_utm, y_utm).buffer(r))
    return zones


def _inject_landmarks(bbox_transformer, area_poly, elev_fn,
                      groups, type_colors):
    """Check for known landmarks in the build area and generate geometry."""
    for name, info in _LANDMARKS.items():
        x_utm, y_utm = bbox_transformer.transform(info['lon'], info['lat'])
        if area_poly and area_poly.contains(Point(x_utm, y_utm)):
            terrain_elev = elev_fn(x_utm, y_utm)
            if name == 'statue_of_liberty':
                _generate_statue_of_liberty(
                    x_utm, y_utm, terrain_elev, groups, type_colors)
            elif name == 'sydney_opera_house':
                _generate_sydney_opera_house(
                    x_utm, y_utm, terrain_elev, groups, type_colors)
            elif name == 'cn_tower':
                _generate_cn_tower(
                    x_utm, y_utm, terrain_elev, groups, type_colors)
            elif name == 'gateway_arch':
                _generate_gateway_arch(
                    x_utm, y_utm, terrain_elev, groups, type_colors)
            elif name == 'brandenburg_gate':
                _generate_brandenburg_gate(
                    x_utm, y_utm, terrain_elev, groups, type_colors)


def _generate_wave_surface(water_poly, amplitude=3.0, wavelength=40.0,
                           grid_spacing=None, seed=42,
                           max_wave_verts=25000,
                           elev_batch_fn=None):
    """Create a sine-displaced triangulated mesh over *water_poly*.

    Returns (verts, faces) where verts are [easting, Y_height, northing].

    When *elev_batch_fn* is provided, the wave surface follows terrain
    elevation (e.g. rivers in canyons).  Otherwise sits at Y ≈ 0.

    *grid_spacing* is adaptive when ``None`` (default): the spacing is
    chosen so the wave mesh stays under *max_wave_verts* vertices,
    with a floor of 5 m (fine detail) and a ceiling of 25 m (coarse
    open ocean).  This prevents water-dominated scenes from blowing up
    to hundreds of thousands of wave vertices.
    """
    if water_poly is None or water_poly.is_empty:
        return [], []

    # Extract only polygonal parts (difference ops can produce GeometryCollections)
    if water_poly.geom_type == 'GeometryCollection':
        polys = [g for g in water_poly.geoms
                 if g.geom_type in ('Polygon', 'MultiPolygon') and not g.is_empty]
        if not polys:
            return [], []
        water_poly = unary_union(polys)
        if water_poly.is_empty:
            return [], []

    # ── Adaptive grid spacing ──────────────────────────────────────
    if grid_spacing is None:
        water_area = water_poly.area  # in UTM m²
        # Estimate grid points needed at a given spacing:
        #   n ≈ water_area / spacing²  (fill fraction <1 but boundary adds more)
        # Solve for spacing that yields ~max_wave_verts:
        #   spacing = sqrt(water_area / max_wave_verts)
        ideal = math.sqrt(water_area / max(max_wave_verts, 100))
        grid_spacing = max(5.0, min(ideal, 25.0))
        logger.info(f"Wave adaptive spacing: {grid_spacing:.1f}m "
                    f"(water area={water_area:.0f}m²)")

    rng = random.Random(seed)
    # Wave direction — random angle for variety
    angle = rng.uniform(0, 2 * math.pi)
    dx = math.cos(angle)
    dy = math.sin(angle)

    minx, miny, maxx, maxy = water_poly.bounds
    prepared = prep(water_poly)

    # Generate grid points inside water
    pts_2d = []
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if prepared.contains(Point(x, y)):
                pts_2d.append((x, y))
            y += grid_spacing
        x += grid_spacing

    # Add boundary sample points for clean edges
    boundary = water_poly.boundary
    lines = []
    if boundary is not None and not boundary.is_empty:
        if boundary.geom_type == 'MultiLineString':
            lines = list(boundary.geoms)
        else:
            lines = [boundary]
    for line in lines:
        length = line.length
        d = 0.0
        while d <= length:
            pt = line.interpolate(d)
            pts_2d.append((pt.x, pt.y))
            d += grid_spacing * 0.7

    if len(pts_2d) < 3:
        return [], []

    pts_arr = np.array(pts_2d, dtype=np.float64)

    # Delaunay triangulation on 2D positions
    try:
        tri = Delaunay(pts_arr)
    except Exception as e:
        logger.warning(f"Wave triangulation failed: {e}")
        return [], []

    # Filter triangles whose centroid falls outside water
    valid_faces = []
    for simplex in tri.simplices:
        cx = (pts_arr[simplex[0], 0] + pts_arr[simplex[1], 0] +
              pts_arr[simplex[2], 0]) / 3.0
        cy = (pts_arr[simplex[0], 1] + pts_arr[simplex[1], 1] +
              pts_arr[simplex[2], 1]) / 3.0
        if prepared.contains(Point(cx, cy)):
            # Reverse winding: Delaunay 2D CCW maps to CW in 3D
            # Y-up (downward normals).  Flipping gives upward normals
            # so the surface is visible from above.
            valid_faces.append([simplex[0], simplex[2], simplex[1]])

    if not valid_faces:
        return [], []

    # Build wave components: multiple overlapping sinusoids at different
    # frequencies, directions, and phases for a choppy, organic look.
    wave_components = [
        # (relative_amplitude, relative_wavelength, angle_offset)
        (1.0,   1.0,   0.0),       # primary swell
        (0.5,   0.55,  0.7),       # secondary cross-wave
        (0.25,  0.3,   1.9),       # short chop
        (0.15,  0.18,  3.1),       # fine ripple
        (0.35,  0.7,  -0.5),       # broad diagonal swell
    ]

    # Pre-compute direction vectors and phases for each component
    components = []
    for rel_amp, rel_wl, angle_off in wave_components:
        a = angle + angle_off
        cdx = math.cos(a)
        cdy = math.sin(a)
        wl = wavelength * rel_wl
        amp = amplitude * rel_amp
        phase = rng.uniform(0, 2 * math.pi)
        components.append((amp, wl, cdx, cdy, phase))

    # Sum of all component amplitudes for normalization
    total_amp = sum(c[0] for c in components)
    # Scale so peak height equals original 2*amplitude range
    scale_factor = amplitude / total_amp

    pts_arr_2d = np.array(pts_2d, dtype=np.float64)

    # Batch-compute terrain elevation for all wave vertices
    if elev_batch_fn is not None:
        ground = elev_batch_fn(pts_arr_2d[:, 0], pts_arr_2d[:, 1])
    else:
        ground = np.zeros(len(pts_2d), dtype=np.float64)

    verts = []
    for i, (x, y) in enumerate(pts_2d):
        h = 0.0
        for amp, wl, cdx, cdy, phase in components:
            proj = x * cdx + y * cdy
            h += amp * math.sin(2 * math.pi * proj / wl + phase)
        # Normalize and offset so surface sits above the blue base
        h = amplitude + h * scale_factor
        # Add small per-vertex noise for micro-texture
        h += rng.uniform(-0.15, 0.15) * amplitude
        verts.append([x, ground[i] + h, y])

    return verts, valid_faces


def _create_sailboat_mesh(cx, cy, rotation=0.0, scale=1.0):
    """Create a simple sailboat mesh at position (cx, cy).

    Hull: elongated hexagonal prism (~8m long, 3m beam, 1.5m draft).
    Sail: thin triangular prism (visible on 3D prints).

    Returns dict with 'hull' and 'sail' keys, each containing
    (verts, faces) in [easting, Y_height, northing] format.
    """
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    def _transform(pts):
        """Apply scale, rotation, and translation to 2D points → 3D verts."""
        result = []
        for x, z in pts:
            sx, sz = x * scale, z * scale
            rx = sx * cos_r - sz * sin_r + cx
            rz = sx * sin_r + sz * cos_r + cy
            result.append((rx, rz))
        return result

    # ── Hull profile (top-down, 2D: x=forward, z=sideways) ──
    # Pointed bow and stern, wide amidships (~24m long, 8m beam)
    hull_profile = [
        (12.0, 0.0),   # bow tip
        (7.5, 3.5),    # bow starboard
        (-7.5, 3.5),   # stern starboard
        (-12.0, 0.0),  # stern tip
        (-7.5, -3.5),  # stern port
        (7.5, -3.5),   # bow port
    ]

    hull_top = _transform(hull_profile)
    # Hull bottom — narrower (60% beam)
    hull_bottom_profile = [(x, z * 0.6) for x, z in hull_profile]
    hull_bot = _transform(hull_bottom_profile)

    n = len(hull_profile)
    hull_verts = []
    hull_faces = []

    # Raise boats above wave surface (waves peak at Y ≈ 2*amplitude = 6).
    # boat_base lifts the entire hull so deck clears wave crests.
    boat_base = 4.0
    # Top ring (deck level, above wave surface)
    deck_y = boat_base + 3.0 * scale
    for ex, nz in hull_top:
        hull_verts.append([ex, deck_y, nz])
    # Bottom ring (hull draft, sits at wave-surface level)
    draft_y = boat_base - 1.5 * scale
    for ex, nz in hull_bot:
        hull_verts.append([ex, draft_y, nz])

    # Side faces (quads split into triangles)
    for i in range(n):
        j = (i + 1) % n
        # top[i], top[j], bottom[j], bottom[i]
        hull_faces.append([i, j, j + n])
        hull_faces.append([i, j + n, i + n])

    # Top cap (deck)
    for i in range(1, n - 1):
        hull_faces.append([0, i, i + 1])

    # Bottom cap
    for i in range(1, n - 1):
        hull_faces.append([n, n + i + 1, n + i])

    # ── Sail: thin triangular prism ──
    sail_verts = []
    sail_faces = []

    # Mast position at ~25% from bow
    mast_x, mast_z = 3.0 * scale, 0.0
    rmx = mast_x * cos_r - mast_z * sin_r + cx
    rmz = mast_x * sin_r + mast_z * cos_r + cy

    sail_h = 18.0 * scale     # sail peak height
    sail_w = 7.0 * scale      # sail width at base
    sail_thick = 0.4 * scale   # thickness for 3D printing

    # Sail triangle — two sides offset by thickness
    # Points: mast base (deck), mast top, and trailing edge at deck
    # Front face
    trail_x = -5.0 * scale
    trail_z = 0.0
    rtx = trail_x * cos_r - trail_z * sin_r + cx
    rtz = trail_x * sin_r + trail_z * cos_r + cy

    # Offset perpendicular to sail plane for thickness
    perp_x = -sin_r * sail_thick / 2
    perp_z = cos_r * sail_thick / 2

    # Front triangle
    sail_verts.append([rmx + perp_x, deck_y, rmz + perp_z])          # 0: mast base front
    sail_verts.append([rmx + perp_x, deck_y + sail_h, rmz + perp_z]) # 1: mast top front
    sail_verts.append([rtx + perp_x, deck_y, rtz + perp_z])          # 2: trailing edge front

    # Back triangle
    sail_verts.append([rmx - perp_x, deck_y, rmz - perp_z])          # 3: mast base back
    sail_verts.append([rmx - perp_x, deck_y + sail_h, rmz - perp_z]) # 4: mast top back
    sail_verts.append([rtx - perp_x, deck_y, rtz - perp_z])          # 5: trailing edge back

    # Front face
    sail_faces.append([0, 1, 2])
    # Back face
    sail_faces.append([3, 5, 4])
    # Edge faces (3 edges of the triangular prism)
    sail_faces.append([0, 3, 4])
    sail_faces.append([0, 4, 1])
    sail_faces.append([1, 4, 5])
    sail_faces.append([1, 5, 2])
    sail_faces.append([2, 5, 3])
    sail_faces.append([2, 3, 0])

    return {
        'hull': (hull_verts, hull_faces),
        'sail': (sail_verts, sail_faces),
    }


def _create_deciduous_tree_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create a deciduous tree: hexagonal trunk + oblate hemisphere canopy.

    Designed for 3D printing — watertight, low-poly.  When trees are
    placed close together the rounded canopies merge into a continuous
    bumpy canopy surface.

    Returns dict with 'trunk' and 'canopy' keys, each containing
    (verts, faces) in [easting, Y_height, northing] format.
    """
    trunk_r = 0.8 * scale
    trunk_h = 4.0 * scale
    canopy_rx = 7.0 * scale   # horizontal radius
    canopy_ry = 5.0 * scale   # vertical radius (oblate)
    n_trunk = 6
    n_lon = 8   # longitude segments for canopy
    n_lat = 3   # latitude rings for canopy

    # ── Trunk (hexagonal prism) ──
    # Extend trunk 2m below terrain to prevent floating on slopes.
    # The small excess is hidden by the terrain mesh.
    trunk_bottom = ground_y - 2.0
    trunk_verts = []
    trunk_faces = []
    angles = [2 * math.pi * i / n_trunk for i in range(n_trunk)]

    # Bottom ring (slightly below terrain, hidden by terrain mesh)
    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a),
            trunk_bottom,
            cy + trunk_r * math.sin(a),
        ])
    # Top ring
    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a),
            ground_y + trunk_h,
            cy + trunk_r * math.sin(a),
        ])

    # Side faces
    for i in range(n_trunk):
        j = (i + 1) % n_trunk
        trunk_faces.append([i, j, j + n_trunk])
        trunk_faces.append([i, j + n_trunk, i + n_trunk])

    # Bottom cap
    for i in range(1, n_trunk - 1):
        trunk_faces.append([0, i + 1, i])

    # Top cap
    for i in range(1, n_trunk - 1):
        trunk_faces.append([n_trunk, n_trunk + i, n_trunk + i + 1])

    # ── Canopy (oblate hemisphere) ──
    canopy_verts = []
    canopy_faces = []
    canopy_base_y = ground_y + trunk_h

    # Equator ring (base of hemisphere) — index 0..n_lon-1
    for j in range(n_lon):
        theta = 2 * math.pi * j / n_lon
        canopy_verts.append([
            cx + canopy_rx * math.cos(theta),
            canopy_base_y,
            cy + canopy_rx * math.sin(theta),
        ])

    # Latitude rings from equator toward pole — index n_lon..n_lon*(n_lat+1)-1
    for i in range(1, n_lat + 1):
        phi = (math.pi / 2) * i / (n_lat + 1)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for j in range(n_lon):
            theta = 2 * math.pi * j / n_lon
            canopy_verts.append([
                cx + canopy_rx * cos_phi * math.cos(theta),
                canopy_base_y + canopy_ry * sin_phi,
                cy + canopy_rx * cos_phi * math.sin(theta),
            ])

    # Pole vertex
    pole_idx = len(canopy_verts)
    canopy_verts.append([cx, canopy_base_y + canopy_ry, cy])

    # Faces: connect equator ring to first latitude ring
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        # Equator ring index
        eq0 = j
        eq1 = j_next
        # First lat ring index
        lat0 = n_lon + j
        lat1 = n_lon + j_next
        canopy_faces.append([eq0, eq1, lat1])
        canopy_faces.append([eq0, lat1, lat0])

    # Faces: connect latitude rings to each other
    for i in range(1, n_lat):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            r0 = n_lon * i + j
            r1 = n_lon * i + j_next
            r2 = n_lon * (i + 1) + j
            r3 = n_lon * (i + 1) + j_next
            canopy_faces.append([r0, r1, r3])
            canopy_faces.append([r0, r3, r2])

    # Faces: connect last latitude ring to pole
    last_ring_start = n_lon * n_lat
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        canopy_faces.append([
            last_ring_start + j,
            last_ring_start + j_next,
            pole_idx,
        ])

    # Bottom cap (flat disc at equator for watertight printing)
    for i in range(1, n_lon - 1):
        canopy_faces.append([0, i + 1, i])

    return {
        'trunk': (trunk_verts, trunk_faces),
        'canopy': (canopy_verts, canopy_faces),
    }


def _create_conifer_tree_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create a conifer tree: thin trunk + cone-shaped canopy.

    Classic spruce/pine silhouette — narrow pointed top.
    Designed so total height at scale=1.0 ≈ 9m (matching _REF_TREE_H).

    Returns dict with 'trunk' and 'canopy' keys.
    """
    trunk_r = 0.4 * scale
    trunk_h = 2.0 * scale
    cone_base_r = 3.0 * scale
    cone_h = 7.0 * scale
    n_trunk = 6
    n_cone = 8

    # ── Trunk ──
    trunk_bottom = ground_y - 2.0
    trunk_verts = []
    trunk_faces = []
    angles = [2 * math.pi * i / n_trunk for i in range(n_trunk)]

    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), trunk_bottom,
            cy + trunk_r * math.sin(a)])
    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), ground_y + trunk_h,
            cy + trunk_r * math.sin(a)])

    for i in range(n_trunk):
        j = (i + 1) % n_trunk
        trunk_faces.append([i, j, j + n_trunk])
        trunk_faces.append([i, j + n_trunk, i + n_trunk])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([0, i + 1, i])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([n_trunk, n_trunk + i, n_trunk + i + 1])

    # ── Canopy (cone) ──
    canopy_verts = []
    canopy_faces = []
    cone_base_y = ground_y + trunk_h

    for j in range(n_cone):
        theta = 2 * math.pi * j / n_cone
        canopy_verts.append([
            cx + cone_base_r * math.cos(theta),
            cone_base_y,
            cy + cone_base_r * math.sin(theta)])

    apex_idx = n_cone
    canopy_verts.append([cx, cone_base_y + cone_h, cy])

    for j in range(n_cone):
        j_next = (j + 1) % n_cone
        canopy_faces.append([j, j_next, apex_idx])

    for j in range(1, n_cone - 1):
        canopy_faces.append([0, j + 1, j])

    return {
        'trunk': (trunk_verts, trunk_faces),
        'canopy': (canopy_verts, canopy_faces),
    }


def _create_palm_tree_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create a palm tree: tall thin trunk + radiating drooping fronds.

    Total height at scale=1.0 ≈ 9m (matching _REF_TREE_H).

    Returns dict with 'trunk' and 'canopy' keys.
    """
    trunk_r = 0.3 * scale
    trunk_h = 8.0 * scale
    n_trunk = 6
    n_fronds = 7
    frond_len = 4.5 * scale
    frond_half_w = 0.7 * scale
    frond_droop = 2.5 * scale

    # ── Trunk (tall and thin) ──
    trunk_bottom = ground_y - 2.0
    trunk_verts = []
    trunk_faces = []
    angles = [2 * math.pi * i / n_trunk for i in range(n_trunk)]

    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), trunk_bottom,
            cy + trunk_r * math.sin(a)])
    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), ground_y + trunk_h,
            cy + trunk_r * math.sin(a)])

    for i in range(n_trunk):
        j = (i + 1) % n_trunk
        trunk_faces.append([i, j, j + n_trunk])
        trunk_faces.append([i, j + n_trunk, i + n_trunk])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([0, i + 1, i])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([n_trunk, n_trunk + i, n_trunk + i + 1])

    # ── Canopy (radiating fronds) ──
    canopy_verts = []
    canopy_faces = []
    crown_y = ground_y + trunk_h

    # Shared center point at crown
    canopy_verts.append([cx, crown_y + 0.5 * scale, cy])

    for i in range(n_fronds):
        theta = 2 * math.pi * i / n_fronds
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        cos_p = math.cos(theta + math.pi / 2)
        sin_p = math.sin(theta + math.pi / 2)

        tip_l_idx = len(canopy_verts)
        canopy_verts.append([
            cx + frond_len * cos_t - frond_half_w * cos_p,
            crown_y - frond_droop,
            cy + frond_len * sin_t - frond_half_w * sin_p])
        canopy_verts.append([
            cx + frond_len * cos_t + frond_half_w * cos_p,
            crown_y - frond_droop,
            cy + frond_len * sin_t + frond_half_w * sin_p])
        canopy_faces.append([0, tip_l_idx, tip_l_idx + 1])

    return {
        'trunk': (trunk_verts, trunk_faces),
        'canopy': (canopy_verts, canopy_faces),
    }


def _create_broadleaf_tropical_tree_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create a tropical broadleaf tree: tall trunk + flat mushroom crown.

    Rainforest canopy tree — tall straight trunk topped by a wide,
    flat-topped crown (competing for light).  Distinctly different from
    the round deciduous dome.
    Total height at scale=1.0 ~ 9m (trunk 6 + crown 3).

    Returns dict with 'trunk' and 'canopy' keys.
    """
    trunk_r = 0.7 * scale
    trunk_h = 6.0 * scale
    crown_r = 7.0 * scale     # wide canopy spread
    crown_h = 3.0 * scale     # relatively flat
    n_trunk = 6
    n_ring = 10

    # ── Trunk (tall and straight) ──
    trunk_bottom = ground_y - 2.0
    trunk_verts = []
    trunk_faces = []
    angles = [2 * math.pi * i / n_trunk for i in range(n_trunk)]

    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), trunk_bottom,
            cy + trunk_r * math.sin(a)])
    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), ground_y + trunk_h,
            cy + trunk_r * math.sin(a)])

    for i in range(n_trunk):
        j = (i + 1) % n_trunk
        trunk_faces.append([i, j, j + n_trunk])
        trunk_faces.append([i, j + n_trunk, i + n_trunk])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([0, i + 1, i])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([n_trunk, n_trunk + i, n_trunk + i + 1])

    # ── Canopy (flat mushroom crown) ──
    # Outer edge slightly below trunk top, rises to a flat plateau,
    # then a gentle dome in the centre.
    canopy_verts = []
    canopy_faces = []
    base_y = ground_y + trunk_h
    ring_angles = [2 * math.pi * j / n_ring for j in range(n_ring)]

    # Ring 0: outer edge — droops below crown base
    for a in ring_angles:
        canopy_verts.append([
            cx + crown_r * math.cos(a),
            base_y - 0.5 * scale,
            cy + crown_r * math.sin(a)])

    # Ring 1: outer plateau — flat top at full width
    plat_r = crown_r * 0.75
    for a in ring_angles:
        canopy_verts.append([
            cx + plat_r * math.cos(a),
            base_y + crown_h * 0.85,
            cy + plat_r * math.sin(a)])

    # Ring 2: inner crown — gentle dome rise
    inner_r = crown_r * 0.35
    for a in ring_angles:
        canopy_verts.append([
            cx + inner_r * math.cos(a),
            base_y + crown_h,
            cy + inner_r * math.sin(a)])

    # Centre vertex — top of dome
    centre_idx = len(canopy_verts)
    canopy_verts.append([cx, base_y + crown_h * 1.05, cy])

    # Faces: outer edge → outer plateau
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        canopy_faces.append([j, j_next, n_ring + j_next])
        canopy_faces.append([j, n_ring + j_next, n_ring + j])

    # Faces: outer plateau → inner crown
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        r0 = n_ring + j
        r1 = n_ring + j_next
        r2 = 2 * n_ring + j
        r3 = 2 * n_ring + j_next
        canopy_faces.append([r0, r1, r3])
        canopy_faces.append([r0, r3, r2])

    # Faces: inner crown → centre
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        canopy_faces.append([2 * n_ring + j, 2 * n_ring + j_next,
                             centre_idx])

    # Bottom cap
    for j in range(1, n_ring - 1):
        canopy_faces.append([0, j + 1, j])

    return {
        'trunk': (trunk_verts, trunk_faces),
        'canopy': (canopy_verts, canopy_faces),
    }


def _create_mangrove_tree_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create a mangrove tree: short trunk with prop roots + low canopy.

    Distinctive mangrove silhouette — visible aerial prop roots splay
    out from the trunk base, supporting a low dense canopy.
    Total height at scale=1.0 ~ 5m (trunk 2.5 + canopy 2.5).

    Returns dict with 'trunk' and 'canopy' keys.
    """
    trunk_r = 0.5 * scale
    trunk_h = 2.5 * scale
    canopy_rx = 5.0 * scale
    canopy_ry = 2.5 * scale
    n_trunk = 6
    n_lon = 8
    n_lat = 2
    n_roots = 5
    root_r = 0.2 * scale
    root_spread = 2.5 * scale   # how far roots reach from trunk
    root_attach_h = 1.5 * scale  # height where root meets trunk

    # ── Trunk (central column) ──
    trunk_bottom = ground_y - 0.5  # shallow — roots do the anchoring
    trunk_verts = []
    trunk_faces = []
    angles = [2 * math.pi * i / n_trunk for i in range(n_trunk)]

    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), trunk_bottom,
            cy + trunk_r * math.sin(a)])
    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), ground_y + trunk_h,
            cy + trunk_r * math.sin(a)])

    for i in range(n_trunk):
        j = (i + 1) % n_trunk
        trunk_faces.append([i, j, j + n_trunk])
        trunk_faces.append([i, j + n_trunk, i + n_trunk])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([0, i + 1, i])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([n_trunk, n_trunk + i, n_trunk + i + 1])

    # ── Prop roots (angled struts from trunk to ground) ──
    # Each root is a triangular prism from ground level out to
    # root_spread, angling up to root_attach_h on the trunk.
    for ri in range(n_roots):
        theta = 2 * math.pi * ri / n_roots + 0.3  # offset from trunk facets
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # Perpendicular direction for root width
        cos_p = math.cos(theta + math.pi / 2)
        sin_p = math.sin(theta + math.pi / 2)

        base = len(trunk_verts)
        # Ground end of root (two points for width)
        gx = cx + root_spread * cos_t
        gz = cy + root_spread * sin_t
        trunk_verts.append([gx - root_r * cos_p, ground_y - 0.3,
                            gz - root_r * sin_p])
        trunk_verts.append([gx + root_r * cos_p, ground_y - 0.3,
                            gz + root_r * sin_p])
        # Trunk attach point (two points for width)
        ax = cx + trunk_r * 1.2 * cos_t
        az = cy + trunk_r * 1.2 * sin_t
        trunk_verts.append([ax - root_r * cos_p, ground_y + root_attach_h,
                            az - root_r * sin_p])
        trunk_verts.append([ax + root_r * cos_p, ground_y + root_attach_h,
                            az + root_r * sin_p])
        # Two triangular faces per root (front and back)
        trunk_faces.append([base, base + 1, base + 3])
        trunk_faces.append([base, base + 3, base + 2])
        trunk_faces.append([base + 1, base, base + 2])
        trunk_faces.append([base + 1, base + 2, base + 3])

    # ── Canopy (low oblate hemisphere) ──
    canopy_verts = []
    canopy_faces = []
    canopy_base_y = ground_y + trunk_h

    for j in range(n_lon):
        theta = 2 * math.pi * j / n_lon
        canopy_verts.append([
            cx + canopy_rx * math.cos(theta),
            canopy_base_y,
            cy + canopy_rx * math.sin(theta)])

    for i in range(1, n_lat + 1):
        phi = (math.pi / 2) * i / (n_lat + 1)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for j in range(n_lon):
            theta = 2 * math.pi * j / n_lon
            canopy_verts.append([
                cx + canopy_rx * cos_phi * math.cos(theta),
                canopy_base_y + canopy_ry * sin_phi,
                cy + canopy_rx * cos_phi * math.sin(theta)])

    pole_idx = len(canopy_verts)
    canopy_verts.append([cx, canopy_base_y + canopy_ry, cy])

    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        canopy_faces.append([j, j_next, n_lon + j_next])
        canopy_faces.append([j, n_lon + j_next, n_lon + j])
    for i in range(1, n_lat):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            r0 = n_lon * i + j
            r1 = n_lon * i + j_next
            r2 = n_lon * (i + 1) + j
            r3 = n_lon * (i + 1) + j_next
            canopy_faces.append([r0, r1, r3])
            canopy_faces.append([r0, r3, r2])
    last_ring_start = n_lon * n_lat
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        canopy_faces.append([last_ring_start + j, last_ring_start + j_next,
                             pole_idx])
    for i in range(1, n_lon - 1):
        canopy_faces.append([0, i + 1, i])

    return {
        'trunk': (trunk_verts, trunk_faces),
        'canopy': (canopy_verts, canopy_faces),
    }


def _create_scrub_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create scrub/brush: very low hemisphere, no visible trunk.

    Arid and polar scrubland vegetation — 2-3m wide, ~1.2m tall.
    Same structure as bush but flagged as 'scrub' for distinct color.

    Returns dict with 'trunk' and 'canopy' keys (trunk is empty).
    """
    canopy_rx = 2.0 * scale
    canopy_ry = 1.2 * scale
    n_lon = 6
    n_lat = 2

    verts = []
    faces = []

    # Equator ring (base on ground)
    for j in range(n_lon):
        theta = 2 * math.pi * j / n_lon
        verts.append([
            cx + canopy_rx * math.cos(theta),
            ground_y,
            cy + canopy_rx * math.sin(theta)])

    for i in range(1, n_lat + 1):
        phi = (math.pi / 2) * i / (n_lat + 1)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for j in range(n_lon):
            theta = 2 * math.pi * j / n_lon
            verts.append([
                cx + canopy_rx * cos_phi * math.cos(theta),
                ground_y + canopy_ry * sin_phi,
                cy + canopy_rx * cos_phi * math.sin(theta)])

    pole_idx = len(verts)
    verts.append([cx, ground_y + canopy_ry, cy])

    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([j, j_next, n_lon + j_next])
        faces.append([j, n_lon + j_next, n_lon + j])
    for i in range(1, n_lat):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            r0 = n_lon * i + j
            r1 = n_lon * i + j_next
            r2 = n_lon * (i + 1) + j
            r3 = n_lon * (i + 1) + j_next
            faces.append([r0, r1, r3])
            faces.append([r0, r3, r2])
    last_ring_start = n_lon * n_lat
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([last_ring_start + j, last_ring_start + j_next,
                      pole_idx])
    for j in range(1, n_lon - 1):
        faces.append([0, j + 1, j])

    return {
        'trunk': ([], []),
        'canopy': (verts, faces),
    }


def _create_sclerophyll_tree_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create a sclerophyll tree: medium trunk + flat umbrella canopy.

    Mediterranean olive / umbrella pine silhouette — wide, flat-topped
    parasol shape, clearly distinct from the rounded deciduous dome.
    Total height at scale=1.0 ~ 5.5m (trunk 3.5 + canopy 2.0).

    Returns dict with 'trunk' and 'canopy' keys.
    """
    trunk_r = 0.5 * scale
    trunk_h = 3.5 * scale
    canopy_r = 5.5 * scale    # wide spread
    canopy_h = 2.0 * scale    # very flat — parasol shape
    n_trunk = 6
    n_ring = 10

    # ── Trunk ──
    trunk_bottom = ground_y - 2.0
    trunk_verts = []
    trunk_faces = []
    angles = [2 * math.pi * i / n_trunk for i in range(n_trunk)]

    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), trunk_bottom,
            cy + trunk_r * math.sin(a)])
    for a in angles:
        trunk_verts.append([
            cx + trunk_r * math.cos(a), ground_y + trunk_h,
            cy + trunk_r * math.sin(a)])

    for i in range(n_trunk):
        j = (i + 1) % n_trunk
        trunk_faces.append([i, j, j + n_trunk])
        trunk_faces.append([i, j + n_trunk, i + n_trunk])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([0, i + 1, i])
    for i in range(1, n_trunk - 1):
        trunk_faces.append([n_trunk, n_trunk + i, n_trunk + i + 1])

    # ── Canopy (flat umbrella / parasol) ──
    # Three rings: outer droop edge, mid rim, inner rim + flat top centre
    canopy_verts = []
    canopy_faces = []
    base_y = ground_y + trunk_h
    ring_angles = [2 * math.pi * j / n_ring for j in range(n_ring)]

    # Ring 0: outer edge — droops slightly below trunk top
    for a in ring_angles:
        canopy_verts.append([
            cx + canopy_r * math.cos(a),
            base_y - 0.3 * scale,
            cy + canopy_r * math.sin(a)])

    # Ring 1: mid canopy — at full height
    mid_r = canopy_r * 0.6
    for a in ring_angles:
        canopy_verts.append([
            cx + mid_r * math.cos(a),
            base_y + canopy_h,
            cy + mid_r * math.sin(a)])

    # Ring 2: inner canopy — slightly lower, near trunk
    inner_r = canopy_r * 0.25
    for a in ring_angles:
        canopy_verts.append([
            cx + inner_r * math.cos(a),
            base_y + canopy_h * 0.85,
            cy + inner_r * math.sin(a)])

    # Centre vertex
    centre_idx = len(canopy_verts)
    canopy_verts.append([cx, base_y + canopy_h * 0.8, cy])

    # Faces: outer edge → mid rim
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        canopy_faces.append([j, j_next, n_ring + j_next])
        canopy_faces.append([j, n_ring + j_next, n_ring + j])

    # Faces: mid rim → inner rim
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        r0 = n_ring + j
        r1 = n_ring + j_next
        r2 = 2 * n_ring + j
        r3 = 2 * n_ring + j_next
        canopy_faces.append([r0, r1, r3])
        canopy_faces.append([r0, r3, r2])

    # Faces: inner rim → centre
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        canopy_faces.append([2 * n_ring + j, 2 * n_ring + j_next,
                             centre_idx])

    # Bottom cap (flat disc at outer edge for watertight)
    for j in range(1, n_ring - 1):
        canopy_faces.append([0, j + 1, j])

    return {
        'trunk': (trunk_verts, trunk_faces),
        'canopy': (canopy_verts, canopy_faces),
    }


def _create_bush_mesh(cx, cy, scale=1.0, ground_y=0.0):
    """Create a bush: low, wide hemisphere mound sitting on ground.

    No trunk — just a flattened dome for low vegetation.
    Watertight for 3D printing.

    Returns dict with 'canopy' key containing (verts, faces).
    """
    bush_rx = 3.0 * scale   # horizontal radius
    bush_ry = 1.5 * scale   # vertical radius (half height)
    n_lon = 6
    n_lat = 2

    verts = []
    faces = []

    # Equator ring (base) — index 0..n_lon-1
    for j in range(n_lon):
        theta = 2 * math.pi * j / n_lon
        verts.append([
            cx + bush_rx * math.cos(theta),
            ground_y,
            cy + bush_rx * math.sin(theta),
        ])

    # Latitude rings
    for i in range(1, n_lat + 1):
        phi = (math.pi / 2) * i / (n_lat + 1)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for j in range(n_lon):
            theta = 2 * math.pi * j / n_lon
            verts.append([
                cx + bush_rx * cos_phi * math.cos(theta),
                ground_y + bush_ry * sin_phi,
                cy + bush_rx * cos_phi * math.sin(theta),
            ])

    # Pole
    pole_idx = len(verts)
    verts.append([cx, ground_y + bush_ry, cy])

    # Connect equator to first lat ring
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        eq0, eq1 = j, j_next
        lat0 = n_lon + j
        lat1 = n_lon + j_next
        faces.append([eq0, eq1, lat1])
        faces.append([eq0, lat1, lat0])

    # Connect lat rings
    for i in range(1, n_lat):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            r0 = n_lon * i + j
            r1 = n_lon * i + j_next
            r2 = n_lon * (i + 1) + j
            r3 = n_lon * (i + 1) + j_next
            faces.append([r0, r1, r3])
            faces.append([r0, r3, r2])

    # Last ring to pole
    last_ring_start = n_lon * n_lat
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([last_ring_start + j, last_ring_start + j_next, pole_idx])

    # Bottom cap
    for i in range(1, n_lon - 1):
        faces.append([0, i + 1, i])

    return {
        'canopy': (verts, faces),
    }


def _classify_canopy_zones(tree_positions, cell_size=10.0, min_neighbors=2):
    """Split tree positions into dense-canopy zones and individual trees.

    Uses a grid to find locally dense clusters.  Trees in cells where
    the cell AND most of its 8 neighbours contain >= *min_neighbors* trees
    are classified as "canopy" (continuous forest).  The rest remain
    "individual" (urban, edges, sparse areas).

    Returns (canopy_positions, individual_positions).
    """
    if not tree_positions:
        return [], tree_positions

    # Build grid index
    cells = {}
    for i, (x, y) in enumerate(tree_positions):
        cx = int(x // cell_size)
        cy = int(y // cell_size)
        cells.setdefault((cx, cy), []).append(i)

    # Mark dense cells — cell + neighbours all populated
    dense_cells = set()
    for (cx, cy), indices in cells.items():
        if len(indices) < min_neighbors:
            continue
        # Count how many of the 8 neighbours also have trees
        populated_neighbors = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (cx + dx, cy + dy) in cells:
                    populated_neighbors += 1
        # Need at least 5 of 8 neighbours populated for continuous canopy
        if populated_neighbors >= 5:
            dense_cells.add((cx, cy))

    canopy = []
    individual = []
    for (cx, cy), indices in cells.items():
        if (cx, cy) in dense_cells:
            for i in indices:
                canopy.append(tree_positions[i])
        else:
            for i in indices:
                individual.append(tree_positions[i])

    return canopy, individual


def _create_canopy_surface(tree_positions, elev_fn, rng,
                           thickness=0.25, trunk_h=4.0, canopy_ry=5.0,
                           max_edge_len=18.0):
    """Create a continuous canopy shell mesh from dense tree positions.

    Instead of individual tree meshes, builds a single bumpy surface
    at canopy height with a hollow interior (top + bottom + side walls).

    Parameters
    ----------
    tree_positions : list of (x, y) — UTM coordinates
    elev_fn : callable(x, y) → float — terrain elevation
    rng : random.Random — for height variation
    thickness : float — shell wall thickness in metres (~10 inches)
    trunk_h : float — average trunk height
    canopy_ry : float — average canopy vertical radius
    max_edge_len : float — max triangle edge before discarding (prevents
        spanning gaps between tree clusters)

    Returns (verts, faces) in [easting, Y_height, northing] format.
    """
    if len(tree_positions) < 3:
        return [], []

    pts_2d = np.array(tree_positions, dtype=np.float64)

    # Delaunay triangulation on the tree positions
    try:
        tri = Delaunay(pts_2d)
    except Exception:
        return [], []

    # Filter triangles with excessively long edges (span gaps)
    max_edge_sq = max_edge_len ** 2
    valid_faces = []
    for simplex in tri.simplices:
        p0, p1, p2 = pts_2d[simplex[0]], pts_2d[simplex[1]], pts_2d[simplex[2]]
        e01 = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
        e12 = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
        e20 = (p0[0] - p2[0]) ** 2 + (p0[1] - p2[1]) ** 2
        if e01 <= max_edge_sq and e12 <= max_edge_sq and e20 <= max_edge_sq:
            # Flip winding for upward-facing normals (Y-up)
            valid_faces.append([simplex[0], simplex[2], simplex[1]])

    if not valid_faces:
        return [], []

    n_pts = len(pts_2d)

    # Build top surface — canopy height with random bumps
    top_verts = []
    for x, y in pts_2d:
        ground = elev_fn(x, y)
        # Canopy top = ground + trunk + canopy radius + random bump
        bump = rng.uniform(-1.5, 2.0)
        h = ground + trunk_h + canopy_ry + bump
        top_verts.append([x, h, y])

    # Top surface only — bottom surface creates a visible gap at
    # bbox boundaries when viewed at oblique angles.
    return top_verts, valid_faces


def _drape_on_terrain(poly, elev_fn, origin, y_offset=0.15,
                      max_tri_area=200.0, thickness=0.0,
                      elev_batch_fn=None):
    """Triangulate *poly* and place each vertex at terrain elevation + offset.

    Returns (verts_3d, faces) where verts are [easting, Y, northing] (Y-up).

    When *thickness* > 0, builds a solid 3D slab: top surface, bottom surface,
    and side walls connecting them.  When 0, produces a single thin surface.

    *max_tri_area* controls triangle density (smaller = more vertices).

    *elev_batch_fn*: optional vectorized elevation function taking (xs, ys)
    arrays.  When provided, skips the per-point Python loop.

    Uses scipy.spatial.Delaunay with vectorized shapely.contains_xy for
    fast containment checks (replaces per-element Python loops).
    """
    import shapely

    # Sanitise geometry.
    try:
        poly = poly.buffer(0)
        if poly.is_empty:
            return [], []
        poly = poly.simplify(0.1, preserve_topology=True)
        if poly.is_empty:
            return [], []
    except Exception:
        return [], []

    ox, oy = origin

    if poly.geom_type == 'MultiPolygon':
        polys = list(poly.geoms)
    elif poly.geom_type == 'Polygon':
        polys = [poly]
    else:
        return [], []

    # Collect boundary vertices from all polygons
    boundary_pts = []
    for p in polys:
        if p.is_empty or p.area < 0.1:
            continue
        coords = list(p.exterior.coords)
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        boundary_pts.extend((c[0], c[1]) for c in coords)
        for interior in p.interiors:
            coords = list(interior.coords)
            if coords and coords[0] == coords[-1]:
                coords = coords[:-1]
            boundary_pts.extend((c[0], c[1]) for c in coords)

    if len(boundary_pts) < 3:
        return [], []

    # Generate interior grid points so the surface follows terrain contours.
    # Uses vectorized shapely.contains_xy instead of per-point Python loop.
    grid_spacing = max(math.sqrt(max_tri_area), 5.0)
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + grid_spacing / 2, maxx, grid_spacing)
    ys = np.arange(miny + grid_spacing / 2, maxy, grid_spacing)
    if len(xs) > 0 and len(ys) > 0:
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        mask = shapely.contains_xy(poly, candidates[:, 0], candidates[:, 1])
        grid_pts = candidates[mask]
    else:
        grid_pts = np.empty((0, 2), dtype=np.float64)

    # Combine boundary + interior points
    boundary_arr = np.array(boundary_pts, dtype=np.float64)
    if len(grid_pts) > 0:
        pts_array = np.vstack([boundary_arr, grid_pts])
    else:
        pts_array = boundary_arr

    # Remove near-duplicate points (within 1cm) to avoid Qhull errors
    pts_rounded = np.round(pts_array, 2)
    _, unique_idx = np.unique(pts_rounded, axis=0, return_index=True)
    pts_array = pts_array[np.sort(unique_idx)]

    if len(pts_array) < 3:
        return [], []

    # Delaunay triangulation (Qhull-based, no segfault risk)
    try:
        tri = Delaunay(pts_array)
    except Exception as e:
        logger.warning(f"Delaunay triangulation failed: {e}")
        return [], []

    # Filter triangles: vectorized centroid containment check
    simplices = tri.simplices
    centroids_x = pts_array[simplices, 0].mean(axis=1)
    centroids_y = pts_array[simplices, 1].mean(axis=1)
    inside = shapely.contains_xy(poly, centroids_x, centroids_y)
    valid_simplices = simplices[inside]

    if len(valid_simplices) == 0:
        return [], []

    # ── Top surface vertices: terrain elevation + y_offset ──
    if elev_batch_fn is not None:
        heights = elev_batch_fn(pts_array[:, 0], pts_array[:, 1])
    else:
        heights = np.array([elev_fn(vx, vy) for vx, vy in pts_array])
    verts_3d = np.column_stack([
        pts_array[:, 0] - ox,
        heights + y_offset,
        pts_array[:, 1] - oy,
    ]).tolist()

    # Flip winding for +Y normals (same as terrain mesh)
    top_faces = valid_simplices[:, [0, 2, 1]]

    return verts_3d, top_faces.tolist()


def generate_glb(db, city_id: int, output_path: str,
                 progress_callback=None) -> str:
    """Generate GLB (binary glTF) file from stored city data.

    Creates separate meshes per feature type, each with a solid PBR
    material colour.  Only the horizontal axes (X / Z) are centred;
    Y (height) is kept as-is so ground stays at Y ≈ 0.

    Returns the absolute path to the generated GLB file.
    """
    from shapely.wkb import loads as wkb_loads

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    output_path = PathManager.get_output_path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    _progress(62, "Extruding buildings...")
    logger.info("Generating GLB file...")
    _t0 = time.perf_counter()
    _timings = {}  # label → seconds

    features = db.get_city_features(city_id)

    # ── Layer stacking ──────────────
    #   Layer 1  blue base    Y = -20 →  0       entire area (no query)
    #   Layer 2  terrain      Y =  0  → elev     CDT mesh with SRTM elevation
    #   Layer 3  roads        Y = elev + 0.15    OSM road features
    #   Layer 4  buildings    Y = elev → elev+h  OSM building features
    # When SRTM is unavailable, falls back to flat green at Y=0→10.
    # Water is visible as the blue base showing through terrain holes.

    groups: dict[str, dict] = {
        'building':   {'verts': [], 'faces': [], 'offset': 0},
        'road':       {'verts': [], 'faces': [], 'offset': 0},
        'bridge':     {'verts': [], 'faces': [], 'offset': 0},
        'foundation': {'verts': [], 'faces': [], 'offset': 0},
        'wall':       {'verts': [], 'faces': [], 'offset': 0},
        'windows':    {'verts': [], 'faces': [], 'offset': 0},
    }

    # Solid PBR colours per type (RGBA, 0-1)
    type_colors = {
        'building':     [0.95, 0.95, 0.95, 1.0],   # white
        'blue_base':    [0.25, 0.52, 0.85, 1.0],   # blue (water)
        'road':         [0.25, 0.25, 0.25, 1.0],   # dark asphalt
        'green':        [0.30, 0.65, 0.30, 1.0],   # green
        'paved':        [0.25, 0.25, 0.25, 1.0],   # dark asphalt parking
        'bridge':       [0.65, 0.65, 0.62, 1.0],   # light grey (bridge)
        'wave':         [0.20, 0.45, 0.80, 1.0],   # slightly darker blue
        'boat_hull':    [0.95, 0.95, 0.95, 1.0],   # white
        'boat_sail':    [0.98, 0.98, 0.95, 1.0],   # off-white
        'tree_trunk':   [0.45, 0.30, 0.15, 1.0],   # brown
        'tree_canopy':    [0.20, 0.50, 0.20, 1.0],   # deciduous dark green
        'conifer_canopy': [0.10, 0.28, 0.10, 1.0],   # dark spruce green
        'palm_canopy':    [0.18, 0.48, 0.15, 1.0],   # tropical green
        'broadleaf_canopy':   [0.12, 0.42, 0.12, 1.0],   # rich tropical green
        'mangrove_canopy':    [0.15, 0.38, 0.12, 1.0],   # dark olive
        'scrub_canopy':       [0.40, 0.48, 0.28, 1.0],   # olive-grey
        'sclerophyll_canopy': [0.28, 0.45, 0.22, 1.0],   # grey-green
        'bush_canopy':    [0.25, 0.55, 0.25, 1.0],   # slightly lighter green
        'foundation':   [0.55, 0.50, 0.45, 1.0],   # darker stone
        'pitch':        [0.42, 0.65, 0.35, 1.0],   # sports field green
        'park':         [0.30, 0.60, 0.28, 1.0],   # natural park green
        'track':        [0.72, 0.45, 0.32, 1.0],   # reddish-brown (tartan)
        'pool':         [0.45, 0.75, 0.88, 1.0],   # light cyan (pool water)
        'railway':      [0.28, 0.28, 0.28, 1.0],   # dark iron grey
        'pier':         [0.62, 0.47, 0.30, 1.0],   # wood brown
        'rock':         [0.55, 0.52, 0.48, 1.0],   # grey-brown alpine rock
        'wall':         [0.50, 0.50, 0.48, 1.0],   # gray rock
        'glacier':      [0.95, 0.96, 0.97, 1.0],   # white (snow/ice)
        'windows':      [0.12, 0.12, 0.15, 1.0],   # dark glass
    }

    # ── Compute area bounds early (needed for terrain grid) ──
    building_footprints_utm = []  # collect for terrain mesh hole-cutting
    city_state = db.get_city_state(city_id)
    bbox_transformer = None
    terrain_available = False
    terrain_bbox = None
    grid_x = grid_y = elev_2d = None
    elev_offset = 0.0
    x_center = z_center = 0.0
    crop_xmin = crop_ymin = crop_xmax = crop_ymax = 0.0
    clat = 0.0
    clon = 0.0
    area_poly = None

    if city_state and city_state.get('bbox'):
        cb = city_state['bbox']
        clat = (cb.north + cb.south) / 2
        clon = (cb.east + cb.west) / 2
        uzone = int((clon + 180) / 6) + 1
        uepsg = 32600 + uzone if clat >= 0 else 32700 + uzone
        bbox_transformer = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{uepsg}", always_xy=True)
        x_min_utm, y_min_utm = bbox_transformer.transform(cb.west, cb.south)
        x_max_utm, y_max_utm = bbox_transformer.transform(cb.east, cb.north)

        # Save original bbox bounds — used to crop the final model
        crop_xmin, crop_ymin = x_min_utm, y_min_utm
        crop_xmax, crop_ymax = x_max_utm, y_max_utm
        logger.info(f"Original bbox: "
                     f"{crop_xmax - crop_xmin:.0f} x {crop_ymax - crop_ymin:.0f} m")

        # Expand area to match the ~200m query buffer in osm_data.py.
        # We build the full model in this expanded area, then crop
        # the final GLB back to the original bbox for clean edges.
        _QUERY_BUFFER_DEG = 0.002  # must match osm_data._QUERY_BUFFER_DEG
        exp_w, exp_s = bbox_transformer.transform(
            cb.west  - _QUERY_BUFFER_DEG, cb.south - _QUERY_BUFFER_DEG)
        exp_e, exp_n = bbox_transformer.transform(
            cb.east  + _QUERY_BUFFER_DEG, cb.north + _QUERY_BUFFER_DEG)
        x_min_utm = exp_w
        y_min_utm = exp_s
        x_max_utm = exp_e
        y_max_utm = exp_n

        # Also check buildings — they may overhang the buffer slightly
        for ft, osm_id, geom_blob, props_json, src, conf in features:
            if ft != 'building':
                continue
            bgeom = wkb_loads(geom_blob)
            bx0, by0, bx1, by1 = bgeom.bounds
            x_min_utm = min(x_min_utm, bx0)
            y_min_utm = min(y_min_utm, by0)
            x_max_utm = max(x_max_utm, bx1)
            y_max_utm = max(y_max_utm, by1)

        # Pad so terrain extends slightly past edges
        _TERRAIN_PAD = 5.0
        x_min_utm -= _TERRAIN_PAD
        y_min_utm -= _TERRAIN_PAD
        x_max_utm += _TERRAIN_PAD
        y_max_utm += _TERRAIN_PAD

        area_poly = Polygon([
            (x_min_utm, y_min_utm), (x_min_utm, y_max_utm),
            (x_max_utm, y_max_utm), (x_max_utm, y_min_utm),
        ])
        x_center = (crop_xmin + crop_xmax) / 2
        z_center = (crop_ymin + crop_ymax) / 2
        logger.info(f"Expanded area: "
                     f"{x_max_utm - x_min_utm:.0f} x {y_max_utm - y_min_utm:.0f} m")

        # Compute expanded WGS84 bbox for terrain elevation grid
        inv_tf = Transformer.from_crs(
            bbox_transformer.target_crs, "EPSG:4326", always_xy=True)
        ew, es = inv_tf.transform(x_min_utm, y_min_utm)
        ee, en = inv_tf.transform(x_max_utm, y_max_utm)
        terrain_bbox = BoundingBox(north=en, south=es, east=ee, west=ew)

        # Sample terrain elevation grid — 10m resolution
        area_span = max(x_max_utm - x_min_utm, y_max_utm - y_min_utm)
        terrain_spacing = 10.0

        logger.info(f"Area span: {area_span:.0f}m")
        try:
            grid_x, grid_y, elev_2d, elev_offset, terrain_available = \
                terrain_mod.get_elevation_grid(
                    terrain_bbox, bbox_transformer, spacing=terrain_spacing)
            if terrain_available:
                logger.info(f"Terrain enabled: offset={elev_offset:.0f}m, "
                             f"range={elev_2d.max():.0f}m")
            else:
                logger.info("SRTM unavailable — using flat terrain")
        except Exception as e:
            logger.warning(f"Terrain grid failed: {e} — using flat terrain")
            terrain_available = False

    def _elev_at(x, y):
        """Get terrain elevation at UTM point (10.0 if terrain unavailable)."""
        if not terrain_available:
            return 10.0  # legacy flat ground level
        return terrain_mod.sample_elevation_at(x, y, grid_x, grid_y, elev_2d)

    def _elev_batch(xs, ys):
        """Vectorized elevation lookup for arrays of UTM points."""
        if not terrain_available:
            return np.full(len(xs), 10.0)
        return terrain_mod.sample_elevation_batch(xs, ys, grid_x, grid_y,
                                                  elev_2d)

    # Collect water and paved geometries (projected coords) for green subtraction
    water_geoms = []
    paved_geoms = []
    pitch_geoms = []
    track_geoms = []
    pool_geoms = []
    railway_lines = []
    pier_geoms = []
    park_geoms = []
    glacier_geoms = []
    road_groups: dict[float, list] = {}    # width → list of LineString/MultiLineString
    bridge_groups: dict[float, list] = {}  # width → list of bridge road geometries

    # Pre-compute landmark exclusion zones so OSM buildings near landmarks
    # are skipped (we generate our own procedural geometry for them).
    landmark_excl = []
    if bbox_transformer and area_poly:
        landmark_excl = _get_landmark_exclusion_zones(
            bbox_transformer, area_poly)

    for feature_type, osm_id, geometry_blob, properties_json, source, confidence in features:
        try:
            geometry = wkb_loads(geometry_blob)
            properties = json.loads(properties_json)

            if feature_type == 'building':
                # Buildings in landmark zones: flat roof, podium height
                in_landmark_zone = False
                if landmark_excl:
                    centroid = geometry.centroid
                    in_landmark_zone = any(
                        zone.contains(centroid) for zone in landmark_excl)
                height = float(properties.get('height', 10))
                if math.isnan(height) or height <= 0:
                    height = 10.0
                # Support min_height for floating building parts (setbacks)
                min_h = float(properties.get('min_height', 0))
                if math.isnan(min_h) or min_h < 0:
                    min_h = 0.0

                # Landmark zone: flat roof, cap at podium height (8 m)
                if in_landmark_zone:
                    height = 8.0
                    min_h = 0.0
                    properties['roof_shape'] = 'flat'

                extrude_height = height - min_h
                if extrude_height <= 0:
                    extrude_height = height
                    min_h = 0.0

                # Sample terrain at building footprint
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    if geometry.geom_type == 'Polygon':
                        fp_coords = list(geometry.exterior.coords[:-1])
                    else:
                        fp_coords = list(geometry.geoms[0].exterior.coords[:-1])
                    terrain_heights = [_elev_at(x, y) for x, y in fp_coords]
                    terrain_max = max(terrain_heights)
                    base_y = terrain_max + min_h

                    # Collect footprint for terrain hole-cutting
                    if terrain_available:
                        building_footprints_utm.append(geometry)

                        # Foundation skirt for sloped terrain
                        if terrain_max - min(terrain_heights) > 0.3:
                            skirt = terrain_mod.build_foundation_skirt(
                                fp_coords, terrain_heights, terrain_max,
                                (0, 0))
                            if skirt:
                                skirt_verts, skirt_faces = skirt
                                g = groups['foundation']
                                off = g['offset']
                                for f in skirt_faces:
                                    g['faces'].append(
                                        [f[0] + off, f[1] + off,
                                         f[2] + off])
                                g['verts'].extend(skirt_verts)
                                g['offset'] += len(skirt_verts)
                else:
                    base_y = _elev_at(
                        geometry.centroid.x, geometry.centroid.y) + min_h

                roof_shape = properties.get('roof_shape', '')

                r_h = properties.get('roof_height')
                has_roof = (roof_shape and roof_shape != 'flat'
                            and roof_shape.lower().replace('-', '_').replace(' ', '_')
                            in {s.replace('-', '_') for s in ROOF_SHAPES_SUPPORTED})

                if has_roof:
                    # Compute shape-specific default roof height
                    shape_norm = (roof_shape.lower()
                                  .replace('-', '_').replace(' ', '_'))
                    default_roof_h = 0.0
                    try:
                        bounds = geometry.bounds  # (minx, miny, maxx, maxy) in UTM
                        bbox_w = bounds[2] - bounds[0]
                        bbox_d = bounds[3] - bounds[1]
                        min_dim = min(bbox_w, bbox_d)
                        if shape_norm in ('pyramidal', 'dome', 'round'):
                            default_roof_h = min_dim / 2.0
                        elif shape_norm == 'onion':
                            default_roof_h = min_dim * 0.75
                        elif shape_norm in ('skillion', 'lean_to'):
                            default_roof_h = math.cos(
                                math.radians(22.5)) * min_dim
                        elif shape_norm in ('gabled', 'hipped'):
                            # ~30% of short side ≈ 6:12 pitch
                            default_roof_h = min_dim * 0.3
                    except Exception:
                        pass

                    if r_h is not None:
                        try:
                            roof_h = float(r_h)
                            if math.isnan(roof_h) or roof_h <= 0:
                                roof_h = default_roof_h
                        except (ValueError, TypeError):
                            roof_h = default_roof_h
                    else:
                        roof_h = default_roof_h

                    # If roof_h is 0 (no default for this shape and
                    # no explicit roof:height), skip roof rendering
                    if roof_h <= 0:
                        has_roof = False
                        wall_h = extrude_height
                    else:
                        wall_h = max(extrude_height - roof_h, 0)

                    # Spec-compliant: wall_h cylinder + roof_h cone.
                    # 7m-wide cones are physically visible spires.

                is_roof_part = properties.get('part_type') == 'roof'

                if is_roof_part and has_roof:
                    # building:part=roof — render ONLY the roof mesh,
                    # no wall extrusion (per OSMBuilding approach).
                    # Eave sits at min_height (base_y), peak at height.
                    eave_y = base_y
                    roof_h = extrude_height  # entire height is roof
                    new_verts, new_faces = [], []
                    roof_dir = properties.get('roof_direction')
                    rv, rf = generate_roof_mesh(
                        geometry, roof_shape, roof_h, eave_y,
                        roof_direction=roof_dir)
                    if rv and rf:
                        new_verts.extend(rv)
                        new_faces.extend(rf)
                elif has_roof:
                    if wall_h > 0:
                        new_verts, new_faces = extrude_watertight(
                            geometry, wall_h, base_y=base_y)
                    else:
                        new_verts, new_faces = [], []
                    eave_y = base_y + wall_h
                    roof_dir = properties.get('roof_direction')
                    rv, rf = generate_roof_mesh(
                        geometry, roof_shape, roof_h, eave_y,
                        roof_direction=roof_dir)
                    if rv and rf:
                        off = len(new_verts)
                        new_verts.extend(rv)
                        for f in rf:
                            new_faces.append([f[0] + off,
                                              f[1] + off,
                                              f[2] + off])
                else:
                    # Flat roof (or flat roof part) — full extrusion
                    new_verts, new_faces = extrude_watertight(
                        geometry, extrude_height, base_y=base_y)
                group_key = 'building'

                # ── Window dimples on large commercial/industrial facades ──
                if (not is_roof_part
                        and extrude_height >= 12.0
                        and geometry.area >= 150.0
                        and _building_wants_windows(properties)):
                    eff_wall_h = wall_h if has_roof else extrude_height
                    if eff_wall_h >= 10.0:
                        win_v, win_f = generate_window_quads(
                            geometry, eff_wall_h, base_y)
                        if win_v:
                            _add_prism_to_group(groups['windows'],
                                                win_v, win_f)

            elif feature_type == 'water':
                # Don't render water meshes — just collect geometry
                # for subtracting from the green layer.
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    water_geoms.append(geometry)
                continue
            elif feature_type == 'paved':
                # Collect for green subtraction; rendered as its own layer below.
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    paved_geoms.append(geometry)
                continue
            elif feature_type == 'pitch':
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    sport = properties.get('sport', '')
                    if (sport == 'athletics'
                            or properties.get('type') == 'track'):
                        track_geoms.append(geometry)
                    else:
                        pitch_geoms.append(geometry)
                continue
            elif feature_type == 'pool':
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    pool_geoms.append(geometry)
                continue
            elif feature_type == 'park':
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    park_geoms.append(geometry)
                continue
            elif feature_type == 'railway':
                railway_lines.append(geometry)
                continue
            elif feature_type == 'pier':
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    pier_geoms.append(geometry)
                elif geometry.geom_type in ('LineString', 'MultiLineString'):
                    pier_geoms.append(geometry.buffer(3.0))
                continue
            elif feature_type == 'glacier':
                if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    glacier_geoms.append(geometry)
                continue
            elif feature_type == 'road':
                # Skip tunnel roads entirely — they're underground
                if properties.get('tunnel') and properties['tunnel'] != 'no':
                    continue
                # Determine road width from type
                road_type = properties.get('type', 'unknown')
                road_widths = FEATURE_CATEGORIES['roads']['style']['widths']
                road_width = road_widths.get('local_road', 8.0)
                for category, types in FEATURE_CATEGORIES['roads']['types'].items():
                    if road_type in types:
                        road_width = road_widths[category]
                        break
                # Route bridge roads to elevated bridge group
                if properties.get('bridge') and properties['bridge'] != 'no':
                    if road_width not in bridge_groups:
                        bridge_groups[road_width] = []
                    bridge_groups[road_width].append(geometry)
                else:
                    if road_width not in road_groups:
                        road_groups[road_width] = []
                    road_groups[road_width].append(geometry)
                continue
            elif feature_type == 'bridge':
                # Bridge deck polygons (man_made=bridge) — drape on terrain
                # using the max boundary elevation so the deck sits at
                # bank height rather than dipping over water.
                if terrain_available and geometry.geom_type in ('Polygon', 'MultiPolygon'):
                    polys = (list(geometry.geoms) if geometry.geom_type == 'MultiPolygon'
                             else [geometry])
                    boundary_elevs = []
                    for bp in polys:
                        for bx, by in bp.exterior.coords:
                            boundary_elevs.append(_elev_at(bx, by))
                    deck_elev = max(boundary_elevs) if boundary_elevs else 10.0
                    new_verts, new_faces = extrude_watertight(
                        geometry, 2.0, base_y=deck_elev)
                else:
                    new_verts, new_faces = extrude_watertight(
                        geometry, 2.0, base_y=10.0)
                group_key = 'bridge'
            elif feature_type == 'green':
                # Skip — green is computed, not queried
                continue
            else:
                continue

            if new_verts and new_faces:
                group = groups[group_key]
                off = group['offset']
                for f in new_faces:
                    group['faces'].append([f[0] + off, f[1] + off, f[2] + off])
                group['verts'].extend(new_verts)
                group['offset'] += len(new_verts)
        except Exception as e:
            logger.error(f"Error processing feature for GLB: {e}")
            continue

    # ── Landmark injection (procedural geometry for famous structures) ──
    if bbox_transformer and area_poly:
        _inject_landmarks(bbox_transformer, area_poly, _elev_at,
                          groups, type_colors)

    # ── Pre-compute water union for clipping roads/rails off water ──
    water_union = None
    if water_geoms:
        try:
            water_union = unary_union(water_geoms)
        except Exception as e:
            logger.warning(f"Water union for road clip failed: {e}")

    _timings['1_features_terrain'] = time.perf_counter() - _t0
    # ── Batch-process roads: buffer individually, union in chunks, extrude ──
    _t0 = time.perf_counter()
    _progress(72, "Building road surfaces...")
    road_polys_for_subtract = []
    ROAD_CHUNK = 500
    for width, lines in road_groups.items():
        try:
            half_w = width / 2.0
            # Buffer each line into a polygon first (robust)
            buffered_parts = []
            for line in lines:
                b = line.buffer(half_w, cap_style=2, resolution=4)
                if not b.is_empty:
                    buffered_parts.append(b)
            if not buffered_parts:
                continue

            # Union in manageable chunks to avoid geometric explosion
            for ci in range(0, len(buffered_parts), ROAD_CHUNK):
                chunk = buffered_parts[ci:ci + ROAD_CHUNK]
                try:
                    merged = unary_union(chunk)
                    merged = merged.simplify(0.5, preserve_topology=True)
                    if merged.is_empty:
                        continue
                    # Clip roads off water — prevents surface roads
                    # rendering over rivers/ocean (tunnels are already
                    # filtered; bridges are processed separately).
                    if water_union and not water_union.is_empty:
                        merged = merged.difference(water_union)
                        if merged.is_empty:
                            continue
                    road_polys_for_subtract.append(merged)
                    if not terrain_available:
                        # No terrain: extrude flat road slabs
                        new_verts, new_faces = extrude_watertight(
                            merged, 2.0, base_y=10.0)
                        if new_verts and new_faces:
                            group = groups['road']
                            off = group['offset']
                            for f in new_faces:
                                group['faces'].append(
                                    [f[0] + off, f[1] + off, f[2] + off])
                            group['verts'].extend(new_verts)
                            group['offset'] += len(new_verts)
                    # When terrain IS available, roads are painted onto
                    # terrain vertex colors — no separate mesh needed.
                except Exception as e:
                    logger.warning(f"Road chunk (width={width}, "
                                   f"offset={ci}) failed: {e}")
        except Exception as e:
            logger.warning(f"Road group (width={width}) processing failed: {e}")

    total_segs = sum(len(v) for v in road_groups.values())
    logger.info(f"Processed {len(road_groups)} road width groups, "
                 f"{total_segs} segments → "
                 f"{len(road_polys_for_subtract)} merged chunks")

    # ── Railways: buffer lines into polygons, drape on terrain ──
    if railway_lines:
        rail_buffered = []
        for line in railway_lines:
            b = line.buffer(2.0, cap_style=2, resolution=4)  # 4m total width
            if not b.is_empty:
                rail_buffered.append(b)
        if rail_buffered:
            try:
                rail_merged = unary_union(rail_buffered)
                rail_merged = rail_merged.simplify(0.5, preserve_topology=True)
                rail_merged = rail_merged.intersection(area_poly)
                if water_union and not water_union.is_empty:
                    rail_merged = rail_merged.difference(water_union)
                if not rail_merged.is_empty:
                    if terrain_available:
                        rv, rf = _drape_on_terrain(
                            rail_merged, _elev_at, (0, 0), y_offset=0.45,
                            thickness=0.2, elev_batch_fn=_elev_batch)
                    else:
                        rv, rf = extrude_watertight(
                            rail_merged, 2.0, base_y=10.0)
                    groups['railway'] = {
                        'verts': [], 'faces': [], 'offset': 0}
                    if rv and rf:
                        groups['railway']['verts'] = rv
                        groups['railway']['faces'] = rf
                        groups['railway']['offset'] = len(rv)
                    road_polys_for_subtract.append(rail_merged)
                    logger.info(f"Processed {len(railway_lines)} railway "
                                f"segments into {len(rv or [])} verts")
            except Exception as e:
                logger.warning(f"Railway rendering failed: {e}")

    # ── Process bridge roads: drape on terrain using endpoint elevations ──
    # Each bridge linestring is draped individually.  The elevation at
    # any point on the bridge surface is linearly interpolated between
    # the terrain heights at the two endpoints.  This means:
    #   - Water bridges stay at bank height (endpoints on land)
    #   - Overpasses sit at ground level (endpoints ≈ same elevation)
    def _make_bridge_elev_fn(line_geom):
        """Return an elevation function that interpolates between the
        terrain heights at the start and end of *line_geom*."""
        coords = list(line_geom.coords)
        sx, sy = coords[0]
        ex, ey = coords[-1]
        s_elev = _elev_at(sx, sy)
        e_elev = _elev_at(ex, ey)
        dx, dy = ex - sx, ey - sy
        length_sq = dx * dx + dy * dy
        def _elev(x, y):
            if length_sq < 0.01:
                return (s_elev + e_elev) / 2.0
            t = ((x - sx) * dx + (y - sy) * dy) / length_sq
            t = max(0.0, min(1.0, t))
            return s_elev + (e_elev - s_elev) * t
        return _elev

    bridge_segs = sum(len(v) for v in bridge_groups.values())
    if bridge_segs > 0:
        _progress(75, "Building bridge surfaces...")
        bridge_count = 0
        for width, lines in bridge_groups.items():
            half_w = width / 2.0
            for line in lines:
                try:
                    buffered = line.buffer(half_w, cap_style=2, resolution=4)
                    if buffered.is_empty:
                        continue
                    road_polys_for_subtract.append(buffered)
                    if terrain_available:
                        elev_fn = _make_bridge_elev_fn(line)
                        drape_poly = buffered.buffer(1.0)
                        new_verts, new_faces = _drape_on_terrain(
                            drape_poly, elev_fn, (0, 0), y_offset=0.0,
                            max_tri_area=200.0, thickness=0.0)
                    else:
                        new_verts, new_faces = extrude_watertight(
                            buffered, 2.0, base_y=10.0)
                    if new_verts and new_faces:
                        group = groups['bridge']
                        off = group['offset']
                        for f in new_faces:
                            group['faces'].append(
                                [f[0] + off, f[1] + off, f[2] + off])
                        group['verts'].extend(new_verts)
                        group['offset'] += len(new_verts)
                        bridge_count += 1
                except Exception as e:
                    logger.warning(f"Bridge road (width={width}) failed: {e}")
        logger.info(f"Processed {bridge_count} bridge roads "
                     f"({bridge_segs} segments)")

    # Sanity check — need at least some geometry (terrain alone is fine)
    land_verts = []
    for g in groups.values():
        land_verts.extend(g['verts'])
    if not land_verts and not terrain_available:
        raise ValueError("No valid geometry to generate GLB file")

    # ── Area bounds fallback (if early computation didn't find stored bbox) ──
    if area_poly is None:
        all_arr = np.array(land_verts, dtype=np.float64)
        x_center = all_arr[:, 0].mean()
        z_center = all_arr[:, 2].mean()
        x_min, x_max = all_arr[:, 0].min(), all_arr[:, 0].max()
        z_min, z_max = all_arr[:, 2].min(), all_arr[:, 2].max()
        pad = 5.0
        area_poly = Polygon([
            (x_min - pad, z_min - pad),
            (x_min - pad, z_max + pad),
            (x_max + pad, z_max + pad),
            (x_max + pad, z_min - pad),
        ])
        logger.warning("No stored bbox — area_poly from feature bounds (may extend into water)")

    _timings['2_roads_bridges'] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    _progress(82, "Creating terrain...")
    # Layer 1: Blue base — flat rectangle at Y = 0 (no side walls)
    # Blue base matches the original (crop) bbox, not the expanded area
    blue_verts = [
        [crop_xmin, -1.0, crop_ymin],  # SW
        [crop_xmax, -1.0, crop_ymin],  # SE
        [crop_xmax, -1.0, crop_ymax],  # NE
        [crop_xmin, -1.0, crop_ymax],  # NW
    ]
    blue_faces = [[0, 2, 1], [0, 3, 2],   # top face, +Y normals
                  [0, 1, 2], [0, 2, 3]]  # bottom face, -Y normals
    groups['blue_base'] = {
        'verts': blue_verts,
        'faces': blue_faces,
        'offset': len(blue_verts),
    }

    # ── Build subtraction geometries for the green layer ──
    subtract_from_green = []
    if water_union is None and water_geoms:
        try:
            water_union = unary_union(water_geoms)
        except Exception as e:
            logger.warning(f"Water union failed: {e}")
    if water_union is not None and not water_union.is_empty:
        subtract_from_green.append(water_union)

    paved_union = None
    if paved_geoms:
        try:
            paved_union = unary_union(paved_geoms)
            subtract_from_green.append(paved_union)
        except Exception as e:
            logger.warning(f"Paved union failed: {e}")

    # Also subtract road surfaces from green
    if road_polys_for_subtract:
        try:
            road_union = unary_union(road_polys_for_subtract)
            subtract_from_green.append(road_union)
        except Exception as e:
            logger.warning(f"Road union for green subtraction failed: {e}")

    # Subtract sports fields, pools, piers, parks, glaciers from green
    for extra_geoms in [pitch_geoms, track_geoms, pool_geoms, pier_geoms,
                        park_geoms, glacier_geoms]:
        if extra_geoms:
            try:
                extra_union = unary_union(extra_geoms)
                subtract_from_green.append(extra_union)
            except Exception:
                pass

    # Layer 2: Green / Terrain surface
    green_poly = area_poly
    if subtract_from_green:
        try:
            cut = unary_union(subtract_from_green)
            green_poly = area_poly.difference(cut)
        except Exception as e:
            logger.warning(f"Green subtraction failed, using full area: {e}")

    # ── Treeline elevation from latitude ──
    # h ≈ 4150 × cos((|φ|/68) × π/2) — above this, rock instead of green
    treeline_elev = 4150.0 * math.cos((abs(clat) / 68.0) * (math.pi / 2))

    # ── Snowline elevation from latitude ──
    # Approximate permanent snowline (equilibrium line altitude).
    abs_lat = abs(clat)
    snowline_elev = 5000.0 - (60.0 * (abs_lat - 20.0)) if abs_lat > 20.0 else 5000.0
    # ── Latitude-based ground colour (biome stops) ──
    # 7-stop gradient matching real-world biome bands.
    _biome_stops = [
        (90, np.array([255, 255, 255])),  # ice cap
        (70, np.array([200, 195, 160])),  # tundra
        (55, np.array([ 80, 120,  60])),  # boreal forest
        (40, np.array([100, 160,  70])),  # temperate
        (25, np.array([210, 190, 130])),  # subtropical dry belt
        (10, np.array([ 50, 130,  50])),  # tropical wet
        ( 0, np.array([ 30, 110,  40])),  # equatorial forest
    ]
    # Piecewise-linear interpolation between stops
    for i in range(len(_biome_stops) - 1):
        lat_hi, rgb_hi = _biome_stops[i]
        lat_lo, rgb_lo = _biome_stops[i + 1]
        if abs_lat >= lat_lo:
            t = (abs_lat - lat_lo) / max(lat_hi - lat_lo, 1.0)
            t = min(t, 1.0)
            lat_ground = (rgb_lo * (1.0 - t) + rgb_hi * t) / 255.0
            break
    else:
        lat_ground = _biome_stops[-1][1] / 255.0
    type_colors['green'] = [float(lat_ground[0]), float(lat_ground[1]),
                            float(lat_ground[2]), 1.0]

    logger.info(f"Treeline: {treeline_elev:.0f}m, snowline: {snowline_elev:.0f}m "
                f"(lat={clat:.2f}°, elev_offset={elev_offset:.0f}m, "
                f"ground_rgb=[{lat_ground[0]:.2f},{lat_ground[1]:.2f},{lat_ground[2]:.2f}])")

    groups['green'] = {'verts': [], 'faces': [], 'offset': 0}
    terrain_all_verts = None
    terrain_all_faces = None
    if terrain_available:
        # CDT terrain mesh with building + water holes
        terrain_holes = building_footprints_utm[:]
        if water_union and not water_union.is_empty:
            if water_union.geom_type == 'Polygon':
                terrain_holes.append(water_union)
            elif water_union.geom_type == 'MultiPolygon':
                terrain_holes.extend(water_union.geoms)
        # Roads sit above terrain (y_offset=0.3) so no holes needed.
        # Paved areas also sit above terrain — skip holes for them too.
        # Only buildings and water need terrain holes.
        try:
            terrain_mesh = terrain_mod.build_terrain_mesh(
                grid_x, grid_y, elev_2d,
                hole_polys=terrain_holes,
                origin=(0, 0))
            if terrain_mesh and len(terrain_mesh.vertices) > 0:
                t_verts = terrain_mesh.vertices   # [N, 3]: easting, elev, northing
                t_faces = terrain_mesh.faces       # [M, 3]

                # Drop near-vertical "false wall" faces caused by
                # CDT triangulating across hole boundaries.
                v0 = t_verts[t_faces[:, 0]]
                v1 = t_verts[t_faces[:, 1]]
                v2 = t_verts[t_faces[:, 2]]
                normals = np.cross(v1 - v0, v2 - v0)
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                lengths[lengths == 0] = 1.0
                normals /= lengths
                upward = np.abs(normals[:, 1])  # Y component
                ground_mask = upward > 0.05     # only drop near-vertical CDT artifacts (>87°)
                n_before = len(t_faces)
                t_faces = t_faces[ground_mask]
                n_dropped = n_before - len(t_faces)

                terrain_all_verts = t_verts
                terrain_all_faces = t_faces
                logger.info(f"Terrain mesh: {len(t_verts)} verts, "
                            f"{len(t_faces)} faces "
                            f"({n_dropped} wall faces removed)")
                # Per-vertex coloring (latitude + elevation) applied later.

                # Build skirt walls from terrain boundary edges down to base
                try:
                    wall_v, wall_f = _build_skirt_walls(
                        t_verts, t_faces, base_y=-1.0)
                    if wall_v:
                        groups['wall']['verts'] = wall_v
                        groups['wall']['faces'] = wall_f
                        groups['wall']['offset'] = len(wall_v)
                        logger.info(f"Skirt walls: {len(wall_v)} verts, "
                                    f"{len(wall_f)} faces")
                except Exception as e:
                    logger.warning(f"Skirt wall generation failed: {e}")
            else:
                logger.warning("Terrain mesh empty — falling back to flat green")
                green_verts, green_faces = extrude_watertight(
                    green_poly, 10.0, base_y=0.0)
                if green_verts and green_faces:
                    groups['green']['verts'] = green_verts
                    groups['green']['faces'] = green_faces
                    groups['green']['offset'] = len(green_verts)
        except Exception as e:
            logger.warning(f"Terrain mesh failed: {e} — using flat green")
            green_verts, green_faces = extrude_watertight(
                green_poly, 10.0, base_y=0.0)
            if green_verts and green_faces:
                groups['green']['verts'] = green_verts
                groups['green']['faces'] = green_faces
                groups['green']['offset'] = len(green_verts)
    else:
        # No terrain — old flat green layer Y = 0 → 10
        green_verts, green_faces = extrude_watertight(
            green_poly, 10.0, base_y=0.0)
        if green_verts and green_faces:
            groups['green']['verts'] = green_verts
            groups['green']['faces'] = green_faces
            groups['green']['offset'] = len(green_verts)

    # Layer 2b: Paved — developed/commercial areas
    if paved_union and not paved_union.is_empty:
        if water_union and not water_union.is_empty:
            try:
                paved_render = paved_union.difference(water_union)
            except Exception:
                paved_render = paved_union
        else:
            paved_render = paved_union
        paved_render = paved_render.intersection(area_poly)
        logger.info(f"Paved drape: {paved_render.geom_type}, "
                    f"area={paved_render.area:.0f}m²")
        if terrain_available:
            paved_verts, paved_faces = _drape_on_terrain(
                paved_render, _elev_at, (0, 0), y_offset=0.3,
                thickness=0.3, elev_batch_fn=_elev_batch)
        else:
            paved_verts, paved_faces = extrude_watertight(
                paved_render, 10.0, base_y=0.0)
        groups['paved'] = {'verts': [], 'faces': [], 'offset': 0}
        if paved_verts and paved_faces:
            groups['paved']['verts'] = paved_verts
            groups['paved']['faces'] = paved_faces
            groups['paved']['offset'] = len(paved_verts)
            logger.info(f"Paved mesh: {len(paved_verts)} verts, "
                        f"{len(paved_faces)} faces")

    # ── Sports fields, pools, piers, parks, glaciers ────────────
    for geom_list, group_name, y_off, thick in [
        (pitch_geoms, 'pitch', 0.35, 0.2),
        (track_geoms, 'track', 0.35, 0.15),
        (pool_geoms, 'pool', 0.25, 1.5),
        (pier_geoms, 'pier', 0.4, 0.5),
        (park_geoms, 'park', 0.3, 0.15),
        (glacier_geoms, 'glacier', 0.15, 2.0),
    ]:
        if not geom_list:
            continue
        try:
            merged = unary_union(geom_list).intersection(area_poly)
            if merged.is_empty:
                continue
            logger.info(f"{group_name} drape: {merged.geom_type}, "
                        f"area={merged.area:.0f}m²")
            if terrain_available:
                verts, faces = _drape_on_terrain(
                    merged, _elev_at, (0, 0), y_offset=y_off,
                    thickness=thick, elev_batch_fn=_elev_batch)
            else:
                verts, faces = extrude_watertight(
                    merged, 10.0, base_y=0.0)
            groups[group_name] = {
                'verts': [], 'faces': [], 'offset': 0}
            if verts and faces:
                groups[group_name]['verts'] = verts
                groups[group_name]['faces'] = faces
                groups[group_name]['offset'] = len(verts)
                logger.info(f"{group_name}: {len(verts)} verts, "
                            f"{len(faces)} faces")
        except Exception as e:
            logger.warning(f"{group_name} rendering failed: {e}")

    _timings['3_terrain_drape'] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    # ── Decorative features: waves, sailboats, trees ──────────────
    _progress(85, "Adding decorative details...")

    # Compute visible water area (blue base showing through green/paved)
    visible_water = None
    if water_union and not water_union.is_empty:
        try:
            visible_water = water_union.intersection(area_poly)
            # Subtract green and paved from water for wave surface
            if not green_poly.is_empty:
                visible_water = visible_water.difference(green_poly)
            if paved_union and not paved_union.is_empty:
                visible_water = visible_water.difference(paved_union)
            if road_polys_for_subtract:
                road_u = unary_union(road_polys_for_subtract)
                visible_water = visible_water.difference(road_u)
            # Difference ops can produce GeometryCollections — extract polygonal parts
            if visible_water.geom_type == 'GeometryCollection':
                polys = [g for g in visible_water.geoms
                         if g.geom_type in ('Polygon', 'MultiPolygon')
                         and not g.is_empty]
                visible_water = unary_union(polys) if polys else None
            if visible_water is None or visible_water.is_empty:
                visible_water = None
        except Exception as e:
            logger.warning(f"Visible water computation failed: {e}")
            visible_water = None

    # Layer 5: Wave surface on visible water
    if visible_water is not None:
        try:
            wave_verts, wave_faces = _generate_wave_surface(
                visible_water, elev_batch_fn=_elev_batch)
            if wave_verts and wave_faces:
                groups['wave'] = {
                    'verts': wave_verts,
                    'faces': wave_faces,
                    'offset': len(wave_verts),
                }
                logger.info(f"Wave surface: {len(wave_verts)} verts, "
                             f"{len(wave_faces)} faces")
        except Exception as e:
            logger.warning(f"Wave surface generation failed: {e}")

    # Layer 6: Sailboats on water
    if visible_water is not None:
        try:
            # Inset water polygon to avoid boats at edges
            boat_area = visible_water.buffer(-30)
            if not boat_area.is_empty and boat_area.area > 100:
                boat_positions = _scatter_on_polygon(
                    boat_area, density_sqm=50000, min_count=3,
                    max_count=80, seed=123)
                if boat_positions:
                    rng_boat = random.Random(123)
                    groups['boat_hull'] = {'verts': [], 'faces': [],
                                           'offset': 0}
                    groups['boat_sail'] = {'verts': [], 'faces': [],
                                           'offset': 0}
                    # Batch-compute water elevation at boat positions
                    _boat_xs = np.array([p[0] for p in boat_positions])
                    _boat_ys = np.array([p[1] for p in boat_positions])
                    _boat_elevs = _elev_batch(_boat_xs, _boat_ys)
                    for i_boat, (bx, by) in enumerate(boat_positions):
                        rot = rng_boat.uniform(0, 2 * math.pi)
                        sc = rng_boat.uniform(1.5, 2.5)
                        boat = _create_sailboat_mesh(bx, by,
                                                     rotation=rot,
                                                     scale=sc)
                        boat_ground = _boat_elevs[i_boat]
                        for part_key, group_key in [('hull', 'boat_hull'),
                                                    ('sail', 'boat_sail')]:
                            v, f = boat[part_key]
                            # Offset boat vertices to water surface elevation
                            if v and boat_ground != 0:
                                v = [[vx, vy + boat_ground, vz]
                                     for vx, vy, vz in v]
                            if v and f:
                                g = groups[group_key]
                                off = g['offset']
                                for face in f:
                                    g['faces'].append(
                                        [face[0] + off, face[1] + off,
                                         face[2] + off])
                                g['verts'].extend(v)
                                g['offset'] += len(v)
                    logger.info(f"Placed {len(boat_positions)} sailboats")
        except Exception as e:
            logger.warning(f"Sailboat placement failed: {e}")

    # Layer 7: Vegetation from satellite imagery
    # Subtract only opaque features (buildings, water) — NOT roads/paved,
    # because tree canopies overhang roads and parking lots.  Subtracting
    # roads killed most street trees in urban areas.
    subtract_geoms = building_footprints_utm[:]
    if water_union and not water_union.is_empty:
        subtract_geoms.append(water_union)
    for extra in [pool_geoms, pier_geoms]:
        subtract_geoms.extend(extra)

    _timings['4_waves_boats'] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    tree_positions = []
    bush_positions = []
    veg_meta = None
    try:
        from . import vegetation as veg_mod

        _progress(85, "Detecting vegetation from WorldCover data...")
        veg_bbox = terrain_bbox if terrain_bbox is not None else city_state['bbox']
        tree_positions, bush_positions, veg_meta = \
            veg_mod.detect_vegetation_positions(
                veg_bbox, bbox_transformer, subtract_geoms)
        logger.info(f"WorldCover vegetation: {len(tree_positions)} trees, "
                     f"{len(bush_positions)} bushes")
        bush_positions = []  # TODO: re-enable when bush rendering is tuned
    except Exception as e:
        logger.warning(f"Vegetation detection failed: {e}")

    # Extract canopy height data for tree scaling
    canopy_heights = (veg_meta.get('canopy_heights', np.array([]))
                      if veg_meta else np.array([]))
    mean_canopy_height = (veg_meta.get('mean_canopy_height', 9.0)
                          if veg_meta else 9.0)

    # ── Latitude + elevation vertex colors for terrain ──
    if terrain_all_verts is not None:
        n_verts = len(terrain_all_verts)
        # Base: every vertex gets the latitude-derived ground colour
        base_rgb = (lat_ground * 255.0).astype(np.uint8)
        rgb_final = np.tile(base_rgb, (n_verts, 1))

        # ── WorldCover moisture modifier ──
        # In the 10-40° band, WorldCover land cover distinguishes
        # Sahara (bare → dry/sandy) from Amazon (tree → wet/green).
        lc_grid = veg_meta.get('lc_grid') if veg_meta else None
        lc_gx   = veg_meta.get('lc_grid_x') if veg_meta else None
        lc_gy   = veg_meta.get('lc_grid_y') if veg_meta else None
        if lc_grid is not None and lc_gx is not None and abs_lat < 45:
            # Map WorldCover class → moisture (0=dry, 1=wet)
            _wc_moisture = {
                10: 1.0,   # tree cover
                20: 0.4,   # shrubland
                30: 0.7,   # grassland
                40: 0.6,   # cropland
                50: 0.3,   # built-up
                60: 0.0,   # bare / sparse
                70: 0.0,   # snow / ice
                80: 0.5,   # water (neutral)
                90: 0.9,   # herbaceous wetland
                95: 1.0,   # mangroves
                100: 0.5,  # moss / lichen
            }
            wet_rgb  = np.array([60, 135, 55], dtype=np.float32)
            dry_rgb  = np.array([210, 190, 130], dtype=np.float32)
            # Nearest-neighbour lookup: terrain vertex UTM → grid index
            vx = terrain_all_verts[:, 0]  # easting
            vy = terrain_all_verts[:, 2]  # northing
            ix = np.clip(np.searchsorted(lc_gx, vx) - 1, 0, len(lc_gx) - 1)
            iy = np.clip(np.searchsorted(lc_gy, vy) - 1, 0, len(lc_gy) - 1)
            vert_classes = lc_grid[iy, ix]
            # Vectorised moisture lookup
            moisture = np.zeros(n_verts, dtype=np.float32)
            for cls_code, m_val in _wc_moisture.items():
                moisture[vert_classes == cls_code] = m_val
            # Strength peaks at |lat|=25° (subtropical high), fades to 0
            # by |lat|=45 (temperate) and |lat|=5 (deep tropics).
            shift_strength = np.float32(
                max(0.0, 1.0 - abs(abs_lat - 25.0) / 20.0) * 0.7)
            if shift_strength > 0.01:
                wc_target = (wet_rgb[None, :] * moisture[:, None]
                             + dry_rgb[None, :] * (1.0 - moisture[:, None]))
                rgb_f = rgb_final.astype(np.float32)
                rgb_final = np.clip(
                    rgb_f * (1.0 - shift_strength) + wc_target * shift_strength,
                    0, 255).astype(np.uint8)
                logger.info(f"WorldCover moisture: mean={moisture.mean():.2f}, "
                            f"shift_strength={shift_strength:.2f}")

        # ── Rock above treeline (400 m transition zone) ──
        rock_rgb = np.array([140, 133, 122], dtype=np.uint8)  # type_colors['rock'] in 0-255
        vert_y = terrain_all_verts[:, 1]
        treeline_y_val = treeline_elev - elev_offset
        fade_bottom_y = (treeline_elev - 400.0) - elev_offset
        above_mask = vert_y >= treeline_y_val
        transition_mask = (vert_y >= fade_bottom_y) & (vert_y < treeline_y_val)
        rgb_final[above_mask] = rock_rgb
        if transition_mask.any():
            t = ((vert_y[transition_mask] - fade_bottom_y)
                 / max(treeline_y_val - fade_bottom_y, 1.0))
            rgb_f = rgb_final[transition_mask].astype(np.float32)
            rgb_final[transition_mask] = np.clip(
                rgb_f * (1.0 - t[:, None]) + rock_rgb * t[:, None],
                0, 255).astype(np.uint8)

        # ── Elevation-based snow blending ──
        # Gradually blend vertices toward white from treeline → snowline.
        # Slope attenuation: snow doesn't stick to steep rock faces.
        max_real_elev = (elev_offset + float(elev_2d.max())) if terrain_available else 0.0
        if terrain_available and max_real_elev >= snowline_elev:
            vert_real_elev = terrain_all_verts[:, 1] + elev_offset
            snow_white = np.array([242, 244, 247], dtype=np.uint8)
            snow_t = np.clip(
                (vert_real_elev - treeline_elev) /
                max(snowline_elev - treeline_elev, 1.0),
                0.0, 1.0)

            # Slope attenuation — per-vertex normals from face normals
            vert_normals = np.zeros((n_verts, 3), dtype=np.float64)
            f0 = terrain_all_verts[terrain_all_faces[:, 0]]
            f1 = terrain_all_verts[terrain_all_faces[:, 1]]
            f2 = terrain_all_verts[terrain_all_faces[:, 2]]
            face_normals = np.cross(f1 - f0, f2 - f0)
            for i in range(3):
                np.add.at(vert_normals, terrain_all_faces[:, i], face_normals)
            v_lengths = np.linalg.norm(vert_normals, axis=1, keepdims=True)
            v_lengths[v_lengths == 0] = 1.0
            vert_normals /= v_lengths
            flatness = np.abs(vert_normals[:, 1])
            slope_factor = np.clip((flatness - 0.7) / 0.2, 0.0, 1.0)
            snow_t *= slope_factor

            snow_mask = snow_t > 0.0
            if snow_mask.any():
                rgb_f = rgb_final[snow_mask].astype(np.float32)
                t_vals = snow_t[snow_mask, None]
                rgb_final[snow_mask] = np.clip(
                    rgb_f * (1.0 - t_vals) + snow_white * t_vals,
                    0, 255).astype(np.uint8)
                n_snow = snow_mask.sum()
                n_full = (snow_t >= 1.0).sum()
                logger.info(f"Snow blending: {n_snow} vertices affected, "
                            f"{n_full} fully snow-covered "
                            f"(treeline={treeline_elev:.0f}m, "
                            f"snowline={snowline_elev:.0f}m)")

        # ── Forest floor under tree/bush canopy ──
        if tree_positions or bush_positions:
            from scipy.spatial import cKDTree
            terrain_xy = np.column_stack([
                terrain_all_verts[:, 0], terrain_all_verts[:, 2]])
            forest_floor = np.array([85, 150, 70], dtype=np.uint8)
            if tree_positions:
                tree_kd = cKDTree(np.array(tree_positions))
                dists, _ = tree_kd.query(terrain_xy)
                rgb_final[dists < 6.0] = forest_floor
            if bush_positions:
                bush_kd = cKDTree(np.array(bush_positions))
                dists, _ = bush_kd.query(terrain_xy)
                rgb_final[dists < 2.5] = forest_floor

        # ── Paint roads onto terrain ──
        # Face-based: check each triangle centroid against road polygons,
        # then color ALL vertices of road-touching faces.  This gives
        # continuous road coverage instead of isolated vertex dots.
        if road_polys_for_subtract:
            import shapely as _shp
            road_union = unary_union(road_polys_for_subtract)
            # Compute face centroids in UTM (easting, northing)
            fc_x = terrain_all_verts[terrain_all_faces, 0].mean(axis=1)
            fc_y = terrain_all_verts[terrain_all_faces, 2].mean(axis=1)
            face_in_road = _shp.contains_xy(road_union, fc_x, fc_y)
            # Collect all vertex indices belonging to road faces
            road_vert_idx = np.unique(terrain_all_faces[face_in_road].ravel())
            road_rgb = np.array([64, 64, 64], dtype=np.uint8)
            rgb_final[road_vert_idx] = road_rgb
            logger.info(f"Road painting: {face_in_road.sum()} faces, "
                        f"{len(road_vert_idx)} verts painted as road")

        rgba = np.column_stack([rgb_final,
                                np.full(n_verts, 255, dtype=np.uint8)])
        groups.pop('green', None)
        groups.pop('rock', None)
        groups['terrain'] = {
            'verts': terrain_all_verts.tolist(),
            'faces': terrain_all_faces.tolist(),
            'offset': len(terrain_all_verts),
            'vertex_colors': rgba,
        }
        logger.info(f"Terrain vertex coloring: {n_verts} verts, "
                    f"{len(terrain_all_faces)} faces")

    # Filter tree/bush positions that fall too close to roads (< 5m buffer)
    if road_polys_for_subtract and (tree_positions or bush_positions):
        try:
            road_buf = unary_union(road_polys_for_subtract).buffer(5.0)
            road_buf_prep = prep(road_buf)
            pre_t = len(tree_positions)
            pre_b = len(bush_positions)
            tree_positions = [
                (x, y) for x, y in tree_positions
                if not road_buf_prep.contains(Point(x, y))
            ]
            bush_positions = [
                (x, y) for x, y in bush_positions
                if not road_buf_prep.contains(Point(x, y))
            ]
            logger.info(f"Road buffer filter: removed {pre_t - len(tree_positions)} trees, "
                        f"{pre_b - len(bush_positions)} bushes within 5m of roads")
        except Exception as e:
            logger.warning(f"Road buffer vegetation filter failed: {e}")

    # Filter tree/bush positions that fall on paved/developed areas
    if paved_union and not paved_union.is_empty and (tree_positions or bush_positions):
        try:
            paved_prep = prep(paved_union)
            pre_t = len(tree_positions)
            pre_b = len(bush_positions)
            tree_positions = [
                (x, y) for x, y in tree_positions
                if not paved_prep.contains(Point(x, y))
            ]
            bush_positions = [
                (x, y) for x, y in bush_positions
                if not paved_prep.contains(Point(x, y))
            ]
            logger.info(f"Paved filter: removed {pre_t - len(tree_positions)} trees, "
                        f"{pre_b - len(bush_positions)} bushes from paved areas")
        except Exception as e:
            logger.warning(f"Paved vegetation filter failed: {e}")

    # Filter tree/bush positions that fall inside building footprints
    # and sports/recreation surfaces (pitches, tracks, pools).
    # Fill interior holes so open structures (stadiums, courtyards)
    # also block vegetation — WorldCover sees grass inside but random
    # tree meshes there look wrong.
    structure_polys = []
    for bp in (building_footprints_utm or []):
        if hasattr(bp, 'exterior'):
            structure_polys.append(Polygon(bp.exterior))
        elif hasattr(bp, 'geoms'):
            for g in bp.geoms:
                if hasattr(g, 'exterior'):
                    structure_polys.append(Polygon(g.exterior))
        else:
            structure_polys.append(bp)
    for geom_list in [pitch_geoms, track_geoms, pool_geoms]:
        for g in geom_list:
            if g is not None and not g.is_empty:
                structure_polys.append(g)
    if structure_polys and (tree_positions or bush_positions):
        try:
            struct_union = unary_union(structure_polys)
            struct_prep = prep(struct_union)
            pre_t = len(tree_positions)
            pre_b = len(bush_positions)
            tree_positions = [
                (x, y) for x, y in tree_positions
                if not struct_prep.contains(Point(x, y))
            ]
            bush_positions = [
                (x, y) for x, y in bush_positions
                if not struct_prep.contains(Point(x, y))
            ]
            logger.info(f"Structure filter: removed {pre_t - len(tree_positions)} trees, "
                        f"{pre_b - len(bush_positions)} bushes from buildings/pitches/tracks/pools")
        except Exception as e:
            logger.warning(f"Structure vegetation filter failed: {e}")

    # Filter tree/bush positions that fall inside glacier areas
    no_tree_polys = []
    if glacier_geoms:
        no_tree_polys.extend(glacier_geoms)
    if no_tree_polys and (tree_positions or bush_positions):
        try:
            no_tree_union = unary_union(no_tree_polys)
            no_tree_prep = prep(no_tree_union)
            pre_trees = len(tree_positions)
            pre_bushes = len(bush_positions)
            tree_positions = [
                (x, y) for x, y in tree_positions
                if not no_tree_prep.contains(Point(x, y))
            ]
            bush_positions = [
                (x, y) for x, y in bush_positions
                if not no_tree_prep.contains(Point(x, y))
            ]
            removed_trees = pre_trees - len(tree_positions)
            removed_bushes = pre_bushes - len(bush_positions)
            if removed_trees or removed_bushes:
                logger.info(f"Removed {removed_trees} trees and "
                            f"{removed_bushes} bushes from glacier areas")
        except Exception as e:
            logger.warning(f"Glacier vegetation filter failed: {e}")

    # Filter trees near/above treeline with gradual fade-out.
    # Trees thin out over a 400m transition zone below the treeline,
    # avoiding a sharp "haircut" boundary.
    transition_zone = 400.0  # metres of gradual thinning
    fade_start = treeline_elev - transition_zone  # 100% density below
    if terrain_available and tree_positions:
        rng_treeline = random.Random(321)
        pre_trees = len(tree_positions)
        filtered = []
        for x, y in tree_positions:
            real_elev = _elev_at(x, y) + elev_offset
            if real_elev < fade_start:
                filtered.append((x, y))       # fully below → keep
            elif real_elev >= treeline_elev:
                pass                           # fully above → remove
            else:
                # Transition zone: survival probability drops linearly
                # from 1.0 at fade_start to 0.0 at treeline_elev
                keep_prob = 1.0 - (real_elev - fade_start) / transition_zone
                if rng_treeline.random() < keep_prob:
                    filtered.append((x, y))
        tree_positions = filtered
        removed = pre_trees - len(tree_positions)
        if removed:
            logger.info(f"Treeline fade: removed {removed} trees "
                        f"({fade_start:.0f}–{treeline_elev:.0f}m)")

    if tree_positions:
        rng_tree = random.Random(456)

        # Build position → canopy height lookup for height-driven scaling
        # Reference tree at scale 1.0 = 9m (trunk_h=4 + canopy_ry=5)
        _REF_TREE_H = 9.0
        tree_height_map = {}
        if (canopy_heights is not None
                and len(canopy_heights) == len(tree_positions)):
            for i, (tx, ty) in enumerate(tree_positions):
                tree_height_map[(tx, ty)] = float(canopy_heights[i])

        # All trees rendered as individual meshes.
        individual_positions = list(tree_positions)
        logger.info(f"Placing {len(individual_positions)} individual trees")

        if individual_positions:
            from . import koppen as koppen_mod
            from .koppen import TreeType

            # Initialize mesh groups for all tree types
            if 'tree_canopy' not in groups:
                groups['tree_canopy'] = {'verts': [], 'faces': [], 'offset': 0}
            groups['tree_trunk'] = {'verts': [], 'faces': [], 'offset': 0}
            for gname in ('conifer_canopy', 'palm_canopy',
                          'broadleaf_canopy', 'mangrove_canopy',
                          'scrub_canopy', 'sclerophyll_canopy'):
                groups[gname] = {'verts': [], 'faces': [], 'offset': 0}

            # Köppen climate-based tree type mix (single pixel lookup)
            tree_mix = koppen_mod.get_tree_mix(clat, clon)
            koppen_code = koppen_mod.get_koppen_code(clat, clon)
            logger.info(
                f"Köppen tree mix ({koppen_code}): "
                f"{', '.join(f'{t.value}={w:.0%}' for t, w in tree_mix.items())}")

            # Map TreeType → (mesh generator, canopy group name)
            _TREE_GENERATORS = {
                TreeType.CONIFER: (_create_conifer_tree_mesh, 'conifer_canopy'),
                TreeType.DECIDUOUS: (_create_deciduous_tree_mesh, 'tree_canopy'),
                TreeType.BROADLEAF_TROPICAL: (
                    _create_broadleaf_tropical_tree_mesh, 'broadleaf_canopy'),
                TreeType.PALM: (_create_palm_tree_mesh, 'palm_canopy'),
                TreeType.MANGROVE: (_create_mangrove_tree_mesh, 'mangrove_canopy'),
                TreeType.SCRUB: (_create_scrub_mesh, 'scrub_canopy'),
                TreeType.SCLEROPHYLL: (
                    _create_sclerophyll_tree_mesh, 'sclerophyll_canopy'),
            }

            # Pre-sort mix keys for stable cumulative weight lookup
            mix_items = list(tree_mix.items())
            tree_counts = {t: 0 for t in tree_mix}

            # Precompute all tree elevations in batch (vectorized)
            _tree_xs = np.array([p[0] for p in individual_positions])
            _tree_ys = np.array([p[1] for p in individual_positions])
            _tree_elevs = _elev_batch(_tree_xs, _tree_ys)

            for i_tree, (tx, ty) in enumerate(individual_positions):
                # Height-driven scale from canopy height data
                real_h = tree_height_map.get((tx, ty), mean_canopy_height)
                base_sc = real_h / _REF_TREE_H
                sc = base_sc * rng_tree.uniform(0.9, 1.1)  # ±10% variation
                sc = max(0.3, min(sc, 4.0))  # clamp (2.7m – 36m)
                # Stunt trees near treeline — scale shrinks in transition zone
                tree_ground_y = _tree_elevs[i_tree]
                if terrain_available:
                    real_elev = tree_ground_y + elev_offset
                    if real_elev > fade_start:
                        altitude_factor = 1.0 - 0.6 * (
                            (real_elev - fade_start) / transition_zone)
                        sc *= max(altitude_factor, 0.4)

                # Pick tree type by cumulative weight lookup
                r = rng_tree.random()
                cumulative = 0.0
                chosen_type = mix_items[-1][0]  # fallback to last
                for tree_type, weight in mix_items:
                    cumulative += weight
                    if r < cumulative:
                        chosen_type = tree_type
                        break

                gen_fn, canopy_group = _TREE_GENERATORS[chosen_type]
                tree = gen_fn(tx, ty, scale=sc, ground_y=tree_ground_y)
                tree_counts[chosen_type] = tree_counts.get(chosen_type, 0) + 1

                for part_key, group_key in [
                    ('trunk', 'tree_trunk'),
                    ('canopy', canopy_group),
                ]:
                    v, f = tree[part_key]
                    if v and f:
                        g = groups[group_key]
                        off = g['offset']
                        for face in f:
                            g['faces'].append(
                                [face[0] + off, face[1] + off,
                                 face[2] + off])
                        g['verts'].extend(v)
                        g['offset'] += len(v)
            counts_str = ', '.join(
                f"{t.value}={c}" for t, c in tree_counts.items() if c > 0)
            logger.info(f"Placed {len(individual_positions)} individual trees "
                        f"({counts_str})")

    if bush_positions:
        rng_bush = random.Random(789)
        groups['bush_canopy'] = {'verts': [], 'faces': [], 'offset': 0}
        # Precompute bush elevations in batch
        _bush_xs = np.array([p[0] for p in bush_positions])
        _bush_ys = np.array([p[1] for p in bush_positions])
        _bush_elevs = _elev_batch(_bush_xs, _bush_ys)
        for i_bush, (bx, by) in enumerate(bush_positions):
            sc = rng_bush.uniform(0.6, 1.2)
            bush_ground_y = _bush_elevs[i_bush]
            bush = _create_bush_mesh(
                bx, by, scale=sc, ground_y=bush_ground_y)
            v, f = bush['canopy']
            if v and f:
                g = groups['bush_canopy']
                off = g['offset']
                for face in f:
                    g['faces'].append(
                        [face[0] + off, face[1] + off,
                         face[2] + off])
                g['verts'].extend(v)
                g['offset'] += len(v)
        logger.info(f"Placed {len(bush_positions)} bushes")

    _timings['5_vegetation_trees'] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    total_verts = sum(len(g['verts']) for g in groups.values())
    if total_verts == 0:
        raise ValueError("No valid geometry to generate GLB file")

    # Build a trimesh Scene – one mesh per layer, each with a
    # solid PBR material so colours export reliably to GLB.
    _progress(90, "Assembling 3D model...")
    glb_scene = trimesh.Scene()

    # Crop to original bbox: remove faces outside the user's requested area.
    # The model was built with a ~200m buffer so edge features (roads, water)
    # render completely; now we trim back to the original bbox for clean edges.
    INSET_M = 1.0
    _clip_xmin = crop_xmin - x_center + INSET_M
    _clip_xmax = crop_xmax - x_center - INSET_M
    _clip_zmin = crop_ymin - z_center + INSET_M
    _clip_zmax = crop_ymax - z_center - INSET_M

    for name, group in groups.items():
        if not group['verts'] or not group['faces']:
            continue

        verts_arr = np.array(group['verts'], dtype=np.float64)
        faces_arr = np.array(group['faces'], dtype=np.int32)

        # Centre X and Z (horizontal), then negate Z so +Z = south.
        # Three.js is right-handed: camera at +Z looking -Z sees +X
        # on the right side of the screen → east = right.
        verts_arr[:, 0] -= x_center
        verts_arr[:, 2] -= z_center
        verts_arr[:, 2] *= -1
        # Negating one axis flips handedness — reverse face winding
        faces_arr = faces_arr[:, ::-1]

        # Crop to original bbox
        if name == 'blue_base':
            pass  # Already crop-sized
        elif 'vertex_colors' in group:
            # Terrain: face-removal crop then clamp vertices to crop box
            # for clean straight edges (preserves vertex colors).
            vx = verts_arr[:, 0]
            vz = verts_arr[:, 2]
            outside = ((vx < _clip_xmin) | (vx > _clip_xmax) |
                       (vz < _clip_zmin) | (vz > _clip_zmax))
            edge_count = outside[faces_arr].sum(axis=1)
            faces_arr = faces_arr[edge_count < 2]
            if len(faces_arr) == 0:
                continue
            # Clamp remaining outside vertices to the crop rectangle
            # so edges are straight, not jagged grid steps.
            verts_arr[:, 0] = np.clip(verts_arr[:, 0], _clip_xmin, _clip_xmax)
            verts_arr[:, 2] = np.clip(verts_arr[:, 2], _clip_zmin, _clip_zmax)
        elif len(faces_arr) > 0:
            # All other meshes: proper plane slicing for clean edges
            tmp = trimesh.Trimesh(vertices=verts_arr, faces=faces_arr)
            for pl_origin, pl_normal in [
                ([_clip_xmin, 0, 0], [1, 0, 0]),
                ([_clip_xmax, 0, 0], [-1, 0, 0]),
                ([0, 0, _clip_zmin], [0, 0, 1]),
                ([0, 0, _clip_zmax], [0, 0, -1]),
            ]:
                tmp = tmp.slice_plane(pl_origin, pl_normal)
                if tmp is None or len(tmp.faces) == 0:
                    break
            if tmp is None or len(tmp.faces) == 0:
                continue
            verts_arr = np.array(tmp.vertices, dtype=np.float64)
            faces_arr = np.array(tmp.faces, dtype=np.int32)

        if 'vertex_colors' in group:
            # Satellite-colored terrain — pass vertex colors in constructor
            # so trimesh creates a proper ColorVisuals (not TextureVisuals).
            mesh = trimesh.Trimesh(
                vertices=verts_arr, faces=faces_arr,
                vertex_colors=group['vertex_colors'])
        else:
            mesh = trimesh.Trimesh(vertices=verts_arr, faces=faces_arr)

            # Fix face normals for decorative meshes (trees, boats).
            if name in ('tree_trunk', 'tree_canopy', 'conifer_canopy',
                        'palm_canopy', 'broadleaf_canopy',
                        'mangrove_canopy', 'scrub_canopy',
                        'sclerophyll_canopy', 'bush_canopy',
                        'boat_hull', 'boat_sail',
                        'statue', 'torch'):
                mesh.fix_normals()

            color = type_colors.get(name, [0.8, 0.8, 0.8, 1.0])
            if name == 'shells':
                material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.96, 0.95, 0.92, 1.0],
                    roughnessFactor=0.30,
                    metallicFactor=0.0,
                    doubleSided=True,
                )
            elif name == 'windows':
                material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=color,
                    roughnessFactor=0.3,
                    metallicFactor=0.5,
                    doubleSided=True,
                )
            else:
                material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=color,
                    doubleSided=True,
                )
            mesh.visual = trimesh.visual.TextureVisuals(material=material)

        glb_scene.add_geometry(mesh, geom_name=name)

    _progress(95, "Finalizing design...")
    glb_scene.export(str(output_path), file_type='glb')
    _timings['6_assembly_export'] = time.perf_counter() - _t0

    # ── Timing summary ──
    logger.info("=" * 60)
    logger.info("GLB GENERATION TIMING BREAKDOWN")
    logger.info("=" * 60)
    _total = 0.0
    for _lbl, _dur in sorted(_timings.items()):
        logger.info(f"  {_lbl}: {_dur:.1f}s")
        _total += _dur
    logger.info(f"  TOTAL: {_total:.1f}s")
    logger.info("=" * 60)

    logger.info(f"GLB file generated successfully: {output_path}")
    return str(output_path)
