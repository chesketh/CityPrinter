"""Coordinate transforms, watertight extrusion, and roof mesh generation."""

import math
import logging

import numpy as np
import trimesh
from shapely.geometry import (
    Polygon, MultiPolygon, LineString,
)
from shapely.geometry.polygon import orient

logger = logging.getLogger(__name__)

# ── Supported roof shapes ────────────────────────────────────────────────
ROOF_SHAPES_SUPPORTED = frozenset({
    'round', 'dome', 'onion', 'pyramidal',
    'gabled', 'hipped',
    'skillion', 'lean_to',
})


# ── Coordinate transforms ───────────────────────────────────────────────

def transform_geometry(geom, transformer):
    """Transform geometry from WGS84 to local projection."""
    if geom is None or not geom.is_valid:
        return None

    try:
        if isinstance(geom, (Polygon, MultiPolygon)):
            transformed = transform_polygon(geom, transformer)
        elif isinstance(geom, LineString):
            transformed = transform_linestring(geom, transformer)
        else:
            return None

        # Validate transformed geometry
        if transformed is None or not transformed.is_valid:
            return None

        # Check for NaN coordinates
        if isinstance(transformed, MultiPolygon):
            coords = []
            for p in transformed.geoms:
                coords.extend(list(p.exterior.coords))
        elif isinstance(transformed, Polygon):
            coords = list(transformed.exterior.coords)
        else:
            coords = list(transformed.coords)

        if any(np.isnan(coord).any() for coord in coords):
            return None

        return transformed

    except Exception as e:
        logger.error(f"Error transforming geometry: {e}")
        return None


def transform_polygon(polygon, transformer):
    """Transform polygon coordinates."""
    try:
        if isinstance(polygon, MultiPolygon):
            transformed_polys = []
            for p in polygon.geoms:
                transformed = transform_polygon(p, transformer)
                if transformed and transformed.is_valid:
                    transformed_polys.append(transformed)
            if not transformed_polys:
                return None
            return MultiPolygon(transformed_polys)

        # Now we know it's a simple Polygon
        # Transform exterior ring
        exterior_coords = []
        for x, y in polygon.exterior.coords:
            try:
                tx, ty = transformer.transform(x, y)
                if np.isnan(tx) or np.isnan(ty):
                    continue
                exterior_coords.append((tx, ty))
            except Exception:
                return None

        # Transform interior rings
        interior_coords = []
        for interior in polygon.interiors:
            interior_ring = []
            for x, y in interior.coords:
                try:
                    tx, ty = transformer.transform(x, y)
                    if np.isnan(tx) or np.isnan(ty):
                        continue
                    interior_ring.append((tx, ty))
                except Exception:
                    continue
            if len(interior_ring) >= 3:  # Need at least 3 points for a valid ring
                interior_coords.append(interior_ring)

        # Create and validate the transformed polygon
        transformed = Polygon(exterior_coords, interior_coords)
        if transformed.is_valid and transformed.area > 0:
            return transformed
        return None

    except Exception as e:
        logger.error(f"Error transforming polygon: {e}")
        return None


def transform_linestring(line, transformer):
    """Transform linestring coordinates."""
    try:
        coords = []
        for x, y in line.coords:
            try:
                tx, ty = transformer.transform(x, y)
                if np.isnan(tx) or np.isnan(ty):
                    continue
                coords.append((tx, ty))
            except Exception:
                continue

        if len(coords) >= 2:  # Need at least 2 points for a valid linestring
            transformed = LineString(coords)
            if transformed.is_valid and transformed.length > 0:
                return transformed
        return None

    except Exception as e:
        logger.error(f"Error transforming linestring: {e}")
        return None


# ── Watertight extrusion ─────────────────────────────────────────────────

def extrude_watertight(geometry, height: float,
                       base_y: float = 0.0,
                       buffer_width: float | None = None):
    """Extrude a 2D geometry into a watertight 3D mesh for GLB export.

    Uses ``trimesh.creation.extrude_polygon`` for proper constrained
    triangulation (handles concave polygons and holes), then swaps Y↔Z
    for glTF Y-up convention and reverses face winding to compensate
    for the handedness flip.

    Returns (vertices, faces) as plain Python lists.
    """
    # Buffer line geometries into polygons (road surface)
    if geometry.geom_type in ('LineString', 'MultiLineString'):
        half_width = (buffer_width or 6.0) / 2.0
        geometry = geometry.buffer(half_width, cap_style=2)
        if geometry.is_empty:
            return [], []

    if geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    elif geometry.geom_type == 'Polygon':
        polygons = [geometry]
    elif geometry.geom_type == 'GeometryCollection':
        polygons = [
            g for g in geometry.geoms
            if g.geom_type in ('Polygon', 'MultiPolygon')
        ]
    else:
        return [], []

    all_verts: list[list[float]] = []
    all_faces: list[list[int]] = []

    for poly in polygons:
        # Flatten MultiPolygon inside GeometryCollection
        sub_polys = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
        for sp in sub_polys:
            if sp.is_empty or sp.area < 0.01:
                continue
            try:
                mesh = trimesh.creation.extrude_polygon(sp, height=height)
            except Exception as e:
                logger.warning(f"extrude_polygon failed: {e}")
                continue

            v = mesh.vertices.copy()
            # Z-up → Y-up: swap Y↔Z, then shift Y by base_y
            new_verts = np.column_stack([v[:, 0], v[:, 2] + base_y, v[:, 1]])
            # Reverse face winding to fix normals after axis swap
            new_faces = [[f[0], f[2], f[1]] for f in mesh.faces.tolist()]

            off = len(all_verts)
            all_verts.extend(new_verts.tolist())
            for f in new_faces:
                all_faces.append([f[0] + off, f[1] + off, f[2] + off])

    return all_verts, all_faces


# ── Window quad generation ──────────────────────────────────────────────

def generate_window_quads(geometry, wall_height: float, base_y: float,
                          win_w: float = 1.5, win_h: float = 1.8,
                          h_gap: float = 3.5, v_gap: float = 4.0,
                          margin_h: float = 1.5, margin_bot: float = 3.0,
                          margin_top: float = 1.0, offset: float = 0.08):
    """Generate dark window quads on exterior walls of a building.

    Places flat rectangles slightly in front of each wall surface.
    Returns (verts, faces) in glTF Y-up ``[easting, elevation, northing]``.
    """
    if geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    elif geometry.geom_type == 'Polygon':
        polygons = [geometry]
    else:
        return [], []

    usable_h = wall_height - margin_bot - margin_top
    if usable_h < win_h:
        return [], []

    all_verts: list[list[float]] = []
    all_faces: list[list[int]] = []

    for poly in polygons:
        poly = orient(poly, sign=1.0)  # CCW exterior
        coords = list(poly.exterior.coords[:-1])
        n = len(coords)
        if n < 3:
            continue

        for i in range(n):
            x0, z0 = coords[i]
            x1, z1 = coords[(i + 1) % n]
            dx, dz = x1 - x0, z1 - z0
            seg_len = math.sqrt(dx * dx + dz * dz)
            if seg_len < 4.0:
                continue

            # Wall direction and outward normal (right-hand for CCW)
            dirx, dirz = dx / seg_len, dz / seg_len
            nx, nz = dirz, -dirx  # outward normal

            # Grid dimensions
            usable_w = seg_len - 2.0 * margin_h
            if usable_w < win_w:
                continue
            n_cols = max(1, int(usable_w / h_gap) + 1)
            n_rows = min(max(1, int(usable_h / v_gap) + 1), 15)

            # Actual spacing
            if n_cols > 1:
                h_space = usable_w / (n_cols - 1)
            else:
                h_space = 0.0
            if n_rows > 1:
                v_space = usable_h / (n_rows - 1)
            else:
                v_space = 0.0

            half_w = win_w / 2.0
            half_h = win_h / 2.0

            for col in range(n_cols):
                t = margin_h + col * h_space if n_cols > 1 else seg_len / 2.0
                cx = x0 + dirx * t + nx * offset
                cz = z0 + dirz * t + nz * offset

                for row in range(n_rows):
                    cy = base_y + margin_bot + row * v_space if n_rows > 1 else base_y + margin_bot + usable_h / 2.0

                    # 4 corners of the window quad
                    vi = len(all_verts)
                    all_verts.append([cx - dirx * half_w, cy - half_h, cz - dirz * half_w])
                    all_verts.append([cx + dirx * half_w, cy - half_h, cz + dirz * half_w])
                    all_verts.append([cx + dirx * half_w, cy + half_h, cz + dirz * half_w])
                    all_verts.append([cx - dirx * half_w, cy + half_h, cz - dirz * half_w])
                    # Two tris — normal faces outward (same dir as nx, nz)
                    all_faces.append([vi, vi + 1, vi + 2])
                    all_faces.append([vi, vi + 2, vi + 3])

    return all_verts, all_faces


# ── Roof mesh generation ────────────────────────────────────────────────

def generate_roof_mesh(geometry, roof_shape: str,
                       roof_height: float, eave_y: float,
                       roof_direction: float = None):
    """Generate a 3D roof mesh for non-flat roof shapes.

    Returns ``(vertices, faces)`` in glTF Y-up coordinates, ready to
    merge directly into the building vertex/face group.

    Parameters
    ----------
    geometry : shapely Polygon / MultiPolygon
        Building footprint in projected (UTM) coordinates.
    roof_shape : str
        One of: pyramidal, gabled, hipped, skillion, round/dome.
    roof_height : float
        Height of the roof portion (eave to peak) in metres.
    eave_y : float
        Y coordinate (glTF up) of the eave line — the top of the
        walls / bottom of the roof.
    roof_direction : float, optional
        Compass bearing (degrees CW from north) of the downslope
        direction.  Used by skillion roofs.
    """
    if geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    elif geometry.geom_type == 'Polygon':
        polygons = [geometry]
    else:
        return [], []

    all_verts: list[list[float]] = []
    all_faces: list[list[int]] = []

    for poly in polygons:
        sub_polys = (list(poly.geoms)
                     if poly.geom_type == 'MultiPolygon' else [poly])
        for sp in sub_polys:
            if sp.is_empty or sp.area < 0.01:
                continue
            coords = list(sp.exterior.coords[:-1])
            if len(coords) < 3:
                continue

            shape = (roof_shape.lower()
                     .replace('-', '_').replace(' ', '_'))

            if shape == 'pyramidal':
                v, f = _roof_pyramidal(coords, roof_height, eave_y)
            elif shape == 'gabled':
                v, f = _roof_gabled(coords, sp,
                                     roof_height, eave_y)
            elif shape == 'hipped':
                v, f = _roof_hipped(coords, sp,
                                     roof_height, eave_y)
            elif shape in ('round', 'dome', 'onion'):
                v, f = _roof_dome(coords, sp,
                                   roof_height, eave_y)
            elif shape in ('skillion', 'lean_to'):
                v, f = _roof_skillion(coords, sp,
                                       roof_height, eave_y,
                                       roof_direction=roof_direction)
            else:
                continue

            if v and f:
                # Fix normals: roof faces must point upward (+Y).
                # Vertical gable-end faces use centroid direction.
                va = np.array(v)
                centroid_3d = va.mean(axis=0)
                for face in f:
                    v0 = va[face[0]]
                    v1 = va[face[1]]
                    v2 = va[face[2]]
                    normal = np.cross(v1 - v0, v2 - v0)
                    nl = np.linalg.norm(normal)
                    if nl < 1e-10:
                        continue
                    ny = normal[1] / nl
                    if ny < -0.01:
                        # Sloped face pointing down — flip
                        face[0], face[1], face[2] = \
                            face[0], face[2], face[1]
                    elif abs(ny) <= 0.01:
                        # Vertical face (gable end) — outward test
                        fc = (v0 + v1 + v2) / 3
                        if np.dot(normal, fc - centroid_3d) < 0:
                            face[0], face[1], face[2] = \
                                face[0], face[2], face[1]

                off = len(all_verts)
                all_verts.extend(v)
                for face in f:
                    all_faces.append([face[0] + off,
                                      face[1] + off,
                                      face[2] + off])

    return all_verts, all_faces


# ── Individual roof shape helpers ────────────────────────────────────────

def _roof_pyramidal(coords, roof_height, eave_y):
    """Pyramid / cone: all edges slope to a single central apex."""
    n = len(coords)
    verts = [[c[0], eave_y, c[1]] for c in coords]
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    verts.append([cx, eave_y + roof_height, cy])
    apex = n
    faces = [[i, (i + 1) % n, apex] for i in range(n)]
    return verts, faces


def _roof_dome(coords, polygon, roof_height, eave_y):
    """Dome / onion: hemisphere fitted to the footprint bounding box.

    Uses a UV-sphere approach matching OSMBuilding's SphereGeometry.
    """
    bounds = polygon.bounds  # (minx, miny, maxx, maxy) in UTM
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    rx = (bounds[2] - bounds[0]) / 2  # easting half-width
    rz = (bounds[3] - bounds[1]) / 2  # northing half-depth

    n_lat = 8   # latitude rings (equator → just below pole)
    n_lon = 16  # longitude segments

    verts = []
    faces = []

    # Rings from equator (phi=0) to just below pole
    for i in range(n_lat):
        phi = (math.pi / 2) * i / n_lat
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for j in range(n_lon):
            theta = 2 * math.pi * j / n_lon
            x = cx + rx * cos_phi * math.cos(theta)
            y = eave_y + roof_height * sin_phi
            z = cy + rz * cos_phi * math.sin(theta)
            verts.append([x, y, z])

    # Pole vertex (single apex point)
    pole_idx = len(verts)
    verts.append([cx, eave_y + roof_height, cy])

    # Quad faces between adjacent rings
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            v0 = i * n_lon + j
            v1 = i * n_lon + j_next
            v2 = (i + 1) * n_lon + j
            v3 = (i + 1) * n_lon + j_next
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Triangle fan connecting top ring to pole
    last_ring = (n_lat - 1) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([last_ring + j, pole_idx, last_ring + j_next])

    return verts, faces


def _roof_ridge_info(polygon):
    """Return (ridge_dir, ridge_normal, half_width, ridge_half_len)
    from the minimum rotated rectangle of *polygon*."""
    mrr = polygon.minimum_rotated_rectangle
    mc = list(mrr.exterior.coords[:-1])
    e1 = np.array([mc[1][0] - mc[0][0], mc[1][1] - mc[0][1]])
    e2 = np.array([mc[2][0] - mc[1][0], mc[2][1] - mc[1][1]])
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if l1 >= l2:
        ridge_dir = e1 / l1 if l1 > 0 else np.array([1.0, 0.0])
        half_width = l2 / 2
        ridge_half_len = l1 / 2
    else:
        ridge_dir = e2 / l2 if l2 > 0 else np.array([0.0, 1.0])
        half_width = l1 / 2
        ridge_half_len = l2 / 2
    ridge_normal = np.array([-ridge_dir[1], ridge_dir[0]])
    return ridge_dir, ridge_normal, half_width, ridge_half_len


def _roof_gabled(coords, polygon, roof_height, eave_y):
    """Gabled roof: two sloped planes meeting at a ridge line."""
    n = len(coords)
    ridge_dir, ridge_normal, half_width, ridge_half_len = \
        _roof_ridge_info(polygon)
    cx, cy = polygon.centroid.x, polygon.centroid.y

    # Ridge endpoints (full length of building)
    r0 = [cx - ridge_dir[0] * ridge_half_len,
          cy - ridge_dir[1] * ridge_half_len]
    r1 = [cx + ridge_dir[0] * ridge_half_len,
          cy + ridge_dir[1] * ridge_half_len]

    # Vertices: eave ring + 2 ridge points
    verts = [[c[0], eave_y, c[1]] for c in coords]
    verts.append([r0[0], eave_y + roof_height, r0[1]])
    verts.append([r1[0], eave_y + roof_height, r1[1]])
    ri0, ri1 = n, n + 1

    faces = []
    for i in range(n):
        j = (i + 1) % n
        mid_x = (coords[i][0] + coords[j][0]) / 2
        mid_y = (coords[i][1] + coords[j][1]) / 2
        perp = abs((mid_x - cx) * ridge_normal[0]
                   + (mid_y - cy) * ridge_normal[1])
        if half_width > 0 and perp > half_width * 0.3:
            # Eave edge (far from ridge) → quad to both ridge pts
            faces.append([i, j, ri1])
            faces.append([i, ri1, ri0])
        else:
            # Gable end (near ridge) → triangle to nearest endpoint
            d0 = (mid_x - r0[0]) ** 2 + (mid_y - r0[1]) ** 2
            d1 = (mid_x - r1[0]) ** 2 + (mid_y - r1[1]) ** 2
            nearest = ri0 if d0 <= d1 else ri1
            faces.append([i, j, nearest])
    return verts, faces


def _roof_hipped(coords, polygon, roof_height, eave_y):
    """Hipped roof: like gabled but with sloped hip ends."""
    n = len(coords)
    ridge_dir, ridge_normal, half_width, ridge_half_len = \
        _roof_ridge_info(polygon)
    cx, cy = polygon.centroid.x, polygon.centroid.y

    # Inset ridge from ends so hip faces slope inward
    inset = min(half_width, ridge_half_len * 0.4)
    eff_half = max(0.1, ridge_half_len - inset)
    r0 = [cx - ridge_dir[0] * eff_half,
          cy - ridge_dir[1] * eff_half]
    r1 = [cx + ridge_dir[0] * eff_half,
          cy + ridge_dir[1] * eff_half]

    verts = [[c[0], eave_y, c[1]] for c in coords]
    verts.append([r0[0], eave_y + roof_height, r0[1]])
    verts.append([r1[0], eave_y + roof_height, r1[1]])
    ri0, ri1 = n, n + 1

    faces = []
    for i in range(n):
        j = (i + 1) % n
        mid_x = (coords[i][0] + coords[j][0]) / 2
        mid_y = (coords[i][1] + coords[j][1]) / 2
        perp = abs((mid_x - cx) * ridge_normal[0]
                   + (mid_y - cy) * ridge_normal[1])
        if half_width > 0 and perp > half_width * 0.3:
            faces.append([i, j, ri1])
            faces.append([i, ri1, ri0])
        else:
            d0 = (mid_x - r0[0]) ** 2 + (mid_y - r0[1]) ** 2
            d1 = (mid_x - r1[0]) ** 2 + (mid_y - r1[1]) ** 2
            nearest = ri0 if d0 <= d1 else ri1
            faces.append([i, j, nearest])
    return verts, faces


def _roof_skillion(coords, polygon, roof_height, eave_y,
                   roof_direction: float = None):
    """Skillion (lean-to): single tilted plane.

    When *roof_direction* is given (compass bearing in degrees CW
    from north), it specifies the downslope direction — the low side
    of the roof faces that bearing.  Without it, the slope direction
    is inferred from the polygon's longest axis.
    """
    n = len(coords)
    cx, cy = polygon.centroid.x, polygon.centroid.y

    if roof_direction is not None:
        # roof:direction = compass bearing of downslope.
        # High side is opposite (bearing + 180°).
        # Convert to unit vector in easting/northing space:
        #   bearing 0° (N) → [0, 1],  90° (E) → [1, 0]
        bearing_rad = math.radians(roof_direction)
        # ridge_normal points toward HIGH side (opposite of downslope)
        ridge_normal = np.array([-math.sin(bearing_rad),
                                 -math.cos(bearing_rad)])
        # Compute half_width from polygon extent in normal direction
        perps = [((c[0] - cx) * ridge_normal[0]
                  + (c[1] - cy) * ridge_normal[1]) for c in coords]
        half_width = max(abs(p) for p in perps) if perps else 1.0
    else:
        _, ridge_normal, half_width, _ = _roof_ridge_info(polygon)

    # Height varies linearly across the slope axis
    verts = []
    for c in coords:
        perp = ((c[0] - cx) * ridge_normal[0]
                + (c[1] - cy) * ridge_normal[1])
        if half_width > 0:
            t = max(0.0, min(1.0, (perp / half_width + 1) / 2))
        else:
            t = 0.5
        verts.append([c[0], eave_y + roof_height * t, c[1]])

    # Fan triangulation from vertex 0
    faces = [[0, i, i + 1] for i in range(1, n - 1)]
    return verts, faces
