"""
Terrain Demo — generates a GLB model showing terrain + buildings
with foundation skirts, using the streets.gl approach.

Techniques demonstrated:
1. SRTM elevation data sampling
2. Terrain mesh via Constrained Delaunay Triangulation (CDT)
3. Building footprint holes cut into terrain
4. Building placement at min(terrain_elevation)
5. Foundation skirt walls from terrain profile down to flat base

Output: output/terrain-demo.glb
"""

import math
import logging
import numpy as np
import triangle as tr
import trimesh
from trimesh.visual.material import PBRMaterial
from pyproj import Transformer
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
# Beacon Hill, Seattle — hilly residential area
BBOX = {
    "north": 47.5672,
    "south": 47.5632,
    "east": -122.3106,
    "west": -122.3166,
}
TERRAIN_GRID_SPACING = 8.0       # metres between terrain grid points
BUILDING_FOOTPRINT_BUFFER = 0.5  # metres to shrink building holes (prevents slivers)
OUTPUT_PATH = "output/terrain-demo.glb"

# Materials
MAT_TERRAIN = PBRMaterial(
    baseColorFactor=[0.42, 0.55, 0.28, 1.0],  # green grass
    metallicFactor=0.0,
    roughnessFactor=0.9,
    name="terrain",
)
MAT_BUILDING = PBRMaterial(
    baseColorFactor=[0.78, 0.75, 0.70, 1.0],  # warm stone
    metallicFactor=0.0,
    roughnessFactor=0.7,
    name="building",
)
MAT_FOUNDATION = PBRMaterial(
    baseColorFactor=[0.55, 0.50, 0.45, 1.0],  # darker stone for skirt
    metallicFactor=0.0,
    roughnessFactor=0.85,
    name="foundation",
)
MAT_ROOF = PBRMaterial(
    baseColorFactor=[0.45, 0.25, 0.22, 1.0],  # reddish roof
    metallicFactor=0.0,
    roughnessFactor=0.6,
    name="roof",
)


def get_elevation_grid(bbox, transformer, spacing=10.0):
    """Sample Copernicus DEM elevation on a regular grid in projected (UTM) coords.

    Returns (grid_x, grid_y, elev_2d) where grid_x/grid_y are 1-D arrays
    and elev_2d is a 2-D array of elevations in metres (normalised, min=0).
    """
    from citybuilder.terrain import get_elevation_grid as _module_get_grid
    grid_x, grid_y, elev_2d, _offset, _available = _module_get_grid(
        bbox, transformer, spacing=spacing)
    return grid_x, grid_y, elev_2d


def sample_elevation_at(x, y, grid_x, grid_y, elev_2d):
    """Bilinear interpolation of elevation at a UTM point."""
    # Clamp to grid bounds
    ix_f = (x - grid_x[0]) / (grid_x[1] - grid_x[0])
    iy_f = (y - grid_y[0]) / (grid_y[1] - grid_y[0])

    ix = int(ix_f)
    iy = int(iy_f)
    fx = ix_f - ix
    fy = iy_f - iy

    # Clamp indices
    ix = max(0, min(ix, len(grid_x) - 2))
    iy = max(0, min(iy, len(grid_y) - 2))
    fx = max(0.0, min(fx, 1.0))
    fy = max(0.0, min(fy, 1.0))

    # Bilinear
    h00 = elev_2d[iy, ix]
    h10 = elev_2d[iy, ix + 1]
    h01 = elev_2d[iy + 1, ix]
    h11 = elev_2d[iy + 1, ix + 1]

    h = (h00 * (1 - fx) * (1 - fy) +
         h10 * fx * (1 - fy) +
         h01 * (1 - fx) * fy +
         h11 * fx * fy)
    return h


def get_buildings(bbox, transformer):
    """Download OSM buildings and project to UTM.

    Returns list of dicts: {polygon_utm, height, footprint_latlon}.
    """
    import osmnx as ox

    bbox_tuple = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    try:
        gdf = ox.features_from_bbox(bbox_tuple, tags={"building": True})
    except Exception as e:
        logger.warning(f"No buildings found: {e}")
        return []

    buildings = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue

        # Project to UTM
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for poly in polys:
            if poly.area < 1e-10:
                continue
            coords = list(poly.exterior.coords)
            proj_coords = [transformer.transform(lon, lat)
                           for lon, lat in coords]
            proj_poly = Polygon(proj_coords)
            if proj_poly.area < 10:  # skip tiny buildings < 10 m²
                continue

            # Height from OSM or default
            h = row.get("height")
            levels = row.get("building:levels")
            if h is not None and not (isinstance(h, float) and math.isnan(h)):
                try:
                    height = float(str(h).replace(" m", "").replace("m", ""))
                except (ValueError, TypeError):
                    height = 8.0
            elif levels is not None and not (isinstance(levels, float) and math.isnan(levels)):
                try:
                    height = int(float(levels)) * 3.0
                except (ValueError, TypeError):
                    height = 8.0
            else:
                height = 8.0

            buildings.append({
                "polygon_utm": proj_poly,
                "height": height,
            })

    logger.info(f"Downloaded {len(buildings)} buildings")
    return buildings


def build_terrain_mesh(grid_x, grid_y, elev_2d, building_polys_utm, origin):
    """Build a terrain mesh using CDT with building footprint holes.

    Args:
        grid_x, grid_y: 1-D arrays of UTM grid coords
        elev_2d: 2-D elevation array
        building_polys_utm: list of Shapely Polygons (UTM) to cut out
        origin: (ox, oy) offset to center the mesh

    Returns a trimesh.Trimesh with vertices in Y-up glTF coords.
    """
    ox, oy = origin

    # Build boundary polygon (bbox in UTM)
    bx0, bx1 = grid_x[0], grid_x[-1]
    by0, by1 = grid_y[0], grid_y[-1]
    boundary = box(bx0, by0, bx1, by1)

    # Subtract building footprints from boundary
    if building_polys_utm:
        buildings_union = unary_union([p.buffer(BUILDING_FOOTPRINT_BUFFER)
                                       for p in building_polys_utm])
        terrain_poly = boundary.difference(buildings_union)
    else:
        terrain_poly = boundary

    # Collect all polygon vertices + segments for CDT input
    vertices = []
    segments = []
    holes = []

    def add_ring(ring_coords, close=True):
        """Add a polygon ring to the CDT input."""
        start_idx = len(vertices)
        coords = list(ring_coords)
        if close and coords[0] == coords[-1]:
            coords = coords[:-1]  # remove duplicate closing vertex
        for c in coords:
            vertices.append([c[0], c[1]])
        n = len(coords)
        for i in range(n):
            segments.append([start_idx + i, start_idx + (i + 1) % n])

    # Process terrain polygon(s)
    if isinstance(terrain_poly, MultiPolygon):
        polys = list(terrain_poly.geoms)
    elif isinstance(terrain_poly, Polygon):
        polys = [terrain_poly]
    else:
        # GeometryCollection — extract polygons
        polys = [g for g in terrain_poly.geoms if isinstance(g, (Polygon, MultiPolygon))]
        expanded = []
        for p in polys:
            if isinstance(p, MultiPolygon):
                expanded.extend(p.geoms)
            else:
                expanded.append(p)
        polys = expanded

    for poly in polys:
        if poly.is_empty or poly.area < 1.0:
            continue
        # Outer ring
        add_ring(poly.exterior.coords)
        # Inner rings (building holes)
        for interior in poly.interiors:
            add_ring(interior.coords)
            # Hole point (inside the interior ring)
            hole_poly = Polygon(interior.coords)
            rep = hole_poly.representative_point()
            holes.append([rep.x, rep.y])

    if not vertices:
        logger.warning("No terrain vertices — returning empty mesh")
        return trimesh.Trimesh()

    # Add interior grid points for denser terrain mesh
    # (only points that fall inside the terrain polygon)
    for y in grid_y[1:-1]:
        for x in grid_x[1:-1]:
            from shapely.geometry import Point
            if terrain_poly.contains(Point(x, y)):
                vertices.append([x, y])

    # Prepare CDT input
    cdt_input = {
        "vertices": np.array(vertices, dtype=np.float64),
        "segments": np.array(segments, dtype=np.int32),
    }
    if holes:
        cdt_input["holes"] = np.array(holes, dtype=np.float64)

    # Run Constrained Delaunay Triangulation
    # 'p' = PSLG mode, 'q' = quality mesh, 'a' = max area
    max_area = (grid_x[1] - grid_x[0]) ** 2 * 2  # roughly 2 grid cells
    cdt_result = tr.triangulate(cdt_input, f"pq30a{max_area:.0f}")

    tri_verts_2d = cdt_result["vertices"]
    tri_faces = cdt_result["triangles"]

    logger.info(f"Terrain CDT: {len(tri_verts_2d)} vertices, {len(tri_faces)} triangles, "
                f"{len(holes)} building holes")

    # Sample elevation at each CDT vertex and build 3D vertices
    # glTF Y-up: [easting, elevation, northing]
    verts_3d = np.zeros((len(tri_verts_2d), 3), dtype=np.float64)
    for i, (vx, vy) in enumerate(tri_verts_2d):
        elev = sample_elevation_at(vx, vy, grid_x, grid_y, elev_2d)
        verts_3d[i] = [vx - ox, elev, vy - oy]  # Y-up

    # CDT produces CCW faces in the X-Z plane → normals point -Y (down).
    # Reverse winding so normals point +Y (up) for glTF Y-up convention.
    tri_faces_fixed = tri_faces[:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices=verts_3d, faces=tri_faces_fixed)
    mesh.visual = trimesh.visual.TextureVisuals(material=MAT_TERRAIN)
    return mesh


def build_building_mesh(bld, grid_x, grid_y, elev_2d, origin):
    """Build a building mesh with foundation skirt.

    Returns (building_mesh, foundation_mesh, roof_mesh) or (None, None, None).
    """
    ox, oy = origin
    poly = bld["polygon_utm"]
    height = bld["height"]

    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    n = len(coords)
    if n < 3:
        return None, None, None

    # Sample terrain elevation at each vertex
    terrain_heights = []
    for x, y in coords:
        h = sample_elevation_at(x, y, grid_x, grid_y, elev_2d)
        terrain_heights.append(h)

    terrain_min = min(terrain_heights)
    terrain_max = max(terrain_heights)
    foundation_height = terrain_max - terrain_min

    # Building base at terrain_max (streets.gl foundation mode)
    # so the building sits on the highest point and the foundation
    # fills down to terrain_min
    base_elev = terrain_max
    top_elev = base_elev + height

    # ── Building walls (from base_elev to top_elev) ──
    wall_verts = []
    wall_faces = []
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = coords[i]
        x1, y1 = coords[j]

        v_base = len(wall_verts)
        # Y-up: [easting, elevation, northing]
        wall_verts.append([x0 - ox, base_elev, y0 - oy])  # bottom-left
        wall_verts.append([x1 - ox, base_elev, y1 - oy])  # bottom-right
        wall_verts.append([x1 - ox, top_elev, y1 - oy])   # top-right
        wall_verts.append([x0 - ox, top_elev, y0 - oy])   # top-left

        # Two triangles per quad (CCW for outward-facing normals)
        wall_faces.append([v_base, v_base + 1, v_base + 2])
        wall_faces.append([v_base, v_base + 2, v_base + 3])

    # ── Roof (flat cap at top_elev) ──
    from shapely.geometry import Polygon as ShapelyPolygon
    from scipy.spatial import Delaunay

    roof_coords_2d = np.array([[x - ox, y - oy] for x, y in coords])
    try:
        dt = Delaunay(roof_coords_2d)
        roof_verts = [[x, top_elev, y] for x, y in roof_coords_2d]
        # Filter triangles to only those inside the polygon
        roof_faces = []
        roof_poly = ShapelyPolygon(roof_coords_2d)
        for simplex in dt.simplices:
            centroid = roof_coords_2d[simplex].mean(axis=0)
            from shapely.geometry import Point
            if roof_poly.contains(Point(centroid)):
                # CCW winding for upward normals
                roof_faces.append([simplex[0], simplex[2], simplex[1]])
    except Exception:
        roof_verts = []
        roof_faces = []

    building_mesh = trimesh.Trimesh(
        vertices=np.array(wall_verts),
        faces=np.array(wall_faces),
    )
    building_mesh.visual = trimesh.visual.TextureVisuals(material=MAT_BUILDING)

    roof_mesh = None
    if roof_verts and roof_faces:
        roof_mesh = trimesh.Trimesh(
            vertices=np.array(roof_verts),
            faces=np.array(roof_faces),
        )
        roof_mesh.visual = trimesh.visual.TextureVisuals(material=MAT_ROOF)

    # ── Foundation skirt (from terrain profile down to base_elev) ──
    # This fills the gap between the sloped terrain and the flat building base
    skirt_verts = []
    skirt_faces = []

    if foundation_height > 0.3:  # only add skirt if there's meaningful slope
        for i in range(n):
            j = (i + 1) % n
            x0, y0 = coords[i]
            x1, y1 = coords[j]
            th0 = terrain_heights[i]
            th1 = terrain_heights[j]

            v_base = len(skirt_verts)
            # Quad from terrain height down to building base
            # Top edge follows terrain, bottom edge is flat at base_elev
            skirt_verts.append([x0 - ox, th0, y0 - oy])       # top-left (at terrain)
            skirt_verts.append([x1 - ox, th1, y1 - oy])       # top-right (at terrain)
            skirt_verts.append([x1 - ox, base_elev, y1 - oy]) # bottom-right (flat base)
            skirt_verts.append([x0 - ox, base_elev, y0 - oy]) # bottom-left (flat base)

            # CCW for outward normals
            skirt_faces.append([v_base, v_base + 1, v_base + 2])
            skirt_faces.append([v_base, v_base + 2, v_base + 3])

    foundation_mesh = None
    if skirt_verts:
        foundation_mesh = trimesh.Trimesh(
            vertices=np.array(skirt_verts),
            faces=np.array(skirt_faces),
        )
        foundation_mesh.visual = trimesh.visual.TextureVisuals(material=MAT_FOUNDATION)

    return building_mesh, foundation_mesh, roof_mesh


def main():
    import pathlib
    pathlib.Path("output").mkdir(exist_ok=True)

    # Set up UTM projection
    center_lat = (BBOX["north"] + BBOX["south"]) / 2
    center_lon = (BBOX["east"] + BBOX["west"]) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}",
                                        always_xy=True)

    # 1. Get elevation grid
    logger.info("Step 1: Sampling Copernicus DEM elevation data...")
    grid_x, grid_y, elev_2d = get_elevation_grid(
        BBOX, transformer, spacing=TERRAIN_GRID_SPACING
    )

    # Origin = center of grid (so model is centered at 0,0)
    ox = (grid_x[0] + grid_x[-1]) / 2
    oy = (grid_y[0] + grid_y[-1]) / 2
    origin = (ox, oy)

    # 2. Get buildings
    logger.info("Step 2: Downloading OSM buildings...")
    buildings = get_buildings(BBOX, transformer)

    # 3. Build terrain mesh with building holes
    logger.info("Step 3: Building terrain mesh with footprint holes...")
    building_polys = [b["polygon_utm"] for b in buildings]
    terrain_mesh = build_terrain_mesh(grid_x, grid_y, elev_2d,
                                       building_polys, origin)

    # 4. Build building meshes with foundation skirts
    logger.info("Step 4: Building meshes with foundation skirts...")
    scene = trimesh.Scene()
    scene.add_geometry(terrain_mesh, node_name="terrain")

    n_foundations = 0
    for i, bld in enumerate(buildings):
        bld_mesh, found_mesh, roof_mesh = build_building_mesh(
            bld, grid_x, grid_y, elev_2d, origin
        )
        if bld_mesh is not None:
            scene.add_geometry(bld_mesh, node_name=f"building_{i}")
        if roof_mesh is not None:
            scene.add_geometry(roof_mesh, node_name=f"roof_{i}")
        if found_mesh is not None:
            scene.add_geometry(found_mesh, node_name=f"foundation_{i}")
            n_foundations += 1

    logger.info(f"Built {len(buildings)} buildings, {n_foundations} with foundation skirts")

    # 5. Export
    logger.info("Step 5: Exporting GLB...")
    scene.export(OUTPUT_PATH)
    logger.info(f"Done! Model saved to {OUTPUT_PATH}")

    # Print stats
    total_verts = sum(g.vertices.shape[0] for g in scene.geometry.values()
                      if hasattr(g, 'vertices'))
    total_faces = sum(g.faces.shape[0] for g in scene.geometry.values()
                      if hasattr(g, 'faces'))
    elev_range = elev_2d.max() - elev_2d.min()
    logger.info(f"Stats: {total_verts} vertices, {total_faces} faces, "
                f"{elev_range:.0f}m elevation range")


if __name__ == "__main__":
    main()
