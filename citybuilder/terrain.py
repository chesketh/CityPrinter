"""Terrain elevation sampling and mesh generation via Copernicus DEM.

Provides functions for:
1. Sampling Copernicus GLO-30 elevation data on a regular UTM grid
2. Bilinear interpolation of elevation at arbitrary points
3. Building a terrain surface mesh (CDT) with footprint holes
4. Foundation skirt walls for buildings on slopes
"""

import logging
import math
import pathlib
import urllib.request
import numpy as np
import trimesh
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from trimesh.visual.material import PBRMaterial
from pyproj import Transformer
import shapely as _shapely

logger = logging.getLogger(__name__)

# Default grid spacing in metres between elevation sample points
DEFAULT_GRID_SPACING = 10.0


# ── Copernicus DEM configuration ─────────────────────────────────
_DEM_CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache" / "dem"

# GLO-30 (30m) — res_code 10 in Copernicus naming
_GLO30_BASE_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
_GLO30_RES_CODE = 10

# GLO-90 (90m) fallback for countries excluded from GLO-30
_GLO90_BASE_URL = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
_GLO90_RES_CODE = 30


def _tile_name(lat_floor: int, lon_floor: int, res_code: int) -> str:
    """Build Copernicus DEM tile name from lower-left integer lat/lon."""
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    lat_str = f"{ns}{abs(lat_floor):02d}_00"
    lon_str = f"{ew}{abs(lon_floor):03d}_00"
    return f"Copernicus_DSM_COG_{res_code}_{lat_str}_{lon_str}_DEM"


def _tiles_for_bbox(south: float, north: float,
                    west: float, east: float) -> list:
    """Return list of (lat_floor, lon_floor) for all 1x1° tiles covering bbox."""
    lat_min = math.floor(south)
    lat_max = math.floor(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)
    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            tiles.append((lat, lon))
    return tiles


def _download_tile(lat_floor: int, lon_floor: int) -> pathlib.Path | None:
    """Download and cache a Copernicus DEM tile.

    Tries GLO-30 first; falls back to GLO-90 if unavailable.
    Returns path to local .tif, or None if both fail.
    """
    _DEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for base_url, res_code, label in [
        (_GLO30_BASE_URL, _GLO30_RES_CODE, "GLO-30"),
        (_GLO90_BASE_URL, _GLO90_RES_CODE, "GLO-90"),
    ]:
        name = _tile_name(lat_floor, lon_floor, res_code)
        local_path = _DEM_CACHE_DIR / f"{name}.tif"

        if local_path.exists():
            return local_path

        url = f"{base_url}/{name}/{name}.tif"
        logger.info(f"Downloading {label} tile: {name}")
        try:
            urllib.request.urlretrieve(url, local_path)
            size_mb = local_path.stat().st_size / 1024 / 1024
            logger.info(f"Cached {label} tile: {local_path.name} ({size_mb:.1f} MB)")
            return local_path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"{label} tile not found (404): {name}")
            else:
                logger.warning(f"{label} download failed ({e.code}): {name}")
            if local_path.exists():
                local_path.unlink()
            continue
        except Exception as e:
            logger.warning(f"{label} download error: {e}")
            if local_path.exists():
                local_path.unlink()
            continue

    logger.warning(f"No DEM tile available for ({lat_floor}, {lon_floor})")
    return None


def get_elevation_grid(bbox, transformer, spacing=DEFAULT_GRID_SPACING):
    """Sample Copernicus DEM elevation on a regular UTM grid.

    Parameters
    ----------
    bbox : BoundingBox or dict
        Must have .north/.south/.east/.west (WGS84 degrees).
    transformer : pyproj.Transformer
        Forward transformer (WGS84 → UTM), always_xy=True.
    spacing : float
        Grid spacing in metres.

    Returns
    -------
    grid_x : np.ndarray  — 1-D UTM easting values
    grid_y : np.ndarray  — 1-D UTM northing values
    elev_2d : np.ndarray — 2-D (ny, nx) normalised elevations (min = 0)
    elev_offset : float  — raw minimum elevation subtracted
    terrain_available : bool — True if DEM returned usable data
    """
    # Get bbox as attributes or dict keys
    north = getattr(bbox, 'north', None) or bbox['north']
    south = getattr(bbox, 'south', None) or bbox['south']
    east = getattr(bbox, 'east', None) or bbox['east']
    west = getattr(bbox, 'west', None) or bbox['west']

    # Project bbox corners to UTM
    sw_x, sw_y = transformer.transform(west, south)
    ne_x, ne_y = transformer.transform(east, north)

    grid_x = np.arange(sw_x, ne_x, spacing)
    grid_y = np.arange(sw_y, ne_y, spacing)

    if len(grid_x) < 2 or len(grid_y) < 2:
        grid_x = np.array([sw_x, ne_x])
        grid_y = np.array([sw_y, ne_y])

    nx, ny = len(grid_x), len(grid_y)

    # ── Acquire DEM tiles ────────────────────────────────────
    tile_coords = _tiles_for_bbox(south, north, west, east)
    tile_paths = []
    for lat_f, lon_f in tile_coords:
        p = _download_tile(lat_f, lon_f)
        if p is not None:
            tile_paths.append(p)

    if not tile_paths:
        logger.warning("No Copernicus DEM tiles available — flat terrain")
        elev = np.zeros((ny, nx), dtype=np.float64)
        return grid_x, grid_y, elev, 0.0, False

    # ── Read tiles and reproject to UTM grid ─────────────────
    src_datasets = [rasterio.open(str(p)) for p in tile_paths]

    try:
        if len(src_datasets) == 1:
            src_data = src_datasets[0].read(1).astype(np.float64)
            src_transform = src_datasets[0].transform
            src_crs = src_datasets[0].crs
            src_nodata = src_datasets[0].nodata
        else:
            merged, src_transform = rio_merge(src_datasets)
            src_data = merged[0].astype(np.float64)  # (bands, h, w) → (h, w)
            src_crs = src_datasets[0].crs  # all tiles are EPSG:4326
            src_nodata = src_datasets[0].nodata

        # Target CRS from the forward transformer
        dst_crs = CRS.from_user_input(transformer.target_crs)

        # Build destination transform: pixel grid → UTM coordinates
        # from_bounds expects (west, south, east, north, width, height)
        dst_transform = from_bounds(
            grid_x[0], grid_y[0], grid_x[-1], grid_y[-1], nx, ny)

        elev = np.full((ny, nx), np.nan, dtype=np.float64)

        reproject(
            source=src_data,
            destination=elev,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata if src_nodata is not None else -9999,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    finally:
        for ds in src_datasets:
            ds.close()

    # rasterio row 0 = north; our convention row 0 = south (grid_y[0])
    elev = elev[::-1, :]

    # ── Validate coverage ────────────────────────────────────
    valid_mask = ~np.isnan(elev)
    non_none_count = int(valid_mask.sum())
    total_cells = nx * ny
    terrain_available = non_none_count >= max(4, total_cells * 0.02)

    if not terrain_available:
        elev[:] = 0.0
        logger.warning("Copernicus DEM data insufficient for this area "
                        f"({non_none_count}/{total_cells} valid) — flat terrain")
        return grid_x, grid_y, elev, 0.0, False

    # ── Fill remaining voids (NaN from nodata / tile edges) ──
    if not valid_mask.all():
        from scipy.interpolate import griddata
        void_mask = ~valid_mask
        n_voids = int(void_mask.sum())

        vy, vx = np.where(valid_mask)
        valid_pts = np.column_stack([vx, vy])
        valid_vals = elev[valid_mask]

        ty, tx = np.where(void_mask)
        target_pts = np.column_stack([tx, ty])

        filled = griddata(valid_pts, valid_vals, target_pts, method='linear')

        nan_mask = np.isnan(filled)
        if nan_mask.any():
            nn = griddata(valid_pts, valid_vals, target_pts[nan_mask],
                          method='nearest')
            filled[nan_mask] = nn

        elev[void_mask] = filled
        logger.info(f"Filled {n_voids} DEM voids via linear interpolation")

    # ── Normalise: subtract minimum so lowest point = Y=0 ────
    elev_offset = float(np.nanmin(elev))
    elev -= elev_offset

    logger.info(f"Terrain grid: {nx}x{ny}, "
                f"offset={elev_offset:.0f}m, range={elev.max():.0f}m")
    return grid_x, grid_y, elev, elev_offset, True


def sample_elevation_at(x, y, grid_x, grid_y, elev_2d):
    """Bilinear interpolation of elevation at a UTM point."""
    if len(grid_x) < 2 or len(grid_y) < 2:
        return 0.0

    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    if dx == 0 or dy == 0:
        return 0.0

    ix_f = (x - grid_x[0]) / dx
    iy_f = (y - grid_y[0]) / dy

    ix = int(ix_f)
    iy = int(iy_f)
    fx = ix_f - ix
    fy = iy_f - iy

    # Clamp
    ix = max(0, min(ix, len(grid_x) - 2))
    iy = max(0, min(iy, len(grid_y) - 2))
    fx = max(0.0, min(fx, 1.0))
    fy = max(0.0, min(fy, 1.0))

    h00 = elev_2d[iy, ix]
    h10 = elev_2d[iy, ix + 1]
    h01 = elev_2d[iy + 1, ix]
    h11 = elev_2d[iy + 1, ix + 1]

    return (h00 * (1 - fx) * (1 - fy) +
            h10 * fx * (1 - fy) +
            h01 * (1 - fx) * fy +
            h11 * fx * fy)


def sample_elevation_batch(xs, ys, grid_x, grid_y, elev_2d):
    """Vectorized bilinear interpolation for arrays of UTM points.

    xs, ys: 1-D numpy arrays of coordinates.
    Returns 1-D numpy array of elevations (same length).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if len(grid_x) < 2 or len(grid_y) < 2:
        return np.zeros(len(xs), dtype=np.float64)

    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    if dx == 0 or dy == 0:
        return np.zeros(len(xs), dtype=np.float64)

    ix_f = (xs - grid_x[0]) / dx
    iy_f = (ys - grid_y[0]) / dy

    ix = np.clip(ix_f.astype(np.intp), 0, len(grid_x) - 2)
    iy = np.clip(iy_f.astype(np.intp), 0, len(grid_y) - 2)
    fx = np.clip(ix_f - ix, 0.0, 1.0)
    fy = np.clip(iy_f - iy, 0.0, 1.0)

    h00 = elev_2d[iy, ix]
    h10 = elev_2d[iy, ix + 1]
    h01 = elev_2d[iy + 1, ix]
    h11 = elev_2d[iy + 1, ix + 1]

    return (h00 * (1 - fx) * (1 - fy) +
            h10 * fx * (1 - fy) +
            h01 * (1 - fx) * fy +
            h11 * fx * fy)


def build_terrain_mesh(grid_x, grid_y, elev_2d, hole_polys, origin):
    """Build terrain surface from regular elevation grid.

    Instead of Constrained Delaunay Triangulation (CDT) with exact polygon
    boundaries, this builds a regular grid mesh and removes cells whose
    centers fall inside hole polygons.  Buildings, roads, and water meshes
    overlap the grid-resolution cell edges, making the approximation
    invisible.

    Parameters
    ----------
    grid_x, grid_y : np.ndarray — 1-D grid coordinate arrays (UTM)
    elev_2d : np.ndarray — 2-D normalised elevation array (ny, nx)
    hole_polys : list[Polygon] — footprints to cut (UTM)
    origin : (float, float) — (ox, oy) for centering output

    Returns
    -------
    trimesh.Trimesh — terrain mesh with Y-up glTF vertices
    """
    ox, oy = origin
    nx, ny = len(grid_x), len(grid_y)

    # ── Vertex positions: direct from elevation grid ────────────
    # Each grid point maps 1:1 to a vertex — no interpolation needed.
    xx, yy = np.meshgrid(grid_x, grid_y)       # both (ny, nx)
    verts_3d = np.empty((ny * nx, 3), dtype=np.float64)
    verts_3d[:, 0] = xx.ravel() - ox            # easting
    verts_3d[:, 1] = elev_2d.ravel()             # elevation (Y-up)
    verts_3d[:, 2] = yy.ravel() - oy            # northing

    # ── Face indices: 2 triangles per grid cell ─────────────────
    n_cx, n_cy = nx - 1, ny - 1
    iy_g, ix_g = np.meshgrid(
        np.arange(n_cy), np.arange(n_cx), indexing='ij')
    iy_f = iy_g.ravel()
    ix_f = ix_g.ravel()

    v00 = iy_f * nx + ix_f                  # (iy,   ix)
    v10 = iy_f * nx + (ix_f + 1)            # (iy,   ix+1)
    v01 = (iy_f + 1) * nx + ix_f            # (iy+1, ix)
    v11 = (iy_f + 1) * nx + (ix_f + 1)      # (iy+1, ix+1)

    # Wound for +Y normals: v00→v01→v10  and  v10→v01→v11
    tri1 = np.column_stack([v00, v01, v10])
    tri2 = np.column_stack([v10, v01, v11])

    # ── Remove cells inside hole polygons ───────────────────────
    n_cells = len(iy_f)
    keep = np.ones(n_cells, dtype=bool)

    if hole_polys:
        cell_cx = (grid_x[:-1] + grid_x[1:]) / 2.0
        cell_cy = (grid_y[:-1] + grid_y[1:]) / 2.0
        ccx, ccy = np.meshgrid(cell_cx, cell_cy)

        tree = _shapely.STRtree(hole_polys)
        cell_pts = _shapely.points(ccx.ravel(), ccy.ravel())
        hits, _ = tree.query(cell_pts, predicate='within')
        if len(hits) > 0:
            keep[np.unique(hits)] = False

    face_keep = np.concatenate([keep, keep])
    faces = np.vstack([tri1, tri2])[face_keep]

    n_removed = n_cells * 2 - len(faces)
    logger.info(f"Terrain grid mesh: {len(verts_3d)} verts, "
                f"{len(faces)} faces ({n_removed} cell faces removed)")

    return trimesh.Trimesh(vertices=verts_3d, faces=faces)


def build_foundation_skirt(footprint_coords, terrain_heights,
                           base_elev, origin):
    """Build foundation skirt walls for a building on sloped terrain.

    Creates a quad strip from the terrain profile down to the flat
    building base, filling the gap on the downhill side.

    Parameters
    ----------
    footprint_coords : list of (x, y) — UTM coordinates
    terrain_heights : list of float — terrain elevation at each vertex
    base_elev : float — the building base elevation (max terrain height)
    origin : (float, float) — (ox, oy) for centering

    Returns
    -------
    (verts_list, faces_list) or None if slope is negligible.
    """
    ox, oy = origin
    n = len(footprint_coords)
    if n < 3:
        return None

    terrain_min = min(terrain_heights)
    if base_elev - terrain_min < 0.3:
        return None  # negligible slope

    verts = []
    faces = []
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = footprint_coords[i]
        x1, y1 = footprint_coords[j]
        th0 = terrain_heights[i]
        th1 = terrain_heights[j]

        v_base = len(verts)
        # Top edge follows terrain, bottom is flat at base_elev
        verts.append([x0 - ox, th0, y0 - oy])
        verts.append([x1 - ox, th1, y1 - oy])
        verts.append([x1 - ox, base_elev, y1 - oy])
        verts.append([x0 - ox, base_elev, y0 - oy])

        # CCW for outward normals
        faces.append([v_base, v_base + 1, v_base + 2])
        faces.append([v_base, v_base + 2, v_base + 3])

    return verts, faces
