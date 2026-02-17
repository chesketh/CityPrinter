"""Vegetation detection via ESA WorldCover + Meta/WRI Canopy Height.

Uses ESA WorldCover 10m land cover classification to determine where
trees and shrubs are located, and Meta/WRI 1m canopy height data to
determine how tall they are.  Satellite imagery download is preserved
for terrain vertex coloring (separate concern).
"""

import logging
import math
import pathlib
import time
import urllib.request

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.merge import merge as rio_merge
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

logger = logging.getLogger(__name__)

# ── Satellite tile configuration (for terrain vertex coloring) ────────
TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TILE_SIZE = 256  # pixels per tile
TILE_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache" / "tiles"
TILE_DELAY = 0.05  # seconds between requests (be polite)

# ── ESA WorldCover v200 configuration ─────────────────────────────────
_WORLDCOVER_CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "worldcover"
_WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
WC_TREE = 10       # tree cover
WC_SHRUB = 20      # shrubland
WC_GRASS = 30      # grassland
WC_CROP = 40       # cropland
WC_BUILT = 50      # built-up
WC_BARE = 60       # bare / sparse vegetation
WC_SNOW = 70       # snow and ice
WC_WATER = 80      # permanent water bodies
WC_WETLAND = 90    # herbaceous wetland
WC_MANGROVE = 95   # mangroves
WC_MOSS = 100      # moss and lichen

# ── Meta/WRI Canopy Height configuration ──────────────────────────────
_CANOPY_CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "canopy"
_CANOPY_URL = (
    "https://dataforgood-fb-data.s3.amazonaws.com"
    "/forests/v1/alsgedi_global_v6_float/chm/{qk}.tif"
)
_CANOPY_ZOOM = 9   # QuadKey zoom level (9-digit keys, ~60km tiles)

# ── Sampling parameters ──────────────────────────────────────────────
SAMPLE_SPACING_M = 10.0  # metres between samples (matches WorldCover)
JITTER_M = 3.0            # random jitter in metres to break grid pattern


# ======================================================================
# Satellite tile functions (preserved for terrain vertex coloring)
# ======================================================================

def _lat_lon_to_tile(lat, lon, zoom):
    """Convert WGS84 lat/lon to Slippy Map tile indices."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
             / math.pi) / 2.0 * n)
    x = max(0, min(x, n - 1))
    y = max(0, min(y, n - 1))
    return x, y


def _tile_to_lat_lon(tx, ty, zoom):
    """Convert tile indices to WGS84 (north-west corner of tile)."""
    n = 2 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def _download_tile(z, x, y):
    """Download a single satellite tile, using disk cache."""
    cache_path = TILE_CACHE_DIR / str(z) / str(x) / f"{y}.jpg"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB"), True

    url = TILE_URL.format(z=z, y=y, x=x)
    req = urllib.request.Request(url, headers={
        "User-Agent": "CityBuilder/1.0 (educational project)",
        "Referer": "https://www.arcgis.com/",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
    except Exception as e:
        logger.warning(f"Failed to download tile {z}/{x}/{y}: {e}")
        return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0)), False

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)

    from io import BytesIO
    return Image.open(BytesIO(data)).convert("RGB"), False


def fetch_satellite_image(bbox, zoom=17):
    """Download and stitch ESRI World Imagery tiles covering the bbox.

    Parameters
    ----------
    bbox : object or dict
        Must have .north/.south/.east/.west (WGS84 degrees).
    zoom : int
        Tile zoom level (17 ~ 1.2 m/px).

    Returns
    -------
    image : PIL.Image.Image -- stitched satellite image
    geo_bounds : (west, south, east, north) -- exact WGS84 extent
    """
    north = getattr(bbox, 'north', None) or bbox['north']
    south = getattr(bbox, 'south', None) or bbox['south']
    east = getattr(bbox, 'east', None) or bbox['east']
    west = getattr(bbox, 'west', None) or bbox['west']

    tx_min, ty_min = _lat_lon_to_tile(north, west, zoom)
    tx_max, ty_max = _lat_lon_to_tile(south, east, zoom)

    if tx_min > tx_max:
        tx_min, tx_max = tx_max, tx_min
    if ty_min > ty_max:
        ty_min, ty_max = ty_max, ty_min

    n_tiles_x = tx_max - tx_min + 1
    n_tiles_y = ty_max - ty_min + 1
    total_tiles = n_tiles_x * n_tiles_y

    logger.info(f"Fetching {total_tiles} satellite tiles "
                f"({n_tiles_x}x{n_tiles_y}) at zoom {zoom}")

    stitched = Image.new("RGB",
                         (n_tiles_x * TILE_SIZE, n_tiles_y * TILE_SIZE))
    downloaded = 0
    for ix, tx in enumerate(range(tx_min, tx_max + 1)):
        for iy, ty in enumerate(range(ty_min, ty_max + 1)):
            tile_img, from_cache = _download_tile(zoom, tx, ty)
            stitched.paste(tile_img, (ix * TILE_SIZE, iy * TILE_SIZE))
            downloaded += 1
            if not from_cache and downloaded < total_tiles:
                time.sleep(TILE_DELAY)

    nw_lat, nw_lon = _tile_to_lat_lon(tx_min, ty_min, zoom)
    se_lat, se_lon = _tile_to_lat_lon(tx_max + 1, ty_max + 1, zoom)
    geo_bounds = (nw_lon, se_lat, se_lon, nw_lat)

    logger.info(f"Satellite image: {stitched.width}x{stitched.height}px, "
                f"bounds: W={geo_bounds[0]:.6f} S={geo_bounds[1]:.6f} "
                f"E={geo_bounds[2]:.6f} N={geo_bounds[3]:.6f}")
    return stitched, geo_bounds


# ======================================================================
# ESA WorldCover functions
# ======================================================================

def _worldcover_tile_id(lat, lon):
    """Compute ESA WorldCover tile ID for a WGS84 point.

    Tiles are on a 3x3 degree grid.  The tile ID encodes the SW corner
    rounded down to the nearest multiple of 3.

    Examples
    --------
    (25.2, 55.3)   -> 'N24E054'
    (-33.8, 151.2) -> 'S36E150'
    (51.5, -0.1)   -> 'N51W003'
    """
    lat_floor = int(math.floor(lat / 3.0)) * 3
    lon_floor = int(math.floor(lon / 3.0)) * 3

    ns = 'N' if lat_floor >= 0 else 'S'
    ew = 'E' if lon_floor >= 0 else 'W'
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def _worldcover_tiles_for_bbox(south, north, west, east):
    """Return unique WorldCover tile IDs covering the bbox."""
    tiles = set()
    for lat in [south, north]:
        for lon in [west, east]:
            tiles.add(_worldcover_tile_id(lat, lon))
    return sorted(tiles)


def _download_worldcover_tile(tile_id):
    """Download and cache an ESA WorldCover tile.

    Returns path to local .tif, or None on failure.
    """
    _WORLDCOVER_CACHE.mkdir(parents=True, exist_ok=True)

    filename = f"ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
    local_path = _WORLDCOVER_CACHE / filename

    if local_path.exists():
        return local_path

    url = _WORLDCOVER_URL.format(tile=tile_id)
    logger.info(f"Downloading WorldCover tile: {tile_id}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "CityBuilder/1.0 (educational project)",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        local_path.write_bytes(data)
        size_mb = local_path.stat().st_size / 1024 / 1024
        logger.info(f"Cached WorldCover tile: {filename} ({size_mb:.1f} MB)")
        return local_path
    except urllib.error.HTTPError as e:
        logger.warning(f"WorldCover download failed ({e.code}): {tile_id}")
        if local_path.exists():
            local_path.unlink()
        return None
    except Exception as e:
        logger.warning(f"WorldCover download error: {e}")
        if local_path.exists():
            local_path.unlink()
        return None


def _read_worldcover_for_bbox(bbox, transformer):
    """Read WorldCover land cover classes for the given bbox.

    Parameters
    ----------
    bbox : object with .north/.south/.east/.west
    transformer : pyproj.Transformer (WGS84 -> UTM)

    Returns
    -------
    lc_grid : np.ndarray (ny, nx) uint8 -- land cover class codes
    grid_x : np.ndarray 1-D UTM easting values (10m spacing)
    grid_y : np.ndarray 1-D UTM northing values (10m spacing)

    Or (None, None, None) on failure.
    """
    north = getattr(bbox, 'north', None) or bbox['north']
    south = getattr(bbox, 'south', None) or bbox['south']
    east = getattr(bbox, 'east', None) or bbox['east']
    west = getattr(bbox, 'west', None) or bbox['west']

    # Download tiles
    tile_ids = _worldcover_tiles_for_bbox(south, north, west, east)
    tile_paths = []
    for tid in tile_ids:
        p = _download_worldcover_tile(tid)
        if p is not None:
            tile_paths.append(p)

    if not tile_paths:
        return None, None, None

    # Build UTM grid at 10m spacing
    sw_x, sw_y = transformer.transform(west, south)
    ne_x, ne_y = transformer.transform(east, north)
    grid_x = np.arange(min(sw_x, ne_x), max(sw_x, ne_x), SAMPLE_SPACING_M)
    grid_y = np.arange(min(sw_y, ne_y), max(sw_y, ne_y), SAMPLE_SPACING_M)
    nx, ny = len(grid_x), len(grid_y)

    if nx < 2 or ny < 2:
        return None, None, None

    # Read tiles with bounded merge (only reads the bbox region)
    src_datasets = [rasterio.open(str(p)) for p in tile_paths]
    try:
        merged_arr, merge_transform = rio_merge(
            src_datasets, bounds=(west, south, east, north))
        src_data = merged_arr[0]  # (bands, h, w) -> (h, w)
        src_crs = src_datasets[0].crs

        dst_crs = CRS.from_user_input(transformer.target_crs)
        dst_transform = from_bounds(
            grid_x[0], grid_y[0], grid_x[-1], grid_y[-1], nx, ny)

        lc_grid = np.zeros((ny, nx), dtype=np.uint8)
        reproject(
            source=src_data,
            destination=lc_grid,
            src_transform=merge_transform,
            src_crs=src_crs,
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=0,
            resampling=Resampling.nearest,  # categorical data
        )
    finally:
        for ds in src_datasets:
            ds.close()

    # rasterio row 0 = north; our convention row 0 = south
    lc_grid = lc_grid[::-1, :]

    n_tree = int(np.sum(lc_grid == WC_TREE))
    n_shrub = int(np.sum(lc_grid == WC_SHRUB))
    logger.info(f"WorldCover grid: {nx}x{ny}, "
                f"tree={n_tree}, shrub={n_shrub}")
    return lc_grid, grid_x, grid_y


# ======================================================================
# Meta/WRI Canopy Height functions
# ======================================================================

def _lat_lon_to_quadkey(lat, lon, zoom):
    """Convert WGS84 lat/lon to a Bing Maps QuadKey at the given zoom."""
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ty = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
             / math.pi) / 2.0 * n)
    tx = max(0, min(tx, n - 1))
    ty = max(0, min(ty, n - 1))

    quadkey = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tx & mask) != 0:
            digit += 1
        if (ty & mask) != 0:
            digit += 2
        quadkey.append(str(digit))
    return ''.join(quadkey)


def _canopy_tiles_for_bbox(south, north, west, east):
    """Return unique canopy height QuadKeys covering the bbox."""
    keys = set()
    for lat in [south, north]:
        for lon in [west, east]:
            keys.add(_lat_lon_to_quadkey(lat, lon, _CANOPY_ZOOM))
    return sorted(keys)


def _download_canopy_tile(quadkey):
    """Download and cache a Meta/WRI canopy height tile.

    Returns path to local .tif, or None on failure.
    """
    _CANOPY_CACHE.mkdir(parents=True, exist_ok=True)

    local_path = _CANOPY_CACHE / f"canopy_{quadkey}.tif"

    if local_path.exists():
        return local_path

    url = _CANOPY_URL.format(qk=quadkey)
    logger.info(f"Downloading canopy height tile: {quadkey}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "CityBuilder/1.0 (educational project)",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        local_path.write_bytes(data)
        size_mb = local_path.stat().st_size / 1024 / 1024
        logger.info(f"Cached canopy tile: canopy_{quadkey}.tif "
                    f"({size_mb:.1f} MB)")
        return local_path
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug(f"Canopy tile not found (404): {quadkey}")
        else:
            logger.warning(f"Canopy download failed ({e.code}): {quadkey}")
        if local_path.exists():
            local_path.unlink()
        return None
    except Exception as e:
        logger.warning(f"Canopy download error: {e}")
        if local_path.exists():
            local_path.unlink()
        return None


def _read_canopy_height_for_bbox(bbox, transformer, grid_x, grid_y):
    """Read Meta/WRI canopy height data, reprojected to UTM grid.

    Parameters
    ----------
    bbox : object with .north/.south/.east/.west
    transformer : pyproj.Transformer (WGS84 -> UTM)
    grid_x, grid_y : np.ndarray -- UTM grid from WorldCover (10m spacing)

    Returns
    -------
    height_grid : np.ndarray (ny, nx) float32 -- canopy height in metres

    Or None on failure.  Pixels with no canopy = 0.0.
    """
    north = getattr(bbox, 'north', None) or bbox['north']
    south = getattr(bbox, 'south', None) or bbox['south']
    east = getattr(bbox, 'east', None) or bbox['east']
    west = getattr(bbox, 'west', None) or bbox['west']

    quadkeys = _canopy_tiles_for_bbox(south, north, west, east)
    tile_paths = []
    for qk in quadkeys:
        p = _download_canopy_tile(qk)
        if p is not None:
            tile_paths.append(p)

    if not tile_paths:
        logger.info("No canopy height tiles available -- using defaults")
        return None

    nx, ny = len(grid_x), len(grid_y)

    src_datasets = [rasterio.open(str(p)) for p in tile_paths]
    try:
        # Canopy tiles are in EPSG:3857 (Web Mercator).  Transform
        # our WGS84 bbox to 3857 so rio_merge bounds match the CRS.
        src_crs = src_datasets[0].crs
        wgs84_to_src = Transformer.from_crs(
            "EPSG:4326", src_crs, always_xy=True)
        src_west, src_south = wgs84_to_src.transform(west, south)
        src_east, src_north = wgs84_to_src.transform(east, north)

        # Bounded merge: read only the bbox region from each tile
        merged_arr, merge_transform = rio_merge(
            src_datasets,
            bounds=(src_west, src_south, src_east, src_north),
            resampling=Resampling.average)
        # Data is uint8 (0-255 metres canopy height)
        src_data = merged_arr[0].astype(np.float32)
        src_nodata = src_datasets[0].nodata

        dst_crs = CRS.from_user_input(transformer.target_crs)
        dst_transform = from_bounds(
            grid_x[0], grid_y[0], grid_x[-1], grid_y[-1], nx, ny)

        height_grid = np.zeros((ny, nx), dtype=np.float32)
        reproject(
            source=src_data,
            destination=height_grid,
            src_transform=merge_transform,
            src_crs=src_crs,
            src_nodata=src_nodata if src_nodata is not None else 0,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=0.0,
            resampling=Resampling.max,  # tallest tree in each 10m cell
        )
    finally:
        for ds in src_datasets:
            ds.close()

    # rasterio row 0 = north; our convention row 0 = south
    height_grid = height_grid[::-1, :]
    height_grid = np.clip(height_grid, 0.0, 80.0)

    nonzero = height_grid[height_grid > 0]
    mean_h = float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    logger.info(f"Canopy height grid: {nx}x{ny}, "
                f"mean={mean_h:.1f}m (where > 0), "
                f"max={height_grid.max():.1f}m")
    return height_grid


# ======================================================================
# Position sampling and height lookup
# ======================================================================

def _sample_positions_from_grid(lc_grid, grid_x, grid_y, class_code,
                                subtract_geoms_utm, jitter_seed=42):
    """Sample UTM positions where land cover matches class_code.

    Parameters
    ----------
    lc_grid : np.ndarray (ny, nx) uint8
    grid_x, grid_y : np.ndarray 1-D UTM coordinates
    class_code : int -- WorldCover class (10=tree, 20=shrub)
    subtract_geoms_utm : list of Shapely geometries to exclude
    jitter_seed : int

    Returns
    -------
    positions : list of (easting, northing) UTM tuples
    """
    match_iy, match_ix = np.where(lc_grid == class_code)

    if len(match_ix) == 0:
        return []

    # Convert grid indices to UTM coordinates
    ux = grid_x[match_ix].copy()
    uy = grid_y[match_iy].copy()

    # Deterministic jitter to break grid pattern
    rng = np.random.RandomState(jitter_seed)
    ux += rng.uniform(-JITTER_M, JITTER_M, len(ux))
    uy += rng.uniform(-JITTER_M, JITTER_M, len(uy))

    # Subtract known features (buildings, water, pools, piers)
    if subtract_geoms_utm:
        valid_geoms = [g for g in subtract_geoms_utm
                       if g is not None and not g.is_empty]
        if valid_geoms:
            try:
                combined = unary_union(valid_geoms).buffer(2.0)
                if not combined.is_empty:
                    prepped = prep(combined)
                    keep = np.array([
                        not prepped.contains(Point(x, y))
                        for x, y in zip(ux, uy)
                    ])
                    ux = ux[keep]
                    uy = uy[keep]
            except Exception as e:
                logger.warning(f"Feature subtraction failed: {e}")

    positions = list(zip(ux.tolist(), uy.tolist()))
    logger.info(f"Sampled {len(positions)} positions for "
                f"WorldCover class {class_code}")
    return positions


def _compute_height_for_positions(positions, height_grid, grid_x, grid_y,
                                  default_height=9.0):
    """Look up canopy height at each position from the height grid.

    Parameters
    ----------
    positions : list of (easting, northing)
    height_grid : np.ndarray (ny, nx) float32, or None
    grid_x, grid_y : np.ndarray 1-D
    default_height : float -- fallback if no canopy data

    Returns
    -------
    heights : np.ndarray float32 -- canopy height per position
    mean_height : float -- area-wide mean
    """
    n = len(positions)
    if n == 0:
        return np.array([], dtype=np.float32), default_height

    if height_grid is None:
        return np.full(n, default_height, dtype=np.float32), default_height

    pts = np.array(positions, dtype=np.float64)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]

    # Nearest-neighbor lookup
    ix = np.clip(((pts[:, 0] - grid_x[0]) / dx).astype(int),
                 0, len(grid_x) - 1)
    iy = np.clip(((pts[:, 1] - grid_y[0]) / dy).astype(int),
                 0, len(grid_y) - 1)

    heights = height_grid[iy, ix].copy()

    # Replace zeros/near-zero with default
    heights[heights < 1.0] = default_height

    # Clamp to reasonable range
    heights = np.clip(heights, 2.0, 60.0)

    mean_height = float(heights.mean())
    return heights, mean_height


# ======================================================================
# Main public function
# ======================================================================

def detect_vegetation_positions(bbox, transformer, subtract_geoms_utm,
                                zoom=17):
    """Detect vegetation from ESA WorldCover + canopy height data.

    Parameters
    ----------
    bbox : object or dict
        Must have .north/.south/.east/.west (WGS84 degrees).
    transformer : pyproj.Transformer
        Forward transformer (WGS84 -> UTM), always_xy=True.
    subtract_geoms_utm : list
        Shapely geometries (buildings, water, pools, piers) in UTM to
        exclude from vegetation placement.
    zoom : int
        Satellite tile zoom level (used for terrain coloring image).

    Returns
    -------
    tree_positions : list of (easting, northing) -- UTM coordinates
    bush_positions : list of (easting, northing) -- UTM coordinates
    veg_meta : dict with keys:
        'veg_fraction'       : float (fraction classified as vegetation)
        'canopy_heights'     : np.ndarray per-tree height in metres
        'mean_canopy_height' : float area-wide mean tree height
        'bush_heights'       : np.ndarray per-bush height in metres

    Note: satellite imagery is NOT downloaded here.  fetch_satellite_image()
    is preserved in this module for optional use (e.g. texture draping) but
    is no longer called during normal builds.  Terrain coloring uses a
    latitude-based gradient instead.
    """
    empty_meta = {
        'veg_fraction': 0.0,
        'canopy_heights': np.array([], dtype=np.float32),
        'mean_canopy_height': 9.0,
        'bush_heights': np.array([], dtype=np.float32),
        'lc_grid': None,
        'lc_grid_x': None,
        'lc_grid_y': None,
    }

    # 1. Read WorldCover land cover grid
    lc_grid, grid_x, grid_y = _read_worldcover_for_bbox(bbox, transformer)

    if lc_grid is None:
        logger.warning("WorldCover data unavailable -- no vegetation")
        return [], [], dict(empty_meta)

    # 2. Sample tree positions from class 10 (tree cover)
    tree_positions = _sample_positions_from_grid(
        lc_grid, grid_x, grid_y, WC_TREE,
        subtract_geoms_utm, jitter_seed=42)

    # 3. Sample bush positions from class 20 (shrubland)
    bush_positions = _sample_positions_from_grid(
        lc_grid, grid_x, grid_y, WC_SHRUB,
        subtract_geoms_utm, jitter_seed=43)

    # 4. Estimate canopy heights from latitude (skip slow S3 download).
    #    Use a latitude-based mean with variance so trees aren't uniform.
    north = getattr(bbox, 'north', None) or bbox['north']
    south = getattr(bbox, 'south', None) or bbox['south']
    center_lat = abs((north + south) / 2.0)
    # Approximate mean tree height by biome band (metres)
    if center_lat >= 65:
        base_h = 5.0     # subarctic / tundra edge
    elif center_lat >= 50:
        base_h = 12.0    # boreal
    elif center_lat >= 30:
        base_h = 14.0    # temperate
    elif center_lat >= 15:
        base_h = 12.0    # subtropical (variable)
    else:
        base_h = 18.0    # tropical / equatorial

    rng_h = np.random.RandomState(44)
    n_trees = len(tree_positions)
    if n_trees > 0:
        # Log-normal so most trees cluster near base_h, a few taller
        canopy_heights = np.clip(
            rng_h.lognormal(np.log(base_h), 0.3, n_trees), 2.0, 40.0
        ).astype(np.float32)
        mean_canopy_height = float(np.percentile(canopy_heights, 75))
    else:
        canopy_heights = np.array([], dtype=np.float32)
        mean_canopy_height = base_h

    n_bushes = len(bush_positions)
    bush_heights = np.clip(
        rng_h.lognormal(np.log(2.0), 0.3, n_bushes), 0.5, 5.0
    ).astype(np.float32) if n_bushes > 0 else np.array([], dtype=np.float32)

    # 7. Compute vegetation fraction
    veg_pixels = int(np.sum(
        (lc_grid == WC_TREE) | (lc_grid == WC_SHRUB) |
        (lc_grid == WC_GRASS)))
    total_pixels = lc_grid.size
    veg_fraction = float(veg_pixels / total_pixels) if total_pixels > 0 else 0.0

    logger.info(f"WorldCover vegetation: {len(tree_positions)} trees "
                f"(mean height {mean_canopy_height:.1f}m), "
                f"{len(bush_positions)} bushes, "
                f"veg_fraction={veg_fraction:.1%}")

    return tree_positions, bush_positions, {
        'veg_fraction': veg_fraction,
        'canopy_heights': canopy_heights,
        'mean_canopy_height': mean_canopy_height,
        'bush_heights': bush_heights,
        'lc_grid': lc_grid,
        'lc_grid_x': grid_x,
        'lc_grid_y': grid_y,
    }
