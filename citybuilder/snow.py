"""Satellite-based snow/ice detection.

Downloads ESRI World Imagery tiles (reusing the vegetation tile cache),
detects snow/ice via HSV thresholding, converts to UTM polygons for
draping on the terrain mesh.
"""

import logging

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape, Polygon, MultiPolygon

from .vegetation import fetch_satellite_image

logger = logging.getLogger(__name__)

# ── HSV thresholds for snow/ice (PIL scale: H=0-255, S=0-255, V=0-255) ──
# Snow in mountain satellite imagery is often shadowed, slightly bluish,
# or mixed with rock — pure white (V>215, S<40) is too strict.
# V > 70% → >180 ; S < 24% → <60
SNOW_S_MAX = 60
SNOW_V_MIN = 180

# Morphological closing iterations — connects nearby snow pixels into
# contiguous regions before polygonizing (prevents over-fragmentation).
MORPH_CLOSE_ITERS = 3

# Minimum polygon area (m²) — discard tiny speckle detections
MIN_AREA_M2 = 200.0


def _detect_snow_mask(image):
    """Apply HSV thresholding to identify snow/ice pixels.

    Returns a boolean 2D numpy array (H x W) where True = snow/ice.
    """
    from scipy.ndimage import binary_closing

    hsv = np.array(image.convert("HSV"))
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = (s <= SNOW_S_MAX) & (v >= SNOW_V_MIN)

    raw_pct = mask.sum() / mask.size * 100
    logger.info(f"Snow mask (raw): {raw_pct:.1f}% snow pixels "
                f"({mask.sum()} / {mask.size})")

    # Morphological closing to bridge small gaps between nearby snow pixels
    if mask.any() and MORPH_CLOSE_ITERS > 0:
        mask = binary_closing(mask, iterations=MORPH_CLOSE_ITERS)
        closed_pct = mask.sum() / mask.size * 100
        logger.info(f"Snow mask (after closing): {closed_pct:.1f}% "
                    f"({mask.sum()} / {mask.size})")

    return mask


def _geo_to_pixel(lon, lat, geo_bounds, img_w, img_h):
    """Convert WGS84 lon/lat to pixel coordinates in the stitched image."""
    west, south, east, north = geo_bounds
    px = (lon - west) / (east - west) * img_w
    py = (north - lat) / (north - south) * img_h
    return px, py


def _subtract_features_from_mask(mask, geo_bounds, subtract_geoms_utm,
                                  inv_transformer):
    """Punch out known features (buildings, roads, water, paved) from mask.

    Parameters
    ----------
    mask : np.ndarray (bool) — snow mask to modify in-place
    geo_bounds : (west, south, east, north) WGS84
    subtract_geoms_utm : list of Shapely geometries in UTM
    inv_transformer : pyproj.Transformer (UTM → WGS84)
    """
    if not subtract_geoms_utm:
        return mask

    img_h, img_w = mask.shape
    punch = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(punch)

    for geom_utm in subtract_geoms_utm:
        if geom_utm is None or geom_utm.is_empty:
            continue
        # Transform UTM → WGS84
        polys = []
        if isinstance(geom_utm, Polygon):
            polys = [geom_utm]
        elif isinstance(geom_utm, MultiPolygon):
            polys = list(geom_utm.geoms)
        else:
            continue

        for poly in polys:
            coords_wgs84 = [
                inv_transformer.transform(x, y)
                for x, y in poly.exterior.coords
            ]
            pixel_coords = [
                _geo_to_pixel(lon, lat, geo_bounds, img_w, img_h)
                for lon, lat in coords_wgs84
            ]
            if len(pixel_coords) >= 3:
                draw.polygon(pixel_coords, fill=255)

    punch_arr = np.array(punch) > 0
    punched = mask.sum()
    mask = mask & ~punch_arr
    logger.info(f"Snow mask: punched {punched - mask.sum()} pixels "
                f"from known features")
    return mask


def detect_snow_polygons(bbox, transformer, subtract_geoms_utm, zoom=17):
    """Detect snow/ice from satellite imagery and return UTM polygons.

    Parameters
    ----------
    bbox : object or dict
        Must have .north/.south/.east/.west (WGS84 degrees).
    transformer : pyproj.Transformer
        Forward transformer (WGS84 → UTM), always_xy=True.
    subtract_geoms_utm : list
        Shapely geometries (buildings, roads, water, paved) in UTM to
        exclude from snow detection.
    zoom : int
        Satellite tile zoom level.

    Returns
    -------
    snow_polygons : list of Shapely Polygon/MultiPolygon in UTM coordinates
    """
    # 0. Compute UTM bounding box for clipping
    north = getattr(bbox, 'north', None) or bbox['north']
    south = getattr(bbox, 'south', None) or bbox['south']
    east = getattr(bbox, 'east', None) or bbox['east']
    west = getattr(bbox, 'west', None) or bbox['west']

    # 1. Fetch satellite image (reuses cached tiles from vegetation)
    image, geo_bounds = fetch_satellite_image(bbox, zoom)

    # 2. Detect snow via HSV
    mask = _detect_snow_mask(image)

    if mask.sum() == 0:
        logger.info("No snow pixels detected — skipping polygon extraction")
        return []

    # 3. Subtract known features
    inv_tf = Transformer.from_crs(
        transformer.target_crs, transformer.source_crs, always_xy=True)
    mask = _subtract_features_from_mask(mask, geo_bounds,
                                         subtract_geoms_utm, inv_tf)

    if mask.sum() == 0:
        logger.info("No snow pixels remaining after feature subtraction")
        return []

    img_h, img_w = mask.shape

    # 4. Convert boolean mask → polygons via rasterio.features.shapes
    # Build an affine transform mapping pixel coords → WGS84
    gw, gs, ge, gn = geo_bounds
    affine = from_bounds(gw, gs, ge, gn, img_w, img_h)

    snow_polygons_utm = []
    mask_uint8 = mask.astype(np.uint8)
    for geom_dict, value in rio_shapes(mask_uint8, mask=mask_uint8,
                                        transform=affine):
        if value == 0:
            continue
        poly_wgs84 = shape(geom_dict)
        if not poly_wgs84.is_valid or poly_wgs84.is_empty:
            continue

        # Transform WGS84 → UTM
        if isinstance(poly_wgs84, Polygon):
            polys = [poly_wgs84]
        elif isinstance(poly_wgs84, MultiPolygon):
            polys = list(poly_wgs84.geoms)
        else:
            continue

        for p in polys:
            ext_utm = [transformer.transform(x, y)
                       for x, y in p.exterior.coords]
            holes_utm = []
            for hole in p.interiors:
                holes_utm.append([transformer.transform(x, y)
                                  for x, y in hole.coords])
            poly_utm = Polygon(ext_utm, holes_utm)
            if poly_utm.is_valid and poly_utm.area >= MIN_AREA_M2:
                snow_polygons_utm.append(poly_utm)

    logger.info(f"Snow detection: {len(snow_polygons_utm)} polygons "
                f"(>= {MIN_AREA_M2} m²) from {img_w}x{img_h}px image")
    return snow_polygons_utm
