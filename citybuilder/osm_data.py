"""OSM data download and feature processing."""

import math
import logging
import time
from typing import Any

import numpy as np
import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry import (
    Point, Polygon, MultiPolygon, MultiLineString, LineString, box,
    GeometryCollection,
)
from shapely.ops import unary_union, linemerge, polygonize
from tqdm import tqdm

from .constants import OSM_TAG_OVERRIDES
from .geometry import transform_geometry
from .database import CityDatabase
from .models import BoundingBox

logger = logging.getLogger(__name__)

# ── Overture tile-based caching helpers ─────────────────────────────────
import pathlib

_OVERTURE_TILE_DEG = 0.05  # ~5.5 km at equator
_OVERTURE_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache" / "overture"


def _overture_tile_key(lat_floor: float, lon_floor: float) -> str:
    """Cache key for an Overture tile (SW corner snapped to 0.05° grid)."""
    return f"{lat_floor:+08.3f}_{lon_floor:+09.3f}"


def _overture_tiles_for_bbox(south, north, west, east):
    """Return list of (lat_floor, lon_floor) for 0.05° tiles covering bbox."""
    deg = _OVERTURE_TILE_DEG
    lat_min = math.floor(south / deg) * deg
    lat_max = math.floor(north / deg) * deg
    lon_min = math.floor(west / deg) * deg
    lon_max = math.floor(east / deg) * deg
    tiles = []
    lat = lat_min
    while lat <= lat_max + 1e-9:
        lon = lon_min
        while lon <= lon_max + 1e-9:
            tiles.append((round(lat, 3), round(lon, 3)))
            lon += deg
        lat += deg
    return tiles


def _load_overture_tile(lat_floor, lon_floor, theme, cache_dir):
    """Download/cache a single Overture tile. Returns GeoDataFrame or None."""
    key = _overture_tile_key(lat_floor, lon_floor)
    cache_path = cache_dir / f"{theme}_{key}.parquet"
    if cache_path.exists():
        gdf = gpd.read_parquet(cache_path)
        if len(gdf) == 0:
            return None
        logger.info(f"  cache hit: {cache_path.name} ({len(gdf)} features)")
        return gdf

    import overturemaps
    deg = _OVERTURE_TILE_DEG
    tile_bbox = (lon_floor, lat_floor,
                 round(lon_floor + deg, 3), round(lat_floor + deg, 3))
    reader = overturemaps.record_batch_reader(theme, bbox=tile_bbox)
    if reader is None:
        # Cache empty result so we don't re-query
        gpd.GeoDataFrame().to_parquet(cache_path)
        return None
    table = reader.read_all()
    gdf = gpd.GeoDataFrame.from_arrow(table)
    if len(gdf) == 0:
        gpd.GeoDataFrame().to_parquet(cache_path)
        return None
    gdf = gdf.set_crs("EPSG:4326")
    gdf.to_parquet(cache_path)
    logger.info(f"  cached: {cache_path.name} ({len(gdf)} features)")
    return gdf


def download_osm_data(db: CityDatabase, bbox: BoundingBox, city_id: int,
                      transformer, progress_callback=None) -> None:
    """Download OpenStreetMap data for the specified bounding box."""
    logger.info("Downloading OSM data...")
    import pandas as pd

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # Show osmnx internal logs so we can see exactly what's happening.
    ox.settings.log_console = True
    # Set a short Overpass timeout for ALL queries so nothing stalls.
    prev_req_timeout = ox.settings.requests_timeout
    ox.settings.requests_timeout = 30  # 30 s per Overpass query

    # Create bbox tuple (left/west, bottom/south, right/east, top/north)
    bbox_tuple = (bbox.west, bbox.south, bbox.east, bbox.north)

    # Expanded query bbox — ~200m buffer to catch complete features near edges
    _QUERY_BUFFER_DEG = 0.002
    expanded_bbox = BoundingBox(
        west=bbox.west   - _QUERY_BUFFER_DEG,
        south=bbox.south - _QUERY_BUFFER_DEG,
        east=bbox.east   + _QUERY_BUFFER_DEG,
        north=bbox.north + _QUERY_BUFFER_DEG,
    )
    query_bbox_tuple = (expanded_bbox.west, expanded_bbox.south,
                        expanded_bbox.east, expanded_bbox.north)

    # ── Single combined features query ──────────────────────────
    # Combine all feature tags into ONE Overpass request to avoid
    # rate-limiting.  Multi-key dicts use OR logic at Overpass level.
    combined_tags = {
        'building': True,
        'building:part': True,
        'natural': ['water', 'bay', 'coastline', 'wetland', 'glacier'],
        'waterway': ['river', 'riverbank', 'stream', 'canal', 'dock',
                     'boatyard'],
        'leisure': ['marina', 'pitch', 'track', 'swimming_pool', 'park'],
        'amenity': 'parking',
        'man_made': ['pier', 'bridge', 'tower'],
        'bridge:support': True,
        'railway': ['rail', 'light_rail', 'tram', 'narrow_gauge', 'subway'],
        'landuse': ['commercial', 'industrial', 'retail',
                    'institutional', 'construction',
                    'basin', 'reservoir', 'harbour'],
    }

    buildings = None
    building_parts = None
    water = None
    coastline = None
    glacier = None
    parking = None
    developed = None
    piers = None
    bridge_decks = None
    towers = None
    bridge_supports = None
    pitches = None
    tracks = None
    pools = None
    parks = None
    railways = None

    _osm_t0 = time.perf_counter()
    _osm_timings = {}
    _progress(10, "Surveying the area...")
    try:
        logger.info("  combined features query ...")
        all_features = ox.features_from_bbox(
            bbox=query_bbox_tuple, tags=combined_tags)
        logger.info(f"  combined query returned {len(all_features)} features total")

        # Split results by tag presence
        has_building = all_features.get('building')
        has_natural = all_features.get('natural')
        has_waterway = all_features.get('waterway')
        has_leisure = all_features.get('leisure')
        has_amenity = all_features.get('amenity')
        has_landuse = all_features.get('landuse')
        has_man_made = all_features.get('man_made')
        has_bridge_support = all_features.get('bridge:support')

        # Building parts: rows where 'building:part' column has a truthy value
        # Per Simple 3D Buildings spec, these replace parent building outlines.
        has_building_part = all_features.get('building:part')
        if has_building_part is not None:
            bp_mask = has_building_part.notna() & (has_building_part != 'no')
            bp_df = all_features[bp_mask]
            if len(bp_df) > 0:
                building_parts = bp_df
                logger.info(f"  -> {len(building_parts)} building parts")

        # Buildings: rows where 'building' column has a truthy value
        # Exclude features that are ALSO building parts (avoid double-counting)
        if has_building is not None:
            bld_mask = has_building.notna() & (has_building != 'no')
            if has_building_part is not None:
                bld_mask = bld_mask & ~(has_building_part.notna() & (has_building_part != 'no'))
            bld_df = all_features[bld_mask]
            if len(bld_df) > 0:
                buildings = bld_df
                logger.info(f"  -> {len(buildings)} buildings")

        # Water: natural=water/bay/wetland OR waterway=river/etc/dock/boatyard
        #        OR leisure=marina OR landuse=basin/reservoir/harbour
        water_values = {'water', 'bay', 'wetland'}
        waterway_values = {'river', 'riverbank', 'stream', 'canal', 'dock',
                           'boatyard'}
        landuse_water_values = {'basin', 'reservoir', 'harbour'}
        w_mask = pd.Series(False, index=all_features.index)
        if has_natural is not None:
            w_mask = w_mask | has_natural.isin(water_values)
        if has_waterway is not None:
            w_mask = w_mask | has_waterway.isin(waterway_values)
        if has_leisure is not None:
            w_mask = w_mask | (has_leisure == 'marina')
        if has_landuse is not None:
            w_mask = w_mask | has_landuse.isin(landuse_water_values)
        w_df = all_features[w_mask]
        if len(w_df) > 0:
            water = w_df
            logger.info(f"  -> {len(water)} water features")

        # Coastline: natural=coastline (LINE features — ocean boundary)
        if has_natural is not None:
            cl_mask = has_natural == 'coastline'
            cl_df = all_features[cl_mask]
            if len(cl_df) > 0:
                coastline = cl_df
                logger.info(f"  -> {len(coastline)} coastline features")

        # Glaciers: natural=glacier (polygon features — ice/snow surfaces)
        if has_natural is not None:
            gl_mask = has_natural == 'glacier'
            gl_df = all_features[gl_mask]
            if len(gl_df) > 0:
                glacier = gl_df
                logger.info(f"  -> {len(glacier)} glacier features")

        # Piers: man_made=pier (area polygons — treated as paved)
        if has_man_made is not None:
            pier_mask = has_man_made == 'pier'
            pier_df = all_features[pier_mask]
            if len(pier_df) > 0:
                piers = pier_df
                logger.info(f"  -> {len(piers)} pier features")

            # Bridge decks: man_made=bridge (area polygons — elevated)
            bridge_mask = has_man_made == 'bridge'
            bridge_df = all_features[bridge_mask]
            if len(bridge_df) > 0:
                bridge_decks = bridge_df
                logger.info(f"  -> {len(bridge_decks)} bridge deck features")

            # Towers: man_made=tower (pylons, bridge towers, etc.)
            # Exclude features already captured as buildings to avoid duplicates.
            tower_mask = has_man_made == 'tower'
            if has_building is not None:
                tower_mask = tower_mask & ~(has_building.notna() & (has_building != 'no'))
            tower_df = all_features[tower_mask]
            if len(tower_df) > 0:
                towers = tower_df
                logger.info(f"  -> {len(towers)} tower features (non-building)")

        # Bridge supports: bridge:support=pylon/pier/abutment etc.
        # Exclude features already captured as buildings.
        if has_bridge_support is not None:
            bs_mask = has_bridge_support.notna() & (has_bridge_support != 'no')
            if has_building is not None:
                bs_mask = bs_mask & ~(has_building.notna() & (has_building != 'no'))
            # Also exclude features already captured as towers
            if has_man_made is not None:
                bs_mask = bs_mask & ~(has_man_made == 'tower')
            bs_df = all_features[bs_mask]
            if len(bs_df) > 0:
                bridge_supports = bs_df
                logger.info(f"  -> {len(bridge_supports)} bridge support features")

        # Parking: rows where amenity == 'parking'
        if has_amenity is not None:
            p_mask = has_amenity == 'parking'
            p_df = all_features[p_mask]
            if len(p_df) > 0:
                parking = p_df
                logger.info(f"  -> {len(parking)} parking features")

        # Developed land-use: rows where landuse is one of our targets
        landuse_values = {'commercial', 'industrial', 'retail',
                          'institutional', 'construction'}
        if has_landuse is not None:
            d_mask = has_landuse.isin(landuse_values)
            d_df = all_features[d_mask]
            if len(d_df) > 0:
                developed = d_df
                logger.info(f"  -> {len(developed)} developed-area features")

        # Sports pitches: leisure=pitch
        if has_leisure is not None:
            pitch_mask = has_leisure == 'pitch'
            pitch_df = all_features[pitch_mask]
            if len(pitch_df) > 0:
                pitches = pitch_df
                logger.info(f"  -> {len(pitches)} sports pitch features")

            # Running tracks: leisure=track
            track_mask = has_leisure == 'track'
            track_df = all_features[track_mask]
            if len(track_df) > 0:
                tracks = track_df
                logger.info(f"  -> {len(tracks)} running track features")

            # Swimming pools: leisure=swimming_pool
            pool_mask = has_leisure == 'swimming_pool'
            pool_df = all_features[pool_mask]
            if len(pool_df) > 0:
                pools = pool_df
                logger.info(f"  -> {len(pools)} swimming pool features")

            # Parks: leisure=park
            park_mask = has_leisure == 'park'
            park_df = all_features[park_mask]
            if len(park_df) > 0:
                parks = park_df
                logger.info(f"  -> {len(parks)} park features")

        # Railways: rail, light_rail, tram, etc.
        has_railway = all_features.get('railway')
        if has_railway is not None:
            rail_values = {'rail', 'light_rail', 'tram', 'narrow_gauge',
                           'subway'}
            rail_mask = has_railway.isin(rail_values)
            rail_df = all_features[rail_mask]
            if len(rail_df) > 0:
                railways = rail_df
                logger.info(f"  -> {len(railways)} railway features")

    except Exception as e:
        logger.warning(f"Combined features query failed: {e}")

    _osm_timings['1_overpass_features'] = time.perf_counter() - _osm_t0
    _osm_t0 = time.perf_counter()
    # ── Road network (separate call — returns a graph, not features) ──
    _progress(22, "Mapping road network...")
    roads = None
    try:
        logger.info("  road network query ...")
        roads = ox.graph_from_bbox(
            bbox=query_bbox_tuple,
            network_type='all'
        )
        logger.info(f"  -> road network with {len(roads.edges)} segments")
    except Exception as e:
        logger.warning(f"No road network found in area: {e}")

    # Restore original timeout
    ox.settings.requests_timeout = prev_req_timeout

    if buildings is None and water is None and roads is None:
        logger.warning("No OSM features found — generating ground plane only.")
        return  # GLB generator will still produce a ground plane

    _osm_timings['2_road_network'] = time.perf_counter() - _osm_t0
    _osm_t0 = time.perf_counter()
    # ── Overture Maps: tile-based cache ────────────────────────────────
    # Buildings are cached per 0.05° tile so overlapping bboxes reuse data.
    _OVERTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _progress(30, "Gathering building heights...")
    overture_buildings = None   # height-enriched subset (for OSM matching)
    overture_all = None         # full set (for gap-filling)
    try:
        _ov_tiles = _overture_tiles_for_bbox(
            bbox.south, bbox.north, bbox.west, bbox.east)
        logger.info(f"Overture building tiles: {len(_ov_tiles)} tiles to check")

        _ov_frames = []
        for _lat_f, _lon_f in _ov_tiles:
            _tile_gdf = _load_overture_tile(
                _lat_f, _lon_f, "building", _OVERTURE_CACHE_DIR)
            if _tile_gdf is not None and len(_tile_gdf) > 0:
                _ov_frames.append(_tile_gdf)

        if _ov_frames:
            overture_all_raw = pd.concat(_ov_frames, ignore_index=True)
            # Clip to actual build bbox
            _bbox_geom = box(bbox.west, bbox.south, bbox.east, bbox.north)
            _mask = overture_all_raw.geometry.intersects(_bbox_geom)
            overture_all_raw = overture_all_raw[_mask].reset_index(drop=True)
            # Deduplicate buildings on tile boundaries
            if 'id' in overture_all_raw.columns:
                overture_all_raw = overture_all_raw.drop_duplicates(subset='id')

            logger.info(f"  -> {len(overture_all_raw)} Overture buildings in bbox")
            overture_all = overture_all_raw
            has_h = overture_all_raw['height'].notna() if 'height' in overture_all_raw.columns else pd.Series(False, index=overture_all_raw.index)
            has_f = overture_all_raw['num_floors'].notna() if 'num_floors' in overture_all_raw.columns else pd.Series(False, index=overture_all_raw.index)
            has_p = overture_all_raw['has_parts'].fillna(False).astype(bool) if 'has_parts' in overture_all_raw.columns else pd.Series(False, index=overture_all_raw.index)
            overture_buildings = overture_all_raw[has_h | has_f | has_p]
            if len(overture_buildings) > 0:
                logger.info(f"  -> {len(overture_buildings)} Overture buildings with height data")
            else:
                overture_buildings = None
                logger.info("  -> no Overture buildings with height data in this area")
    except ImportError:
        logger.warning("overturemaps package not installed — skipping Overture height enrichment")
    except Exception as e:
        logger.warning(f"Overture Maps query failed: {e}")
        overture_buildings = None
        overture_all = None

    # Download building parts (same tile-based cache)
    _progress(38, "Analyzing building details...")
    overture_parts = None
    if overture_buildings is not None:
        try:
            _pt_frames = []
            for _lat_f, _lon_f in _ov_tiles:
                _tile_gdf = _load_overture_tile(
                    _lat_f, _lon_f, "building_part", _OVERTURE_CACHE_DIR)
                if _tile_gdf is not None and len(_tile_gdf) > 0:
                    _pt_frames.append(_tile_gdf)

            if _pt_frames:
                overture_parts = pd.concat(_pt_frames, ignore_index=True)
                _bbox_geom_pt = box(bbox.west, bbox.south, bbox.east, bbox.north)
                _mask_pt = overture_parts.geometry.intersects(_bbox_geom_pt)
                overture_parts = overture_parts[_mask_pt].reset_index(drop=True)
                if 'id' in overture_parts.columns:
                    overture_parts = overture_parts.drop_duplicates(subset='id')

            if overture_parts is not None and len(overture_parts) > 0:
                has_h = overture_parts['height'].notna() if 'height' in overture_parts.columns else pd.Series(False, index=overture_parts.index)
                has_f = overture_parts['num_floors'].notna() if 'num_floors' in overture_parts.columns else pd.Series(False, index=overture_parts.index)
                overture_parts = overture_parts[has_h | has_f]
                if len(overture_parts) > 0:
                    logger.info(f"  -> {len(overture_parts)} Overture building parts with height data")
                else:
                    overture_parts = None
            else:
                overture_parts = None
        except Exception as e:
            logger.warning(f"Overture building_part query failed: {e}")
            overture_parts = None
    # ── End Overture Maps ───────────────────────────────────────────────

    _osm_timings['3_overture_heights'] = time.perf_counter() - _osm_t0
    _osm_t0 = time.perf_counter()
    _progress(44, "Processing buildings...")
    process_features(
        db, transformer,
        buildings if buildings is not None else gpd.GeoDataFrame(),
        water if water is not None else gpd.GeoDataFrame(),
        roads if roads is not None else nx.MultiDiGraph(),
        city_id,
        parking=parking if parking is not None else gpd.GeoDataFrame(),
        developed=developed if developed is not None else gpd.GeoDataFrame(),
        piers=piers if piers is not None else gpd.GeoDataFrame(),
        bridge_decks=bridge_decks if bridge_decks is not None else gpd.GeoDataFrame(),
        towers=towers if towers is not None else gpd.GeoDataFrame(),
        bridge_supports=bridge_supports if bridge_supports is not None else gpd.GeoDataFrame(),
        coastline=coastline if coastline is not None else gpd.GeoDataFrame(),
        pitches=pitches if pitches is not None else gpd.GeoDataFrame(),
        tracks=tracks if tracks is not None else gpd.GeoDataFrame(),
        pools=pools if pools is not None else gpd.GeoDataFrame(),
        parks=parks if parks is not None else gpd.GeoDataFrame(),
        railways=railways if railways is not None else gpd.GeoDataFrame(),
        glaciers=glacier if glacier is not None else gpd.GeoDataFrame(),
        clip_bbox=expanded_bbox,
        overture_buildings=overture_buildings,
        overture_all=overture_all,
        overture_parts=overture_parts,
        osm_building_parts=building_parts,
        progress_callback=progress_callback,
    )
    _osm_timings['4_process_features'] = time.perf_counter() - _osm_t0

    # ── Timing summary ──
    logger.info("=" * 60)
    logger.info("OSM DOWNLOAD TIMING BREAKDOWN")
    logger.info("=" * 60)
    _osm_total = 0.0
    for _lbl, _dur in sorted(_osm_timings.items()):
        logger.info(f"  {_lbl}: {_dur:.1f}s")
        _osm_total += _dur
    logger.info(f"  TOTAL: {_osm_total:.1f}s")
    logger.info("=" * 60)



def _process_structural_features(db, city_id, features_df, transformer,
                                  clip_poly, label):
    """Store tower / bridge-support polygon features as buildings.

    These are structural elements (pylons, towers, abutments) that render
    like buildings — extruded from ground level with their OSM height.
    """
    if features_df is None or len(features_df) == 0:
        return
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    count = 0
    for idx, row in features_df.iterrows():
        if not (row.geometry and row.geometry.is_valid and not row.geometry.is_empty):
            continue
        try:
            raw_geom = row.geometry
            if clip_poly is not None:
                if not raw_geom.intersects(clip_poly):
                    continue
            geom_proj = transform_geometry(raw_geom, transformer)
            if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                continue
            area = geom_proj.area
            if not area or np.isnan(area) or area <= 0:
                continue
            # Extract height — use OSM value or a reasonable default
            height = row.get('height', None)
            try:
                height = float(str(height).replace('m', '').strip())
            except (ValueError, TypeError):
                height = None
            if height is None or np.isnan(height) or height <= 0:
                height = 20.0  # default tower height
            db.store_features(
                city_id, 'building', geom_proj,
                {'height': height, 'type': label,
                 'name': str(row.get('name', ''))},
                source='osm', confidence=1.0, osm_id=str(idx))
            count += 1
        except Exception as e:
            logger.error(f"Error processing {label} feature {idx}: {e}")
    if count > 0:
        logger.info(f"Stored {count} {label} features as buildings")


def process_features(db: CityDatabase, transformer,
                     buildings: gpd.GeoDataFrame,
                     water: gpd.GeoDataFrame,
                     roads: nx.MultiDiGraph,
                     city_id: int,
                     parking: gpd.GeoDataFrame | None = None,
                     developed: gpd.GeoDataFrame | None = None,
                     piers: gpd.GeoDataFrame | None = None,
                     bridge_decks: gpd.GeoDataFrame | None = None,
                     towers: gpd.GeoDataFrame | None = None,
                     bridge_supports: gpd.GeoDataFrame | None = None,
                     coastline: gpd.GeoDataFrame | None = None,
                     pitches: gpd.GeoDataFrame | None = None,
                     tracks: gpd.GeoDataFrame | None = None,
                     pools: gpd.GeoDataFrame | None = None,
                     parks: gpd.GeoDataFrame | None = None,
                     railways: gpd.GeoDataFrame | None = None,
                     glaciers: gpd.GeoDataFrame | None = None,
                     clip_bbox: 'BoundingBox | None' = None,
                     progress_callback=None,
                     overture_buildings: gpd.GeoDataFrame | None = None,
                     overture_all: gpd.GeoDataFrame | None = None,
                     overture_parts: gpd.GeoDataFrame | None = None,
                     osm_building_parts: gpd.GeoDataFrame | None = None) -> None:
    """Process and store different feature types."""
    import pandas as pd
    logger.info("Processing features...")

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # Build a WGS84 clip polygon from the bounding box so we can
    # truncate large features (e.g. Lake Union) to the requested area.
    clip_poly = None
    if clip_bbox is not None:
        clip_poly = box(clip_bbox.west, clip_bbox.south,
                        clip_bbox.east, clip_bbox.north)

    # Build spatial index for Overture height lookups
    overture_sindex = None
    if overture_buildings is not None and len(overture_buildings) > 0:
        overture_sindex = overture_buildings.sindex
        logger.info(f"Built spatial index for {len(overture_buildings)} Overture buildings with height data")

    # Build spatial index for full Overture set (gap-fill deduplication)
    overture_all_sindex = None
    matched_overture_indices = set()  # tracks which overture_all rows overlap OSM
    if overture_all is not None and len(overture_all) > 0:
        overture_all_sindex = overture_all.sindex
        logger.info(f"Built spatial index for {len(overture_all)} Overture buildings (gap-fill)")

    # Build lookup: Overture building_id → list of parts
    overture_parts_by_building = {}
    if overture_parts is not None and len(overture_parts) > 0:
        for _, part_row in overture_parts.iterrows():
            bid = part_row.get('building_id')
            if bid:
                overture_parts_by_building.setdefault(bid, []).append(part_row)
        logger.info(f"  {len(overture_parts_by_building)} buildings have detailed parts")

    # ── OSM building:part processing (Simple 3D Buildings spec) ──
    # Per spec: when building:part features exist, they replace the
    # parent building outline for 3D rendering.  We build a spatial
    # index so we can skip parent buildings that have parts.
    parents_with_parts = set()  # set of parent building indices to skip
    # Map parent idx → list of overlapping part geometries (for remainder calc)
    parent_part_geoms: dict[Any, list] = {}
    # Reverse map: part label index → parent building row (for property inheritance)
    part_to_parent: dict[Any, Any] = {}
    if osm_building_parts is not None and len(osm_building_parts) > 0:
        # Build spatial index of building parts
        bp_sindex = osm_building_parts.sindex

        # For each parent building, collect overlapping parts
        for bld_idx, bld_row in buildings.iterrows():
            if bld_row.geometry is None or bld_row.geometry.is_empty:
                continue
            candidates = list(bp_sindex.intersection(bld_row.geometry.bounds))
            overlapping = []
            for c_idx in candidates:
                part_row = osm_building_parts.iloc[c_idx]
                try:
                    if bld_row.geometry.intersects(part_row.geometry):
                        overlapping.append(part_row.geometry)
                        # Record reverse mapping for property inheritance
                        part_label = osm_building_parts.index[c_idx]
                        part_to_parent[part_label] = bld_row
                except Exception:
                    continue
            if overlapping:
                parents_with_parts.add(bld_idx)
                parent_part_geoms[bld_idx] = overlapping

        logger.info(f"  {len(parents_with_parts)} parent buildings have OSM building:part features")

        # Process OSM building parts as buildings
        logger.info("Processing OSM building parts...")
        skipped_no_height = 0
        parts_batch = []  # (geometry, properties, confidence, osm_id)
        for idx, row in tqdm(osm_building_parts.iterrows(), total=len(osm_building_parts), desc="Parts"):
            if not row.geometry or row.geometry.is_empty:
                continue

            # building:part=roof — render roof-only (no wall extrusion).
            # Per OSMBuilding: wall is hidden, only roof shape is rendered.
            part_type = row.get('building:part')

            try:
                raw_geom = row.geometry
                if clip_poly is not None:
                    if not raw_geom.intersects(clip_poly):
                        continue

                geom_proj = transform_geometry(raw_geom, transformer)
                if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                    continue

                area = geom_proj.area
                if not area or np.isnan(area) or area <= 0:
                    continue

                # Extract properties from OSM building:part tags
                properties = {'area': float(area), 'is_part': True}
                if part_type == 'roof':
                    properties['part_type'] = 'roof'

                # Apply per-feature tag overrides (fills missing tags only)
                overrides = OSM_TAG_OVERRIDES.get(idx)

                # Parse OSM tags first (needed for fallback calc)
                osm_h = row.get('height')
                osm_levels = row.get('building:levels')
                osm_min_h = row.get('min_height')
                osm_roof_shape = row.get('roof:shape')
                osm_roof_h = row.get('roof:height')
                osm_roof_dir = row.get('roof:direction')

                if overrides:
                    if pd.isna(osm_roof_shape) and 'roof:shape' in overrides:
                        osm_roof_shape = overrides['roof:shape']
                    if pd.isna(osm_roof_h) and 'roof:height' in overrides:
                        osm_roof_h = overrides['roof:height']
                    if pd.isna(osm_roof_dir) and 'roof:direction' in overrides:
                        osm_roof_dir = overrides['roof:direction']
                    if 'height' in overrides:
                        osm_h = overrides['height']

                # min_height (always extract regardless of height source)
                if pd.notna(osm_min_h):
                    try:
                        properties['min_height'] = float(str(osm_min_h).replace(' m', '').replace('m', ''))
                    except (ValueError, TypeError):
                        pass

                # Roof shape and height from OSM
                if pd.notna(osm_roof_shape):
                    properties['roof_shape'] = str(osm_roof_shape)
                if pd.notna(osm_roof_h):
                    try:
                        properties['roof_height'] = float(str(osm_roof_h).replace(' m', '').replace('m', ''))
                    except (ValueError, TypeError):
                        pass
                if pd.notna(osm_roof_dir):
                    try:
                        properties['roof_direction'] = float(str(osm_roof_dir))
                    except (ValueError, TypeError):
                        pass

                # Height cascade for building parts (priority order):
                # Parts have their own detailed OSM tags.  Overture heights
                # represent the whole-building envelope (e.g. 96m for the
                # Palace of Westminster) and would wrongly flatten every
                # sub-part to the max height.  So skip Overture for parts.
                # 1) OSM height tag (on part itself)
                # 2) OSM building:levels × 3m (on part itself)
                # 3) Inherit parent building height/levels (per OSMBuilding)
                # 4) roof_height + 3 (OSMBuilding final fallback)
                if pd.notna(osm_h):
                    try:
                        h_val = float(str(osm_h).replace(' m', '').replace('m', ''))
                        if h_val > 0:
                            properties['height'] = h_val
                            properties['height_source'] = 'osm_part'
                    except (ValueError, TypeError):
                        pass

                if 'height' not in properties and pd.notna(osm_levels):
                    try:
                        properties['height'] = int(float(osm_levels)) * 3.0
                        properties['height_source'] = 'osm_part_levels'
                    except (ValueError, TypeError):
                        pass

                # Per OSMBuilding: building:part=roof with no roof:shape
                # produces a zero-height flat extrusion with hidden walls
                # — effectively invisible.  Skip these.
                if part_type == 'roof' and 'roof_shape' not in properties:
                    skipped_no_height += 1
                    continue

                # 5) Inherit parent building height/levels (per OSMBuilding)
                if 'height' not in properties and idx in part_to_parent:
                    parent = part_to_parent[idx]
                    parent_h = parent.get('height')
                    parent_levels = parent.get('building:levels')
                    if pd.notna(parent_h):
                        try:
                            h_val = float(str(parent_h).replace(' m', '').replace('m', ''))
                            if h_val > 0:
                                properties['height'] = h_val
                                properties['height_source'] = 'parent_inherited'
                        except (ValueError, TypeError):
                            pass
                    if 'height' not in properties and pd.notna(parent_levels):
                        try:
                            properties['height'] = int(float(parent_levels)) * 3.0
                            properties['height_source'] = 'parent_levels_inherited'
                        except (ValueError, TypeError):
                            pass

                # 6) OSMBuilding fallback: roof_height + 3 (one story wall)
                if 'height' not in properties:
                    rh = properties.get('roof_height', 0)
                    properties['height'] = rh + 3.0
                    properties['height_source'] = 'osm_default'

                parts_batch.append(
                    (geom_proj, properties, 0.95, str(idx)))
            except Exception as e:
                logger.error(f"Error processing building part {idx}: {e}")
                continue

        # ── Smart base fills for floating parts ──
        # Group parts by parent building, then only fill vertical gaps
        # that no sibling part already covers.  This prevents generating
        # massive pillars (e.g. a 510m column for a CN Tower antenna
        # part when the main shaft already covers 0-440m).
        base_fill_batch = []
        # Build mapping: parent building → list of (geom, min_h, height)
        parent_parts_map: dict[Any, list] = {}  # parent_idx → [(geom, min_h, h)]
        for geom_proj, props, conf, osm_id in parts_batch:
            min_h = props.get('min_height', 0)
            h = props.get('height', 0)
            # Find which parent this part belongs to
            parent_idx = None
            try:
                # osm_id is str(idx), reconstruct the original index
                orig_idx = eval(osm_id) if osm_id.startswith('(') else osm_id
                if orig_idx in part_to_parent:
                    # Find the parent building index
                    parent_row = part_to_parent[orig_idx]
                    parent_idx = parent_row.name  # the index of the parent
            except Exception:
                pass
            if parent_idx is None:
                parent_idx = f'_orphan_{osm_id}'
            parent_parts_map.setdefault(parent_idx, []).append(
                (geom_proj, min_h, h, props, osm_id))

        base_fills_created = 0
        for parent_idx, siblings in parent_parts_map.items():
            for geom_proj, min_h, h, props, osm_id in siblings:
                if min_h <= 2:
                    continue  # part starts at ground, no fill needed
                # Check if any sibling part covers ground-to-min_h
                # at this location (spatial overlap check)
                covered = False
                for s_geom, s_min_h, s_h, _, _ in siblings:
                    if s_geom is geom_proj:
                        continue  # skip self
                    # Sibling covers vertically if it starts at/near ground
                    # and reaches up to at least this part's min_height
                    if s_min_h <= 2 and s_h >= min_h * 0.8:
                        # Check spatial overlap
                        try:
                            if geom_proj.intersects(s_geom):
                                overlap = geom_proj.intersection(s_geom).area
                                if overlap > 0.3 * geom_proj.area:
                                    covered = True
                                    break
                        except Exception:
                            continue
                if not covered:
                    base_props = {
                        'area': props.get('area', 0),
                        'height': min_h,
                        'height_source': 'part_base_fill',
                        'building_type': props.get('building_type',
                                                   'yes'),
                    }
                    base_fill_batch.append(
                        (geom_proj, base_props, 0.9,
                         f'{osm_id}_base'))
                    base_fills_created += 1

        all_parts = parts_batch + base_fill_batch
        if all_parts:
            stored = db.store_features_batch(city_id, 'building',
                                             all_parts, source='osm')
            logger.info(f"  Stored {stored} OSM building parts "
                        f"({len(parts_batch)} parts + "
                        f"{base_fills_created} base fills)")

        if skipped_no_height:
            logger.info(f"  Skipped {skipped_no_height} parts with no height data")

    # Process buildings with enhanced attributes
    logger.info("Processing buildings...")
    remainder_stored = 0
    # Collect buildings for batch insert (much faster than per-row INSERT)
    osm_building_batch = []       # (geometry, properties, confidence, osm_id)
    overture_building_batch = []  # parts from Overture with separate source
    for idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Buildings"):
        # Per Simple 3D Buildings spec: when building:part features exist,
        # the parent building outline is NOT rendered.  However, if parts
        # only partially cover the parent footprint, render the UNCOVERED
        # remainder as a base so the building isn't left hollow.
        if idx in parents_with_parts:
            # Even though the parent building is NOT rendered (parts replace
            # it), we still need to mark overlapping Overture buildings as
            # matched so they don't get added as gap-fill duplicates.
            if overture_all_sindex is not None and row.geometry is not None:
                ov_candidates = list(overture_all_sindex.intersection(row.geometry.bounds))
                osm_area_wgs = row.geometry.area
                for oci in ov_candidates:
                    ov_geom = overture_all.iloc[oci].geometry
                    if ov_geom is None or ov_geom.is_empty:
                        continue
                    try:
                        overlap = row.geometry.intersection(ov_geom).area
                        threshold = 0.5 * min(osm_area_wgs, ov_geom.area)
                        if overlap > threshold:
                            matched_overture_indices.add(oci)
                    except Exception:
                        continue
            part_geoms_list = parent_part_geoms.get(idx, [])
            parent_geom = row.geometry
            if parent_geom is not None and not parent_geom.is_empty and part_geoms_list:
                try:
                    parts_union = unary_union(part_geoms_list)
                    coverage = parts_union.area / parent_geom.area if parent_geom.area > 0 else 1.0
                    if coverage < 0.8:
                        # Parts cover <80% of parent — render uncovered remainder
                        remainder = parent_geom.difference(parts_union.buffer(0.000005))
                        if not remainder.is_empty and remainder.area > 0.0000000001:
                            remainder_proj = transform_geometry(remainder, transformer)
                            if remainder_proj and remainder_proj.is_valid and not remainder_proj.is_empty:
                                # Use min part height as base height, or estimate from building type
                                base_h = 10.0  # conservative default
                                btype = str(row.get('building', 'yes')).lower()
                                type_h = {'church': 20, 'cathedral': 25, 'basilica': 25,
                                          'chapel': 12, 'temple': 20, 'mosque': 15,
                                          'civic': 15, 'public': 12}.get(btype)
                                if type_h:
                                    base_h = type_h
                                rem_props = {
                                    'area': float(remainder_proj.area),
                                    'height': base_h,
                                    'height_source': 'parent_remainder',
                                    'building_type': btype,
                                }
                                osm_building_batch.append(
                                    (remainder_proj, rem_props, 0.8,
                                     f'{idx}_remainder'))
                                remainder_stored += 1
                                logger.debug(f"  Stored remainder for {idx}: {remainder_proj.area:.0f}m2, h={base_h}m ({btype})")
                except Exception as e:
                    logger.warning(f"Failed to compute remainder for {idx}: {e}")
            continue

        if not row.geometry or not hasattr(row.geometry, 'is_valid') or not row.geometry.is_valid or row.geometry.is_empty:
            logger.warning(f"Skipping invalid building geometry for index {idx}")
            continue

        try:
            # Keep full building geometry — no clipping at bbox edge.
            # Buildings that intersect the bbox are included whole;
            # terrain expands to cover any overhang.
            raw_geom = row.geometry
            if clip_poly is not None:
                if not raw_geom.intersects(clip_poly):
                    continue

            # Track overlapping Overture buildings for gap-fill dedup
            if overture_all_sindex is not None:
                ov_candidates = list(overture_all_sindex.intersection(raw_geom.bounds))
                osm_area_wgs = raw_geom.area
                for oci in ov_candidates:
                    ov_geom = overture_all.iloc[oci].geometry
                    if ov_geom is None or ov_geom.is_empty:
                        continue
                    try:
                        overlap = raw_geom.intersection(ov_geom).area
                        threshold = 0.5 * min(osm_area_wgs, ov_geom.area)
                        if overlap > threshold:
                            matched_overture_indices.add(oci)
                    except Exception:
                        continue

            # Transform coordinates to local projection
            geom_proj = transform_geometry(raw_geom, transformer)
            if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                logger.warning(f"Skipping building {idx} due to invalid transformed geometry")
                continue

            # Calculate area and skip if invalid
            area = geom_proj.area
            if not area or np.isnan(area) or area <= 0:
                logger.warning(f"Skipping building {idx} due to invalid area")
                continue

            # Get building properties from OSM, with type checking
            properties = {}
            try:
                # Basic properties with type validation
                if hasattr(row, 'building'):
                    properties['building_type'] = str(row.building) if row.building else 'yes'
                if hasattr(row, 'building:levels') and row['building:levels']:
                    try:
                        lvl = float(row['building:levels'])
                        if not math.isnan(lvl) and lvl > 0:
                            properties['levels'] = int(lvl)
                    except (ValueError, TypeError):
                        pass

                # Try multiple height-related OSM tags
                for tag in ['height', 'building:height', 'est_height']:
                    if tag in row.index:
                        try:
                            raw = row[tag]
                            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                                continue
                            raw_str = str(raw).strip().lower()
                            # Handle "XX ft" / "XX m" suffixes
                            if raw_str.endswith(' ft') or raw_str.endswith("'"):
                                h = float(raw_str.replace(' ft', '').replace("'", '')) * 0.3048
                            elif raw_str.endswith(' m'):
                                h = float(raw_str.replace(' m', ''))
                            else:
                                h = float(raw_str)
                            if not math.isnan(h) and h > 0:
                                properties['height'] = h
                                break
                        except (ValueError, TypeError):
                            continue

                # Optional string properties
                for prop in ['name', 'amenity', 'building:material', 'roof:shape', 'roof:material']:
                    if hasattr(row, prop) and getattr(row, prop):
                        val = getattr(row, prop)
                        if not (isinstance(val, float) and math.isnan(val)):
                            properties[prop] = str(val)

                properties['area'] = float(area)
            except Exception as e:
                logger.warning(f"Error processing building properties for {idx}: {e}")
                properties = {
                    'building_type': 'yes',
                    'area': float(area)
                }

            # ── Height resolution chain ──────────────────────────────
            # Priority: Overture parts > Overture height > OSM tags > estimates

            # 1. Match to Overture building by spatial overlap
            matched_ov_id = None
            matched_ov_has_parts = False
            ov_height = None
            ov_floors = None
            ov_roof_shape = None
            ov_roof_height = None
            if overture_sindex is not None:
                candidates = list(overture_sindex.intersection(row.geometry.bounds))
                best_overlap = 0.0
                for ci in candidates:
                    ov_row = overture_buildings.iloc[ci]
                    if ov_row.geometry is None or ov_row.geometry.is_empty:
                        continue
                    try:
                        overlap = row.geometry.intersection(ov_row.geometry).area
                    except Exception:
                        continue
                    if overlap > best_overlap:
                        best_overlap = overlap
                        matched_ov_id = ov_row.get('id')
                        matched_ov_has_parts = bool(ov_row.get('has_parts'))
                        if pd.notna(ov_row.get('height')):
                            ov_height = float(ov_row['height'])
                        if pd.notna(ov_row.get('num_floors')):
                            ov_floors = int(ov_row['num_floors'])
                        if pd.notna(ov_row.get('roof_shape')):
                            ov_roof_shape = str(ov_row['roof_shape'])
                        if pd.notna(ov_row.get('roof_height')):
                            ov_roof_height = float(ov_row['roof_height'])

            # 2. If matched building has parts, store each part separately
            parts = overture_parts_by_building.get(matched_ov_id, []) if matched_ov_id and matched_ov_has_parts else []
            if parts:
                stored_any_part = False
                for part in parts:
                    part_geom = part.geometry
                    if part_geom is None or part_geom.is_empty:
                        continue
                    # Determine part height
                    p_height = None
                    p_source = None
                    if pd.notna(part.get('height')) and float(part['height']) > 0:
                        p_height = float(part['height'])
                        p_source = 'overture_part'
                    elif pd.notna(part.get('num_floors')) and int(part['num_floors']) > 0:
                        p_height = int(part['num_floors']) * 3.0
                        p_source = 'overture_part_floors'
                    if p_height is None:
                        continue

                    # Skip parts outside bbox, but keep full geometry
                    if clip_poly is not None:
                        if not part_geom.intersects(clip_poly):
                            continue

                    # Transform part to UTM
                    part_geom_proj = transform_geometry(part_geom, transformer)

                    # Build properties for this part
                    part_props = dict(properties)  # inherit parent building props
                    part_props['height'] = p_height
                    part_props['height_source'] = p_source
                    # min_height for floating parts (rare but handled)
                    if pd.notna(part.get('min_height')) and float(part['min_height']) > 0:
                        part_props['min_height'] = float(part['min_height'])
                    # Store roof shape/height for 3D roof generation
                    if pd.notna(part.get('roof_shape')):
                        part_props['roof_shape'] = str(part['roof_shape'])
                    if pd.notna(part.get('roof_height')) and float(part['roof_height']) > 0:
                        part_props['roof_height'] = float(part['roof_height'])
                    part_props['area'] = float(part_geom_proj.area) if hasattr(part_geom_proj, 'area') else float(area)

                    overture_building_batch.append(
                        (part_geom_proj, part_props, 0.95,
                         str(part.get('id', f'{idx}_part'))))
                    stored_any_part = True
                if stored_any_part:
                    # Per Simple 3D Buildings spec: when parts exist,
                    # skip the parent building outline for 3D rendering.
                    # Parts replace it entirely.
                    continue

            # 3. No parts — use Overture parent height or fall through
            if ov_height is not None and ov_height > 0:
                properties['height'] = ov_height
                properties['height_source'] = 'overture'
            elif ov_floors is not None and ov_floors > 0:
                properties['height'] = ov_floors * 3.0
                properties['height_source'] = 'overture_floors'
            # 4. OSM height tags (already parsed above into properties['height'])
            elif properties.get('height') and not math.isnan(float(properties['height'])):
                properties['height_source'] = 'osm'
            # 5. OSM building:levels
            elif properties.get('levels'):
                properties['height'] = float(properties['levels']) * 3.0
                properties['height_source'] = 'estimated_from_levels'
            else:
                # 6. Estimate from building type
                btype = properties.get('building_type', 'yes').lower()
                type_heights = {
                    'apartments': 14.0, 'residential': 12.0,
                    'hotel': 20.0, 'office': 15.0,
                    'commercial': 12.0, 'retail': 5.0,
                    'warehouse': 8.0, 'industrial': 8.0,
                    'house': 7.0, 'detached': 7.0,
                    'terrace': 8.0, 'semidetached_house': 7.0,
                    'garage': 3.0, 'garages': 3.0,
                    'shed': 3.0, 'roof': 4.0,
                    'cabin': 4.0, 'hut': 3.0,
                    'church': 15.0, 'cathedral': 30.0,
                    'school': 10.0, 'university': 12.0,
                    'hospital': 15.0, 'train_station': 10.0,
                }
                if btype in type_heights:
                    properties['height'] = type_heights[btype]
                    properties['height_source'] = 'estimated_from_type'
                else:
                    # 7. Estimate from footprint area
                    a = properties.get('area', 100)
                    if a < 80:
                        properties['height'] = 6.0
                    elif a < 300:
                        properties['height'] = 10.0
                    elif a < 1000:
                        properties['height'] = 14.0
                    else:
                        properties['height'] = 18.0
                    properties['height_source'] = 'estimated_from_area'

            # 8. Final fallback — one story (3m wall + 3m roof = 6m)
            # per OSMBuilding default
            if 'height' not in properties:
                properties['height'] = 6.0
                properties['height_source'] = 'default'

            # Store roof shape/height from Overture for 3D roof generation
            if ov_roof_shape:
                properties['roof_shape'] = ov_roof_shape
            if ov_roof_height and ov_roof_height > 0:
                properties['roof_height'] = ov_roof_height

            # Store the feature with error handling
            confidence = 0.9 if properties.get('height_source') in ('overture', 'osm') else 0.5
            osm_building_batch.append(
                (geom_proj, properties, confidence, str(idx)))

        except Exception as e:
            logger.error(f"Error processing building {idx}: {e}")
            continue

    # Bulk-insert all buildings in one or two transactions
    if osm_building_batch:
        stored = db.store_features_batch(city_id, 'building',
                                         osm_building_batch, source='osm')
        logger.info(f"  Stored {stored} OSM buildings (batch)")
    if overture_building_batch:
        stored = db.store_features_batch(city_id, 'building',
                                         overture_building_batch,
                                         source='overture')
        logger.info(f"  Stored {stored} Overture building parts (batch)")
    if remainder_stored:
        logger.info(f"  Stored {remainder_stored} parent remainder footprints (partial part coverage)")

    # ── Overture gap-fill: add buildings not covered by OSM ───────────
    if overture_all is not None and len(overture_all) > 0:
        _progress(50, "Adding Overture gap-fill buildings...")
        logger.info(f"Overture gap-fill: {len(matched_overture_indices)} / "
                     f"{len(overture_all)} matched OSM, checking remainder...")
        overture_gap_batch = []
        for oi in range(len(overture_all)):
            if oi in matched_overture_indices:
                continue
            ov_row = overture_all.iloc[oi]
            ov_geom = ov_row.geometry
            if ov_geom is None or ov_geom.is_empty or not ov_geom.is_valid:
                continue

            try:
                # Skip buildings outside bbox, but keep full geometry
                if clip_poly is not None:
                    if not ov_geom.intersects(clip_poly):
                        continue

                # Transform to UTM
                geom_proj = transform_geometry(ov_geom, transformer)
                if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                    continue
                area = geom_proj.area
                if area < 5.0:  # skip tiny fragments
                    continue

                # Build properties
                props = {}
                props['area'] = float(area)

                # Building type from Overture class/subtype
                ov_class = ov_row.get('class') if pd.notna(ov_row.get('class')) else None
                ov_subtype = ov_row.get('subtype') if pd.notna(ov_row.get('subtype')) else None
                props['building_type'] = str(ov_subtype or ov_class or 'yes')

                # Roof shape from Overture
                if pd.notna(ov_row.get('roof_shape')):
                    props['roof_shape'] = str(ov_row['roof_shape'])
                if pd.notna(ov_row.get('roof_height')):
                    rh = float(ov_row['roof_height'])
                    if rh > 0:
                        props['roof_height'] = rh

                # Height cascade
                confidence = 0.5
                if pd.notna(ov_row.get('height')) and float(ov_row['height']) > 0:
                    props['height'] = float(ov_row['height'])
                    props['height_source'] = 'overture'
                    confidence = 0.85
                elif pd.notna(ov_row.get('num_floors')) and int(ov_row['num_floors']) > 0:
                    props['height'] = int(ov_row['num_floors']) * 3.0
                    props['height_source'] = 'overture_floors'
                    confidence = 0.75
                else:
                    # Estimate from footprint area
                    if area < 80:
                        props['height'] = 6.0
                    elif area < 300:
                        props['height'] = 10.0
                    elif area < 1000:
                        props['height'] = 14.0
                    else:
                        props['height'] = 18.0
                    props['height_source'] = 'estimated_from_area'

                ov_id = ov_row.get('id', f'overture_{oi}')
                overture_gap_batch.append(
                    (geom_proj, props, confidence, str(ov_id)))

            except Exception as e:
                logger.debug(f"Error processing Overture gap-fill building {oi}: {e}")
                continue

        if overture_gap_batch:
            stored = db.store_features_batch(city_id, 'building',
                                             overture_gap_batch,
                                             source='overture')
            logger.info(f"  Stored {stored} Overture gap-fill buildings")
        else:
            logger.info("  No Overture gap-fill buildings to add")
    # ── End Overture gap-fill ─────────────────────────────────────────

    # Process water features
    _progress(52, "Creating water and coastline...")
    logger.info("Processing water features...")
    for idx, row in tqdm(water.iterrows(), total=len(water), desc="Water features"):
        if row.geometry and row.geometry.is_valid and not row.geometry.is_empty:
            try:
                # Clip water to bounding box BEFORE projection so large
                # bodies like Lake Union don't extend outside the area.
                raw_geom = row.geometry
                if clip_poly is not None:
                    clipped = raw_geom.intersection(clip_poly)
                    if clipped.is_empty:
                        continue
                    # intersection() can return a GeometryCollection
                    # (mix of polygons, lines, points).  Keep only the
                    # polygonal parts for water surfaces.
                    if isinstance(clipped, GeometryCollection):
                        polys = [
                            g for g in clipped.geoms
                            if isinstance(g, (Polygon, MultiPolygon))
                        ]
                        if not polys:
                            continue
                        raw_geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
                    else:
                        raw_geom = clipped
                geom_proj = transform_geometry(raw_geom, transformer)
                if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                    continue

                # Polygon water → check area; LineString water → check length
                if geom_proj.geom_type in ('Polygon', 'MultiPolygon'):
                    measure = geom_proj.area
                else:
                    measure = geom_proj.length
                if not measure or np.isnan(measure) or measure <= 0:
                    continue

                water_type = (row.get('water') or row.get('waterway')
                              or row.get('natural') or 'unknown')

                db.store_features(
                    city_id,
                    'water',
                    geom_proj,
                    {'type': str(water_type), 'area': float(geom_proj.area)},
                    source='osm',
                    confidence=1.0,
                    osm_id=str(idx)
                )
            except Exception as e:
                logger.error(f"Error processing water feature {idx}: {e}")
                continue

    # Process glacier features
    if glaciers is not None and len(glaciers) > 0:
        logger.info(f"Processing {len(glaciers)} glacier features...")
        glacier_batch = []
        for idx, row in tqdm(glaciers.iterrows(), total=len(glaciers), desc="Glaciers"):
            if row.geometry and row.geometry.is_valid and not row.geometry.is_empty:
                try:
                    raw_geom = row.geometry
                    if clip_poly is not None:
                        clipped = raw_geom.intersection(clip_poly)
                        if clipped.is_empty:
                            continue
                        if isinstance(clipped, GeometryCollection):
                            polys = [
                                g for g in clipped.geoms
                                if isinstance(g, (Polygon, MultiPolygon))
                            ]
                            if not polys:
                                continue
                            raw_geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
                        else:
                            raw_geom = clipped
                    geom_proj = transform_geometry(raw_geom, transformer)
                    if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                        continue
                    if geom_proj.geom_type not in ('Polygon', 'MultiPolygon'):
                        continue
                    if geom_proj.area <= 0:
                        continue
                    glacier_batch.append((
                        geom_proj,
                        {'type': 'glacier', 'area': float(geom_proj.area)},
                        1.0,
                        str(idx),
                    ))
                except Exception as e:
                    logger.error(f"Error processing glacier feature {idx}: {e}")
                    continue
        if glacier_batch:
            stored = db.store_features_batch(city_id, 'glacier', glacier_batch,
                                             source='osm')
            logger.info(f"  Stored {stored} glacier features (batch)")

    # ── Construct ocean polygon from OSM coastline ─────────────────
    # OSM coastline convention: ways are directed so that water is on
    # the RIGHT side when following the way direction.
    # 1. Combine coastline linestrings + bbox boundary
    # 2. Polygonize into regions
    # 3. Classify each region using the coastline direction convention
    if clip_poly is not None and coastline is not None and len(coastline) > 0:
        try:
            # Collect OSM coastline linestrings
            coastline_lines = []
            for _, row in coastline.iterrows():
                g = row.geometry
                if isinstance(g, LineString):
                    coastline_lines.append(g)
                elif isinstance(g, MultiLineString):
                    coastline_lines.extend(g.geoms)

            if coastline_lines:
                # Combine coastline + bbox boundary, node at intersections
                merged = linemerge(coastline_lines)
                all_lines = [clip_poly.boundary]
                if isinstance(merged, LineString):
                    all_lines.append(merged)
                elif isinstance(merged, MultiLineString):
                    all_lines.extend(merged.geoms)
                noded = unary_union(all_lines)

                # Polygonize the noded line network
                result_polys = list(polygonize(noded))
                logger.info(f"  Coastline split bbox into "
                            f"{len(result_polys)} polygons")

                # Classify each polygon: water is on the RIGHT side of
                # the coastline direction.  For each polygon, find the
                # nearest coastline segment and check which side its
                # interior point falls on via cross product.
                ocean_polys = []
                for poly in result_polys:
                    rep = poly.representative_point()
                    # Find nearest original coastline line
                    min_dist = float('inf')
                    nearest = None
                    for line in coastline_lines:
                        d = line.distance(rep)
                        if d < min_dist:
                            min_dist = d
                            nearest = line
                    if nearest is None:
                        continue
                    # Get direction at projection point
                    proj_dist = nearest.project(rep)
                    eps = max(nearest.length * 1e-6, 1e-10)
                    p1 = nearest.interpolate(max(proj_dist - eps, 0))
                    p2 = nearest.interpolate(
                        min(proj_dist + eps, nearest.length))
                    # Cross product: negative = right side = water
                    cross = ((p2.x - p1.x) * (rep.y - p1.y)
                             - (p2.y - p1.y) * (rep.x - p1.x))
                    if cross < 0:
                        ocean_polys.append(poly)

                if ocean_polys:
                    ocean = unary_union(ocean_polys)
                    geom_proj = transform_geometry(ocean, transformer)
                    if geom_proj and geom_proj.is_valid:
                        water_frac = ocean.area / clip_poly.area
                        db.store_features(
                            city_id, 'water', geom_proj,
                            {'type': 'ocean', 'source': 'osm_coastline'},
                            source='osm_coastline', confidence=0.95,
                            osm_id='ocean-osm')
                        logger.info(
                            f"Ocean from OSM coastline: "
                            f"{ocean.area:.6f} sq deg "
                            f"({water_frac:.0%} of bbox)")
                else:
                    logger.info("  No ocean polygons found after "
                                "classification")
        except Exception as e:
            logger.warning(f"Ocean construction failed: {e}")

    # Process roads
    _progress(56, "Laying out roads...")
    logger.info("Processing roads...")
    road_edges = list(roads.edges(data=True))
    road_batch = []
    orig_verts = 0
    simp_verts = 0
    for u, v, data in tqdm(road_edges, desc="Roads"):
        try:
            if 'geometry' in data:
                geom = data['geometry']
            else:
                u_data = roads.nodes[u]
                v_data = roads.nodes[v]
                geom = LineString([
                    (u_data['x'], u_data['y']),
                    (v_data['x'], v_data['y'])
                ])

            geom_proj = transform_geometry(geom, transformer)
            if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                continue

            # Simplify: remove redundant vertices on near-straight
            # segments (0.5 m tolerance in projected coords).
            if geom_proj.geom_type == 'LineString':
                orig_verts += len(geom_proj.coords)
            simplified = geom_proj.simplify(0.5, preserve_topology=True)
            if simplified.is_empty:
                continue
            if simplified.geom_type == 'LineString':
                simp_verts += len(simplified.coords)

            length = simplified.length
            if not length or np.isnan(length) or length <= 0:
                continue

            road_batch.append((
                simplified,
                {
                    'type': data.get('highway', 'unknown'),
                    'name': data.get('name'),
                    'length': float(length),
                    'lanes': data.get('lanes'),
                    'surface': data.get('surface'),
                    'bridge': data.get('bridge'),
                    'tunnel': data.get('tunnel'),
                },
                1.0,
                f"{u}-{v}"
            ))
        except Exception as e:
            logger.error(f"Error processing road {u}-{v}: {e}")
            continue

    # Batch-store all roads in one transaction
    stored = db.store_features_batch(city_id, 'road', road_batch)
    if orig_verts > 0:
        logger.info(f"  Roads simplified: {orig_verts} -> {simp_verts} vertices "
                    f"({(1 - simp_verts / orig_verts) * 100:.0f}% reduction), "
                    f"{stored} stored")

    # ── Helper: transform + validate + clip a geometry ──
    def _prep_geom(row, require_area=True):
        """Transform, validate, and optionally clip a feature geometry.
        Returns (geom_proj, osm_id) or None."""
        if not row.geometry or not row.geometry.is_valid or row.geometry.is_empty:
            return None
        raw_geom = row.geometry
        if clip_poly is not None:
            clipped = raw_geom.intersection(clip_poly)
            if clipped.is_empty:
                return None
            if isinstance(clipped, GeometryCollection):
                polys = [g for g in clipped.geoms
                         if isinstance(g, (Polygon, MultiPolygon))]
                if not polys:
                    return None
                raw_geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
            else:
                raw_geom = clipped
        geom_proj = transform_geometry(raw_geom, transformer)
        if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
            return None
        if require_area and geom_proj.geom_type in ('Polygon', 'MultiPolygon'):
            area = geom_proj.area
            if not area or np.isnan(area) or area <= 0:
                return None
        return geom_proj

    # ── Batch-process all remaining feature types ──
    # Each tuple: (dataframe, feature_type, label, props_fn, clip, require_area)
    _batch_specs = []

    # Parking areas as paved surfaces
    if parking is not None and len(parking) > 0:
        _batch_specs.append((parking, 'paved', 'Parking', False,
            lambda row: {'type': 'parking',
                         'name': str(row.get('name', '')),
                         'surface': str(row.get('surface', 'asphalt'))}))

    # Developed land-use areas
    if developed is not None and len(developed) > 0:
        _batch_specs.append((developed, 'paved', 'Developed', False,
            lambda row: {'type': str(row.get('landuse', 'commercial')),
                         'name': str(row.get('name', ''))}))

    # Pier features
    if piers is not None and len(piers) > 0:
        _batch_specs.append((piers, 'pier', 'Piers', True,
            lambda row: {'type': 'pier',
                         'name': str(row.get('name', ''))}))

    # Sports pitches, tracks, pools, parks
    for feat_df, feat_type, feat_label in [
        (pitches, 'pitch', 'Pitches'),
        (tracks, 'pitch', 'Tracks'),
        (pools, 'pool', 'Pools'),
        (parks, 'park', 'Parks'),
    ]:
        if feat_df is not None and len(feat_df) > 0:
            _batch_specs.append((feat_df, feat_type, feat_label, True,
                lambda row, _fl=feat_label: {
                    'type': _fl.lower().rstrip('s'),
                    'sport': str(row.get('sport', '')),
                    'name': str(row.get('name', ''))}))

    # Railway features
    if railways is not None and len(railways) > 0:
        _batch_specs.append((railways, 'railway', 'Railways', True,
            lambda row: {'type': str(row.get('railway', 'rail')),
                         'name': str(row.get('name', ''))}))

    # Bridge deck polygons
    if bridge_decks is not None and len(bridge_decks) > 0:
        _batch_specs.append((bridge_decks, 'bridge', 'Bridges', True,
            lambda row: {'type': 'bridge_deck',
                         'name': str(row.get('name', ''))}))

    for feat_df, feat_type, feat_label, needs_clip, props_fn in _batch_specs:
        logger.info(f"Processing {feat_label.lower()}...")
        batch = []
        for idx, row in feat_df.iterrows():
            try:
                if needs_clip:
                    geom_proj = _prep_geom(row, require_area=True)
                else:
                    # Simple path: no clipping needed (parking/developed)
                    if not row.geometry or not row.geometry.is_valid or row.geometry.is_empty:
                        continue
                    geom_proj = transform_geometry(row.geometry, transformer)
                    if not geom_proj or not geom_proj.is_valid or geom_proj.is_empty:
                        continue
                    area = geom_proj.area
                    if not area or np.isnan(area) or area <= 0:
                        continue
                if geom_proj is None:
                    continue
                batch.append((geom_proj, props_fn(row), 1.0, str(idx)))
            except Exception as e:
                logger.error(f"Error processing {feat_label.lower()} "
                             f"feature {idx}: {e}")
        if batch:
            stored = db.store_features_batch(
                city_id, feat_type, batch, source='osm')
            logger.info(f"Stored {stored} {feat_label.lower()}")

    # Process tower features (man_made=tower) as buildings
    # These capture bridge pylons and other towers not tagged as building=*.
    _process_structural_features(
        db, city_id, towers, transformer, clip_poly, 'tower')

    # Process bridge support features (bridge:support=pylon/pier/abutment)
    _process_structural_features(
        db, city_id, bridge_supports, transformer, clip_poly, 'bridge_support')
