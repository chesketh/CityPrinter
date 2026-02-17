"""CityBuilder — thin orchestrator that delegates to focused modules."""

import math
import logging
import pathlib
import pickle
import shutil
import time
from typing import Union

from pyproj import Transformer
from shapely import wkb as wkb_mod
from shapely.geometry import box as shapely_box

from .database import CityDatabase
from .ai_helper import AIHelper
from .models import BoundingBox
from . import osm_data
from . import glb as glb_mod
from . import ply as ply_mod

logger = logging.getLogger(__name__)

# ── Caches ─────────────────────────────────────────────────────────────
_PROCESSED_CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "processed"
_GLB_CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "glb"


_CACHE_VERSION = 3  # bump when feature processing changes

def _cache_key(bbox):
    """Filename for a processed cache entry."""
    return (f"features_v{_CACHE_VERSION}_{bbox.south:.4f}_{bbox.north:.4f}"
            f"_{bbox.west:.4f}_{bbox.east:.4f}.pkl")


def _find_containing_cache(bbox, epsg):
    """Find a cached build whose bbox fully contains the requested bbox."""
    _PROCESSED_CACHE.mkdir(parents=True, exist_ok=True)
    for path in _PROCESSED_CACHE.glob("*.pkl"):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            continue
        if data.get("epsg") != epsg:
            continue
        cb = data["bbox"]
        if (cb["south"] <= bbox.south + 1e-6 and
                cb["north"] >= bbox.north - 1e-6 and
                cb["west"] <= bbox.west + 1e-6 and
                cb["east"] >= bbox.east - 1e-6):
            logger.info(f"Processed-feature cache hit: {path.name} "
                        f"({len(data['features'])} features)")
            return data
    return None


def _bbox_matches(cached_bbox, bbox):
    """Check if bboxes are effectively identical."""
    return (abs(cached_bbox["south"] - bbox.south) < 1e-4 and
            abs(cached_bbox["north"] - bbox.north) < 1e-4 and
            abs(cached_bbox["west"] - bbox.west) < 1e-4 and
            abs(cached_bbox["east"] - bbox.east) < 1e-4)


def _filter_features_to_bbox(features, bbox, transformer):
    """Filter cached features to a sub-bbox. Features are in UTM coords."""
    x1, y1 = transformer.transform(bbox.west, bbox.south)
    x2, y2 = transformer.transform(bbox.east, bbox.north)
    clip_box = shapely_box(min(x1, x2), min(y1, y2),
                           max(x1, x2), max(y1, y2))

    result = []
    for row in features:
        ft, osm_id, geom_wkb, props, src, conf = row
        geom = wkb_mod.loads(geom_wkb)
        if clip_box.intersects(geom):
            result.append(row)
    logger.info(f"Filtered {len(features)} cached features → "
                f"{len(result)} in sub-bbox")
    return result


def _save_processed_cache(db, city_id, bbox, epsg):
    """Save processed features to cache after full build."""
    _PROCESSED_CACHE.mkdir(parents=True, exist_ok=True)
    features = db.get_city_features(city_id)
    data = {
        "bbox": {"south": bbox.south, "north": bbox.north,
                 "west": bbox.west, "east": bbox.east},
        "epsg": epsg,
        "features": features,
    }
    path = _PROCESSED_CACHE / _cache_key(bbox)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(f"Saved {len(features)} processed features to cache "
                f"({size_mb:.1f} MB)")


class CityBuilder:
    def __init__(self, coordinate_scale=111320.0, center_lat=47.6062,
                 center_lon=-122.3321, use_cache=True):
        """
        coordinate_scale: approximate units per degree.
        center_lat, center_lon: the 'anchor' latitude and longitude.
        use_cache: set False to bypass the processed-feature cache.
        """
        self.coordinate_scale = coordinate_scale
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.db = CityDatabase()
        self.ai_helper = AIHelper(self.db)
        self.transformer = None
        self.use_cache = use_cache
        self._last_bbox = None          # set by process_city for GLB cache
        self._exact_cache_hit = False   # True when bbox matched exactly

    async def process_city(self, location: Union[str, BoundingBox],
                           progress_callback=None) -> int:
        """Process a city either by name or bounding box."""
        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        try:
            logger.info(f"Starting to process location: {location}")
            _progress(8, "Preparing workspace...")
            if isinstance(location, str):
                logger.info(f"Getting city info for: {location}")
                bbox_dict = await self.ai_helper.get_city_info(location)
                bbox = BoundingBox(**bbox_dict)
                city_name = location
            else:
                bbox = location
                city_name = f"Custom Area {bbox.north:.2f}, {bbox.west:.2f}"

            logger.info(f"Processing city: {city_name}")
            logger.info(f"Bounding box: N={bbox.north}, S={bbox.south}, "
                        f"E={bbox.east}, W={bbox.west}")

            # Store city in database
            city_id = self.db.add_city(city_name, bbox)
            logger.info(f"Added city to database with ID: {city_id}")

            # Initialize transformer for coordinate conversion.
            center_lat = (bbox.north + bbox.south) / 2
            center_lon = (bbox.east + bbox.west) / 2
            utm_zone = int((center_lon + 180) / 6) + 1
            utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
            self.transformer = Transformer.from_crs(
                "EPSG:4326",
                f"EPSG:{utm_epsg}",
                always_xy=True
            )
            logger.info(f"Using UTM zone {utm_zone} (EPSG:{utm_epsg}) "
                        f"for coordinate transform")

            self._last_bbox = bbox
            self._exact_cache_hit = False

            # ── Check processed-feature cache ──────────────────────────
            if self.use_cache:
                t0 = time.perf_counter()
                cached = _find_containing_cache(bbox, utm_epsg)
                if cached is not None:
                    self.db.clear_city_features(city_id)
                    features = cached["features"]

                    exact = _bbox_matches(cached["bbox"], bbox)
                    if not exact:
                        features = _filter_features_to_bbox(
                            features, bbox, self.transformer)

                    restored = self.db.restore_features_batch(
                        city_id, features)
                    elapsed = time.perf_counter() - t0
                    logger.info(f"Restored {restored} features from cache "
                                f"in {elapsed:.1f}s")
                    self._exact_cache_hit = exact
                    return city_id

            # ── Cache miss — full processing ───────────────────────────
            self.db.clear_city_features(city_id)
            osm_data.download_osm_data(self.db, bbox, city_id,
                                       self.transformer,
                                       progress_callback=progress_callback)

            # Save to processed-feature cache
            if self.use_cache:
                _save_processed_cache(self.db, city_id, bbox, utm_epsg)

            return city_id

        except Exception as e:
            logger.error(f"Error processing city: {e}")
            raise

    def generate_glb(self, city_id: int, output_path: str,
                     progress_callback=None) -> str:
        """Generate GLB file. Returns the absolute path to the generated file."""
        from .models import PathManager

        # ── GLB cache ─────────────────────────────────────────────
        glb_cache_path = None
        if self.use_cache and self._last_bbox is not None:
            _GLB_CACHE.mkdir(parents=True, exist_ok=True)
            glb_name = _cache_key(self._last_bbox).replace('.pkl', '.glb')
            glb_cache_path = _GLB_CACHE / glb_name
            if glb_cache_path.exists():
                resolved = PathManager.get_output_path(output_path)
                resolved.parent.mkdir(exist_ok=True)
                shutil.copy2(glb_cache_path, resolved)
                size_mb = glb_cache_path.stat().st_size / 1024 / 1024
                logger.info(f"GLB cache hit: {glb_name} ({size_mb:.1f} MB)")
                return str(resolved)

        # ── Full GLB generation ────────────────────────────────────
        result = glb_mod.generate_glb(self.db, city_id, output_path,
                                      progress_callback=progress_callback)

        # Save to GLB cache
        if glb_cache_path is not None and result:
            try:
                shutil.copy2(result, glb_cache_path)
                size_mb = glb_cache_path.stat().st_size / 1024 / 1024
                logger.info(f"Saved GLB to cache: {glb_cache_path.name} "
                            f"({size_mb:.1f} MB)")
            except Exception as e:
                logger.warning(f"Failed to cache GLB: {e}")

        return result

    def generate_ply_single(self, city_id: int, output_path: str,
                            name: str = "city", scale: float = 1.0,
                            progress_callback=None) -> dict:
        """Generate a single watertight PLY from a city build.

        All feature meshes are color-coded and merged into one file.
        """
        glb_path = self.generate_glb(city_id, output_path,
                                     progress_callback=progress_callback)
        ply_path = str(pathlib.Path(glb_path).with_suffix('.ply'))
        return ply_mod.generate_ply_single(
            glb_path=glb_path,
            output_path=ply_path,
            name=name,
            scale=scale,
            progress_callback=progress_callback,
        )

    def generate_ply(self, city_id: int, output_path: str,
                     name: str = "city", scale: float = 1.0,
                     progress_callback=None) -> dict:
        """Generate printable PLY layers from a city build.

        First generates a GLB (using cache if available), then splits it
        into separate watertight PLY files per layer with solid colors.

        Returns dict with layer info and manifest path.
        """
        # Generate GLB first (leverages existing cache)
        glb_path = self.generate_glb(city_id, output_path,
                                     progress_callback=progress_callback)

        # Split into PLY layers
        output_dir = str(pathlib.Path(glb_path).parent / f"{name}_print")
        return ply_mod.generate_ply_layers(
            glb_path=glb_path,
            output_dir=output_dir,
            name=name,
            scale=scale,
            progress_callback=progress_callback,
        )

    def generate_stl(self, city_id: int, output_path: str) -> None:
        """Generate STL file from stored city data."""
        glb_mod.generate_stl(self.db, city_id, output_path)

    def approximate_local_xy(self, lat: float, lon: float) -> tuple:
        """
        Replicate the transform from landmark_extractor.py:
        x = (lon - center_lon) * scale * cos(radians(center_lat))
        y = (lat - center_lat) * scale
        """
        scale = self.coordinate_scale
        x = (lon - self.center_lon) * scale * math.cos(math.radians(self.center_lat))
        y = (lat - self.center_lat) * scale
        return x, y

    def approximate_local_bbox(self, north, south, east, west) -> dict:
        """
        Convert a lat/lon bounding box to local XY, matching the approach above.
        Returns a dict with keys: 'west', 'east', 'south', 'north'
        """
        west_x, south_y = self.approximate_local_xy(south, west)
        east_x, north_y = self.approximate_local_xy(north, east)

        return {
            'west':  min(west_x, east_x),
            'east':  max(west_x, east_x),
            'south': min(south_y, north_y),
            'north': max(south_y, north_y)
        }
