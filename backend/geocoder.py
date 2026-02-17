import logging
import math
import re
import sqlite3

from geopy.geocoders import Nominatim, Photon
from geopy.extra.rate_limiter import RateLimiter

from backend import config

logger = logging.getLogger(__name__)


class GeocoderService:
    """Geocoding service that checks the local DB first, then falls back to
    Nominatim.  Replaces the OpenAI-backed AIHelper for location resolution."""

    def __init__(self):
        self._nominatim = Nominatim(
            user_agent=config.NOMINATIM_USER_AGENT,
            timeout=config.NOMINATIM_TIMEOUT,
        )
        self._geocode_nominatim = RateLimiter(
            self._nominatim.geocode,
            min_delay_seconds=1.0,
        )
        self._photon = Photon(
            user_agent=config.NOMINATIM_USER_AGENT,
            timeout=config.NOMINATIM_TIMEOUT,
        )
        self._geocode_photon = RateLimiter(
            self._photon.geocode,
            min_delay_seconds=0.5,
        )

    # ------------------------------------------------------------------
    # Internal DB helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(config.DB_PATH), timeout=20)

    def _check_landmark(self, location_name: str) -> dict | None:
        """Mirror of AIHelper lines 849-867 -- look up landmarks table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT center_lat, center_lon, bbox_radius, description
                FROM landmarks
                WHERE LOWER(name) = LOWER(?)
                """,
                (location_name,),
            )
            result = cursor.fetchone()
            if result is None:
                return None

            center_lat, center_lon, radius, description = result
            logger.info(f"Found landmark building {location_name}: {description}")

            lat_radius = radius / 111320.0
            lon_radius = radius / (111320.0 * math.cos(math.radians(center_lat)))

            return {
                "north": center_lat + lat_radius,
                "south": center_lat - lat_radius,
                "east": center_lon + lon_radius,
                "west": center_lon - lon_radius,
                "display_name": location_name,
            }

    def _check_neighborhood(self, location_name: str) -> dict | None:
        """Mirror of AIHelper lines 870-884 -- look up neighborhoods table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT bbox_north, bbox_south, bbox_east, bbox_west, description
                FROM neighborhoods
                WHERE LOWER(name) = LOWER(?)
                """,
                (location_name,),
            )
            result = cursor.fetchone()
            if result is None:
                return None

            logger.info(f"Found neighborhood {location_name}: {result[4]}")
            return {
                "north": result[0],
                "south": result[1],
                "east": result[2],
                "west": result[3],
                "display_name": location_name,
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bbox(self, location_name: str) -> dict:
        """Resolve *location_name* to a bounding-box dict.

        Lookup order:
        1. landmarks table in the local SQLite DB
        2. neighborhoods table in the local SQLite DB
        3. Nominatim geocoding (free, no API key)

        Returns ``{"north", "south", "east", "west", "display_name"}``.
        Raises ``ValueError`` when the location cannot be resolved.
        """

        # 1. landmarks
        result = self._check_landmark(location_name)
        if result is not None:
            return result

        # 2. neighborhoods
        result = self._check_neighborhood(location_name)
        if result is not None:
            return result

        # 3. Photon → Nominatim fallback chain
        result = self._try_photon(location_name)
        if result is not None:
            return result

        result = self._try_nominatim(location_name)
        if result is not None:
            return result

        raise ValueError(
            f"Could not geocode location: {location_name}"
        )

    # ------------------------------------------------------------------
    # Geocoder backends
    # ------------------------------------------------------------------

    def _expand_point(self, lat: float, lon: float, display_name: str,
                      min_span: float = 0.005) -> dict:
        """Expand a point result into a usable bounding box (~500m)."""
        return {
            "north": lat + min_span / 2,
            "south": lat - min_span / 2,
            "east": lon + min_span / 2,
            "west": lon - min_span / 2,
            "display_name": display_name,
        }

    def _ensure_min_span(self, north, south, east, west, display_name,
                         min_span=0.005) -> dict:
        """Ensure bbox has a minimum span (~500m)."""
        lat_span = north - south
        lon_span = east - west
        if lat_span < min_span:
            center_lat = (north + south) / 2
            north = center_lat + min_span / 2
            south = center_lat - min_span / 2
        if lon_span < min_span:
            center_lon = (east + west) / 2
            east = center_lon + min_span / 2
            west = center_lon - min_span / 2
        return {
            "north": north, "south": south,
            "east": east, "west": west,
            "display_name": display_name,
        }

    def _try_nominatim(self, location_name: str) -> dict | None:
        """Try Nominatim geocoder. Returns bbox dict or None."""
        try:
            logger.info(f"Trying Nominatim for '{location_name}'...")
            geo_result = self._geocode_nominatim(
                location_name, exactly_one=True)

            if geo_result is None:
                # Retry with stripped building-type prefixes
                stripped = re.sub(
                    r'^(Cathedral|Church|Basilica|Tower|Palace|Castle|Temple|'
                    r'Mosque|Shrine|Museum|Statue|Monument|Fort|Fortress|'
                    r'Bridge|Library|Stadium|Arena|Theatre|Theater|Abbey|'
                    r'Chapel|Monastery|Citadel|Tomb|Mausoleum)\s+(of\s+)?',
                    '', location_name, count=1, flags=re.IGNORECASE,
                )
                if stripped != location_name:
                    logger.info(f"Retrying Nominatim with: '{stripped}'")
                    geo_result = self._geocode_nominatim(
                        stripped, exactly_one=True)

            if geo_result is None:
                logger.info("Nominatim returned no results")
                return None

            raw_bbox = geo_result.raw.get("boundingbox")
            if raw_bbox is None:
                return self._expand_point(
                    geo_result.latitude, geo_result.longitude,
                    geo_result.address)

            south, north, west, east = (float(v) for v in raw_bbox)
            return self._ensure_min_span(
                north, south, east, west, geo_result.address)

        except Exception as e:
            logger.warning(f"Nominatim failed: {e}")
            return None

    def _try_photon(self, location_name: str) -> dict | None:
        """Try Photon (Komoot) geocoder as fallback. Returns bbox dict or None."""
        try:
            logger.info(f"Trying Photon for '{location_name}'...")
            geo_result = self._geocode_photon(
                location_name, exactly_one=True)

            if geo_result is None:
                logger.info("Photon returned no results")
                return None

            display_name = geo_result.address or location_name

            # Photon returns extent as [west, south, east, north] in properties
            props = geo_result.raw.get("properties", {})
            extent = props.get("extent")
            if extent and len(extent) == 4:
                west, south, east, north = (float(v) for v in extent)
                logger.info(f"Photon resolved '{location_name}' with extent")
                return self._ensure_min_span(
                    north, south, east, west, display_name)

            # No extent — use point coordinates
            logger.info(f"Photon resolved '{location_name}' as point, "
                        f"expanding to default bbox")
            return self._expand_point(
                geo_result.latitude, geo_result.longitude, display_name)

        except Exception as e:
            logger.warning(f"Photon failed: {e}")
            return None
