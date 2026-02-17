"""Configuration constants, paths, and osmnx monkey-patch."""

import os
import pathlib
import logging

import osmnx as ox
import osmnx._overpass as _ox_overpass
from dotenv import load_dotenv

# -- Monkey-patch osmnx to prevent infinite Overpass retry/pause loops ----
#
# Problem 1: _get_overpass_pause() can return 28,000+ second waits or
#   loop forever when status says "Currently running a query".
#
# Problem 2: _overpass_request() has an osmnx bug where `this_pause` is
#   undefined when `pause` is not None, AND it recursively retries on
#   HTTP 429/504 with no max-retry limit.
#
# Fix: fully replace _overpass_request with a clean implementation that
#   caps pause to 5s and limits retries to 2.

import time as _patch_time
import requests as _patch_requests
from osmnx import settings as _ox_settings
from osmnx import utils as _ox_utils
from osmnx._http import (
    _config_dns,
    _hostname_from_url,
    _get_http_headers,
    _retrieve_from_cache,
    _save_to_cache,
    _parse_response,
)
from osmnx._errors import InsufficientResponseError
import logging as _patch_lg

_original_get_overpass_pause = _ox_overpass._get_overpass_pause


def _capped_get_overpass_pause(base_endpoint, **kwargs):
    """Cap Overpass pause to 5s max to prevent long/infinite waits."""
    try:
        pause = _original_get_overpass_pause(base_endpoint, **kwargs)
    except Exception:
        return 1  # fallback on any error
    if pause > 5:
        _ox_utils.log(f"Overpass wanted {pause:.0f}s pause - capping to 5s", level=_patch_lg.WARNING)
        return 5
    return pause


def _full_replacement_overpass_request(data, *, pause=None, error_pause=60, _attempt=1):
    """Full replacement for osmnx._overpass._overpass_request.

    Fixes the this_pause bug and caps retries to 2.
    """
    if _attempt > 2:
        raise RuntimeError("Overpass API failed after 2 attempts - skipping this query.")

    # resolve url to same IP
    _config_dns(_ox_settings.overpass_url)

    # prepare URL and check cache
    url = _ox_settings.overpass_url.rstrip("/") + "/interpreter"
    prepared_url = str(_patch_requests.Request("GET", url, params=data).prepare().url)
    cached = _retrieve_from_cache(prepared_url)
    if isinstance(cached, dict):
        return cached

    # determine pause (capped to 5s max)
    if pause is None:
        this_pause = _capped_get_overpass_pause(_ox_settings.overpass_url)
    else:
        this_pause = min(pause, 5)

    hostname = _hostname_from_url(url)
    _ox_utils.log(f"Pausing {this_pause} second(s) before making HTTP POST request to {hostname!r}", level=_patch_lg.INFO)
    _patch_time.sleep(this_pause)

    # transmit the HTTP POST request
    _ox_utils.log(f"Post {prepared_url} with timeout={_ox_settings.requests_timeout}", level=_patch_lg.INFO)
    response = _patch_requests.post(
        url,
        data=data,
        timeout=_ox_settings.requests_timeout,
        headers=_get_http_headers(),
        **_ox_settings.requests_kwargs,
    )

    # handle 429 and 504 errors with retry
    if response.status_code in {429, 504}:
        retry_pause = min(error_pause, 10)
        _ox_utils.log(
            f"{hostname!r} responded {response.status_code} {response.reason}: retrying in {retry_pause}s (attempt {_attempt}/2)",
            level=_patch_lg.WARNING,
        )
        _patch_time.sleep(retry_pause)
        return _full_replacement_overpass_request(data, pause=pause, error_pause=error_pause, _attempt=_attempt + 1)

    response_json = _parse_response(response)
    if not isinstance(response_json, dict):
        raise InsufficientResponseError("Overpass API did not return a dict of results.")
    _save_to_cache(prepared_url, response_json, response.ok)
    return response_json


_ox_overpass._get_overpass_pause = _capped_get_overpass_pause
_ox_overpass._overpass_request = _full_replacement_overpass_request
# -- End monkey-patch ----------------------------------------------------

# ── OSM tag overrides for specific features ─────────────────────────────
# Corrects missing or incorrect OSM tags on known features.
# Key: (element_type, osm_id).  Value: dict of tag overrides.
# roof:shape / roof:height only fill missing tags; height always replaces.
OSM_TAG_OVERRIDES = {
    # Sagrada Familia central towers — realistic tapered spire shapes.
    #
    # Virgin Mary (18m base): full taper from ground to 138m peak.
    ('way', 1207779418): {'roof:shape': 'pyramidal', 'roof:height': '138', 'height': '138'},
    #
    # Jesus tower: taper from 3rd stacked cylinder (110m level, 20m base)
    # to 172m peak.  The spire is placed on the 110m tier; the two
    # cylinders above it (120m, 130m) are shrunk so they disappear
    # inside the wider 110m tier.
    ('way', 359084993):  {'roof:shape': 'pyramidal', 'roof:height': '62', 'height': '172'},  # 110m tier → spire
    ('way', 359084990):  {'height': '1'},   # 120m 19m-wide cylinder — hide
    ('way', 359084991):  {'height': '1'},   # 130m 18m-wide cylinder — hide
}
# ── End OSM tag overrides ────────────────────────────────────────────────

# Feature categorization and styling
FEATURE_CATEGORIES = {
    'buildings': {
        'types': {
            'landmark': ['landmark', 'tower', 'cathedral', 'castle', 'monument', 'bunker', 'bridge', 'historic'],
            'commercial': ['commercial', 'retail', 'shop', 'store', 'mall', 'supermarket', 'office'],
            'residential': ['house', 'residential', 'apartments', 'dormitory', 'bungalow'],
            'industrial': ['industrial', 'warehouse', 'factory', 'manufacturing'],
            'civic': ['government', 'hospital', 'school', 'university', 'library', 'police', 'fire_station'],
            'cultural': ['museum', 'theatre', 'cinema', 'arts_centre', 'concert_hall'],
            'religious': ['church', 'mosque', 'temple', 'synagogue', 'religious'],
            'transportation': ['train_station', 'bus_station', 'airport', 'terminal', 'hangar'],
            'parking': ['parking', 'garage', 'carport'],
            'other': ['yes', 'building']  # Default category
        },
        'style': {
            'wall_thickness': 2.0,  # meters
            'min_height': 10.0,     # meters
            'colors': {
                'landmark': [0.95, 0.95, 0.95],     # Almost white
                'commercial': [0.90, 0.90, 0.90],   # Light grey
                'residential': [0.85, 0.85, 0.85],  # Grey
                'industrial': [0.80, 0.80, 0.80],   # Darker grey
                'civic': [0.92, 0.92, 0.92],        # Very light grey
                'cultural': [0.88, 0.88, 0.88],     # Medium light grey
                'religious': [0.93, 0.93, 0.93],    # Very light grey
                'transportation': [0.82, 0.82, 0.82],# Medium dark grey
                'parking': [0.75, 0.75, 0.75],      # Dark grey
                'other': [0.87, 0.87, 0.87]         # Default grey
            }
        }
    },
    'water': {
        'types': {
            'ocean': ['sea', 'ocean'],
            'river': ['river', 'riverbank', 'stream', 'canal'],
            'lake': ['lake', 'reservoir', 'water'],
            'fountain': ['fountain', 'water_feature'],
            'pool': ['swimming_pool', 'pool']
        },
        'style': {
            'depth': 1.0,  # meters
            'colors': {
                'ocean': [0.0, 0.0, 0.4],      # Dark blue
                'river': [0.0, 0.0, 0.5],      # Medium dark blue
                'lake': [0.0, 0.1, 0.6],       # Medium blue
                'fountain': [0.1, 0.2, 0.7],   # Light blue
                'pool': [0.2, 0.3, 0.8]        # Light bright blue
            }
        }
    },
    'green': {
        'types': {
            'park': ['park', 'garden', 'playground'],
            'forest': ['forest', 'wood', 'natural'],
            'grass': ['grass', 'grassland', 'meadow'],
            'sport': ['pitch', 'golf_course', 'sports_centre'],
            'cemetery': ['cemetery', 'grave_yard'],
            'agriculture': ['farmland', 'orchard', 'vineyard']
        },
        'style': {
            'height': 0.3,  # meters
            'colors': {
                'park': [0.0, 0.3, 0.0],       # Dark green
                'forest': [0.0, 0.2, 0.0],     # Darker green
                'grass': [0.0, 0.4, 0.0],      # Medium green
                'sport': [0.0, 0.5, 0.0],      # Light green
                'cemetery': [0.0, 0.25, 0.0],  # Dark medium green
                'agriculture': [0.1, 0.4, 0.1]  # Olive green
            }
        }
    },
    'roads': {
        'types': {
            'highway': ['motorway', 'trunk', 'primary',
                        'motorway_link', 'trunk_link', 'primary_link'],
            'main_road': ['secondary', 'tertiary',
                          'secondary_link', 'tertiary_link'],
            'local_road': ['residential', 'service', 'unclassified',
                           'living_street', 'road', 'bus_guideway',
                           'construction', 'turning_circle', 'pedestrian'],
            'path': ['footway', 'cycleway', 'path', 'track',
                     'steps', 'bridleway', 'corridor'],
            'parking': ['parking', 'parking_aisle']
        },
        'style': {
            'widths': {
                'highway': 16.0,      # meters
                'main_road': 12.0,    # meters
                'local_road': 8.0,    # meters
                'path': 2.0,          # meters
                'parking': 6.0        # meters
            },
            'heights': {
                'highway': 0.5,       # meters
                'main_road': 0.3,     # meters
                'local_road': 0.2,    # meters
                'path': 0.1,          # meters
                'parking': 0.15       # meters
            },
            'colors': {
                'highway': [0.2, 0.2, 0.2],      # Very dark grey
                'main_road': [0.25, 0.25, 0.25], # Dark grey
                'local_road': [0.3, 0.3, 0.3],   # Medium dark grey
                'path': [0.4, 0.4, 0.4],         # Medium grey
                'parking': [0.35, 0.35, 0.35]    # Medium dark grey
            }
        }
    }
}

# Load environment variables
load_dotenv()

# Configure base paths
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
