"""Köppen-Geiger climate classification lookup for tree type selection.

Uses Beck et al. (2023) 1km global raster — single pixel lookup per build
at bbox center.  Downloads ~125MB zip on first use, caches permanently.
"""

import logging
import pathlib
import zipfile
from enum import Enum

import rasterio

logger = logging.getLogger(__name__)

# ── Paths & URLs ──────────────────────────────────────────────────────

_KOPPEN_CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache" / "koppen"
_KOPPEN_URL = "https://ndownloader.figshare.com/files/61012822"  # V3 zip
_KOPPEN_TIF = "koppen_geiger_0p00833333.tif"
_KOPPEN_ZIP_MEMBER = "1991_2020/koppen_geiger_0p00833333.tif"

# ── Integer code → string mapping (Beck 2023) ────────────────────────

KOPPEN_CODES = {
    1: "Af",  2: "Am",  3: "Aw",
    4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dfa", 18: "Dfb", 19: "Dfc", 20: "Dfd",
    21: "Dsa", 22: "Dsb", 23: "Dsc", 24: "Dsd",
    25: "Dwa", 26: "Dwb", 27: "Dwc", 28: "Dwd",
    29: "ET",  30: "EF",
}


# ── Tree types ────────────────────────────────────────────────────────

class TreeType(str, Enum):
    CONIFER = "conifer"
    DECIDUOUS = "deciduous"
    BROADLEAF_TROPICAL = "broadleaf_tropical"
    PALM = "palm"
    MANGROVE = "mangrove"
    SCRUB = "scrub"
    SCLEROPHYLL = "sclerophyll"


# ── Climate table ─────────────────────────────────────────────────────
# Each entry: {TreeType: weight}  — weights sum to 1.0.
# WorldCover already determines WHERE and HOW MANY trees exist;
# this table only controls WHAT TYPE of tree goes at each position.

_C = TreeType.CONIFER
_D = TreeType.DECIDUOUS
_BT = TreeType.BROADLEAF_TROPICAL
_P = TreeType.PALM
_M = TreeType.MANGROVE
_S = TreeType.SCRUB
_SC = TreeType.SCLEROPHYLL

CLIMATE_TABLE: dict[str, dict[TreeType, float]] = {
    # ── Tropical (A) ──
    "Af":  {_BT: 0.75, _P: 0.20, _M: 0.05},
    "Am":  {_BT: 0.55, _P: 0.30, _M: 0.15},
    "Aw":  {_BT: 0.40, _D: 0.25, _P: 0.20, _S: 0.15},

    # ── Arid (B) ──
    "BWh": {_S: 1.0},
    "BWk": {_S: 1.0},
    "BSh": {_S: 0.70, _SC: 0.30},
    "BSk": {_S: 0.80, _SC: 0.20},

    # ── Temperate (C) ──
    "Csa": {_SC: 0.45, _C: 0.25, _S: 0.30},
    "Csb": {_C: 0.50, _D: 0.30, _SC: 0.20},
    "Csc": {_C: 0.55, _D: 0.25, _S: 0.20},
    "Cwa": {_BT: 0.45, _D: 0.30, _P: 0.15, _S: 0.10},
    "Cwb": {_D: 0.40, _C: 0.35, _S: 0.25},
    "Cwc": {_C: 0.55, _D: 0.25, _S: 0.20},
    "Cfa": {_D: 0.55, _C: 0.25, _P: 0.10, _BT: 0.10},
    "Cfb": {_D: 0.50, _C: 0.40, _S: 0.10},
    "Cfc": {_C: 0.55, _D: 0.25, _S: 0.20},

    # ── Continental (D) ──
    "Dfa": {_D: 0.50, _C: 0.45, _S: 0.05},
    "Dfb": {_C: 0.45, _D: 0.45, _S: 0.10},
    "Dfc": {_C: 0.90, _D: 0.10},
    "Dfd": {_C: 0.95, _D: 0.05},
    "Dsa": {_SC: 0.45, _C: 0.25, _S: 0.30},
    "Dsb": {_C: 0.50, _D: 0.30, _SC: 0.20},
    "Dsc": {_C: 0.90, _D: 0.10},
    "Dsd": {_C: 0.95, _D: 0.05},
    "Dwa": {_D: 0.45, _C: 0.40, _S: 0.15},
    "Dwb": {_C: 0.45, _D: 0.45, _S: 0.10},
    "Dwc": {_C: 0.90, _D: 0.10},
    "Dwd": {_C: 0.95, _D: 0.05},

    # ── Polar (E) ──
    "ET":  {_S: 0.70, _C: 0.30},
    "EF":  {_S: 1.0},
}


# ── Raster download & cache ──────────────────────────────────────────

def _ensure_koppen_tif() -> pathlib.Path:
    """Download and extract the Köppen GeoTIFF if not cached."""
    tif_path = _KOPPEN_CACHE_DIR / _KOPPEN_TIF
    if tif_path.exists():
        return tif_path

    _KOPPEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _KOPPEN_CACHE_DIR / "koppen_geiger_tif.zip"

    if not zip_path.exists():
        logger.info("Downloading Köppen-Geiger raster (~125 MB) …")
        import requests as _req
        resp = _req.get(_KOPPEN_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        logger.info("Download complete.")

    logger.info("Extracting Köppen GeoTIFF …")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Extract only the present-day 1km TIF
        with zf.open(_KOPPEN_ZIP_MEMBER) as src, open(tif_path, 'wb') as dst:
            import shutil
            shutil.copyfileobj(src, dst)

    # Clean up zip to save space
    zip_path.unlink(missing_ok=True)
    logger.info(f"Köppen raster cached at {tif_path}")
    return tif_path


def _sample_koppen(lat: float, lon: float) -> int:
    """Sample the Köppen raster at a single point.  Returns integer code 1-30."""
    tif_path = _ensure_koppen_tif()
    with rasterio.open(tif_path) as src:
        # Sample at (lon, lat) — rasterio uses (x, y) = (lon, lat)
        vals = list(src.sample([(lon, lat)]))
        if vals and len(vals[0]) > 0:
            return int(vals[0][0])
    return 0  # no data


# ── Public API ────────────────────────────────────────────────────────

def get_koppen_code(lat: float, lon: float) -> str:
    """Return the Köppen code string (e.g. 'Cfb') at the given point."""
    raw = _sample_koppen(lat, lon)
    return KOPPEN_CODES.get(raw, "unknown")


def get_tree_mix(lat: float, lon: float) -> dict[TreeType, float]:
    """Sample Köppen raster and return tree type mix weights.

    Returns dict mapping TreeType → probability (sums to 1.0).
    Falls back to latitude heuristic if raster unavailable or returns
    unknown code.
    """
    try:
        code = get_koppen_code(lat, lon)
    except Exception as exc:
        logger.warning(f"Köppen lookup failed ({exc}), using latitude fallback")
        code = "unknown"

    # PNW fix: wet Csb at high latitude → temperate rainforest
    if code == "Csb" and lat > 45:
        logger.info(f"PNW override: Csb at lat {lat:.1f}° → conifer-dominant")
        return {_C: 0.80, _D: 0.12, _S: 0.08}

    entry = CLIMATE_TABLE.get(code)
    if entry is not None:
        logger.info(f"Köppen zone: {code}")
        return dict(entry)

    # Fallback: simple latitude heuristic (original behavior)
    logger.info(f"Köppen code '{code}' unknown, using latitude fallback")
    al = abs(lat)
    w_conifer = max(0.0, min(1.0, (al - 35) / 20.0))
    if al < 5:
        w_palm = al / 5.0 * 0.15
    elif al < 15:
        w_palm = 0.15 + 0.65 * (al - 5) / 10.0
    elif al < 20:
        w_palm = 0.8
    elif al < 28:
        w_palm = 0.8 * (28 - al) / 8.0
    else:
        w_palm = 0.0
    w_deciduous = max(0.0, 1.0 - w_conifer - w_palm)
    mix = {}
    if w_conifer > 0:
        mix[_C] = w_conifer
    if w_deciduous > 0:
        mix[_D] = w_deciduous
    if w_palm > 0:
        mix[_P] = w_palm
    return mix
