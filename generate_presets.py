"""Batch-generate GLB models for all preset locations.

Usage:
    python generate_presets.py                    # build all presets
    python generate_presets.py --start 50         # resume from preset #50
    python generate_presets.py --only manhattan-nyc
    python generate_presets.py --category Downtown
    python generate_presets.py --dry-run           # preview without building
"""

import argparse
import asyncio
import json
import logging
import math
import pathlib
import sys
import time
import traceback

# Project root must be on sys.path so citybuilder package resolves.
PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from citybuilder.builder import CityBuilder
from citybuilder.models import BoundingBox

# ── Paths ────────────────────────────────────────────────────────────────
PRESETS_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "presets.json"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Bbox span limits (must match backend/routers/build.py) ───────────────
_MAX_LAT_SPAN = 0.045
_MAX_LON_SPAN_AT_EQ = 0.045
_MIN_LAT_SPAN = 0.005
_MIN_LON_SPAN_AT_EQ = 0.005

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_presets")


# ── Helpers ──────────────────────────────────────────────────────────────

def _clamp_bbox(bbox: dict) -> dict:
    """Clamp bbox to between ~500 m and ~5 km, centred on the midpoint.

    Mirrors backend/routers/build.py exactly so batch output matches the
    web-triggered pipeline.
    """
    lat_span = bbox["north"] - bbox["south"]
    lon_span = bbox["east"] - bbox["west"]

    mid_lat = (bbox["north"] + bbox["south"]) / 2.0
    mid_lon = (bbox["east"] + bbox["west"]) / 2.0

    cos_lat = max(math.cos(math.radians(mid_lat)), 0.01)
    max_lon = _MAX_LON_SPAN_AT_EQ / cos_lat
    min_lon = _MIN_LON_SPAN_AT_EQ / cos_lat

    needs_clamp = (
        lat_span > _MAX_LAT_SPAN
        or lon_span > max_lon
        or lat_span < _MIN_LAT_SPAN
        or lon_span < min_lon
    )

    if needs_clamp:
        clamped_lat = max(_MIN_LAT_SPAN, min(lat_span, _MAX_LAT_SPAN))
        clamped_lon = max(min_lon, min(lon_span, max_lon))
        half_lat = clamped_lat / 2.0
        half_lon = clamped_lon / 2.0
        bbox = {
            "north": mid_lat + half_lat,
            "south": mid_lat - half_lat,
            "east": mid_lon + half_lon,
            "west": mid_lon - half_lon,
        }
    return bbox


def bbox_from_preset(preset: dict) -> dict:
    """Compute bbox dict from a preset's lat, lon, span fields.

    The span value in presets.json is in degrees.  We apply it symmetrically
    around the center, adjusting the longitude span by cos(lat) so the
    physical extent is roughly square.
    """
    lat = preset["lat"]
    lon = preset["lon"]
    span = preset["span"]

    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    half_lat = span / 2.0
    half_lon = (span / cos_lat) / 2.0

    return {
        "north": lat + half_lat,
        "south": lat - half_lat,
        "east": lon + half_lon,
        "west": lon - half_lon,
    }


def format_duration(seconds: float) -> str:
    """Format seconds into a readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


def progress_printer(preset_id: str):
    """Return a progress callback that logs under the preset name."""
    def _cb(pct, msg):
        logger.info("  [%s] %3.0f%% — %s", preset_id, pct, msg)
    return _cb


# ── Main ─────────────────────────────────────────────────────────────────

async def build_one(builder: CityBuilder, preset: dict) -> bool:
    """Build a single preset. Returns True on success."""
    preset_id = preset["id"]
    output_path = OUTPUT_DIR / f"{preset_id}.glb"

    raw_bbox = bbox_from_preset(preset)
    clamped = _clamp_bbox(raw_bbox)
    bbox = BoundingBox(
        north=clamped["north"],
        south=clamped["south"],
        east=clamped["east"],
        west=clamped["west"],
    )

    cb = progress_printer(preset_id)

    city_id = await builder.process_city(bbox, progress_callback=cb)
    builder.generate_glb(city_id, str(output_path), progress_callback=cb)

    if not output_path.exists():
        raise RuntimeError(f"GLB file was not created at {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("  [%s] GLB saved (%.1f MB)", preset_id, size_mb)
    return True


async def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate GLB city models for preset locations."
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Skip the first N presets (for resuming).",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Build only the preset with this id.",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Build only presets in this category.",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Stop before this preset index (for sharding across workers).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the GLB already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be built without actually building.",
    )
    args = parser.parse_args()

    # Load presets
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        presets = json.load(f)

    logger.info("Loaded %d presets from %s", len(presets), PRESETS_PATH)

    # Filter
    if args.only:
        presets = [p for p in presets if p["id"] == args.only]
        if not presets:
            logger.error("No preset found with id '%s'", args.only)
            sys.exit(1)
    elif args.category:
        presets = [p for p in presets if p["category"].lower() == args.category.lower()]
        if not presets:
            logger.error("No presets found in category '%s'", args.category)
            sys.exit(1)
        logger.info("Filtered to %d presets in category '%s'", len(presets), args.category)

    # Apply --start / --end slicing
    if args.start > 0 or args.end is not None:
        presets = presets[args.start:args.end]
        logger.info("Slice [%d:%s] → %d presets", args.start,
                     args.end if args.end is not None else "", len(presets))

    total = len(presets)

    # Dry run
    if args.dry_run:
        skipped = 0
        would_build = 0
        for i, p in enumerate(presets, 1):
            output_path = OUTPUT_DIR / f"{p['id']}.glb"
            exists = output_path.exists()
            if exists and not args.force:
                status = "SKIP (exists)"
                skipped += 1
            else:
                status = "REBUILD" if exists else "BUILD"
                would_build += 1
            print(f"  {i:3d}/{total}  [{status:14s}]  {p['id']:<35s}  {p['name']}")
        print(f"\nDry run summary: {would_build} to build, {skipped} already exist, {total} total")
        return

    # Build
    OUTPUT_DIR.mkdir(exist_ok=True)
    builder = CityBuilder()

    successes = 0
    failures = 0
    skipped = 0
    failed_ids = []
    build_times = []

    for i, preset in enumerate(presets, 1):
        preset_id = preset["id"]
        output_path = OUTPUT_DIR / f"{preset_id}.glb"

        # Skip if already exists (unless --force)
        if output_path.exists() and not args.force:
            logger.info("[%3d/%d] SKIP  %s  (already exists)", i, total, preset_id)
            skipped += 1
            continue

        # ETA calculation
        if build_times:
            avg_time = sum(build_times) / len(build_times)
            remaining = total - i - skipped + 1
            eta = format_duration(avg_time * remaining)
            eta_str = f"  ETA: {eta}"
        else:
            eta_str = ""

        logger.info(
            "[%3d/%d] BUILD %s  (%s)%s",
            i, total, preset_id, preset["name"], eta_str,
        )

        t0 = time.monotonic()
        try:
            await build_one(builder, preset)
            elapsed = time.monotonic() - t0
            build_times.append(elapsed)
            successes += 1
            logger.info(
                "[%3d/%d] OK    %s  (%s)",
                i, total, preset_id, format_duration(elapsed),
            )
        except Exception:
            elapsed = time.monotonic() - t0
            failures += 1
            failed_ids.append(preset_id)
            logger.error(
                "[%3d/%d] FAIL  %s  (%s)\n%s",
                i, total, preset_id, format_duration(elapsed),
                traceback.format_exc(),
            )

    # Summary
    print("\n" + "=" * 60)
    print("BATCH GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total presets:  {total}")
    print(f"  Successes:      {successes}")
    print(f"  Failures:       {failures}")
    print(f"  Skipped:        {skipped}")
    if build_times:
        print(f"  Avg build time: {format_duration(sum(build_times) / len(build_times))}")
        print(f"  Total time:     {format_duration(sum(build_times))}")
    if failed_ids:
        print(f"\n  Failed presets:")
        for fid in failed_ids:
            print(f"    - {fid}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
