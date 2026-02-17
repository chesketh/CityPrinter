"""Generate 3D models of 20 famous global buildings."""
import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from citybuilder import CityBuilder, BoundingBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 20 famous buildings with tight bounding boxes (~200m around center)
OFFSET = 0.0015  # ~150m in degrees

BUILDINGS = [
    {"name": "Empire State Building", "city": "New York", "lat": 40.7484, "lon": -73.9857},
    {"name": "Chrysler Building", "city": "New York", "lat": 40.7516, "lon": -73.9755},
    {"name": "One World Trade Center", "city": "New York", "lat": 40.7127, "lon": -74.0134},
    {"name": "Willis Tower", "city": "Chicago", "lat": 41.8789, "lon": -87.6359},
    {"name": "Space Needle", "city": "Seattle", "lat": 47.6205, "lon": -122.3493},
    {"name": "Eiffel Tower", "city": "Paris", "lat": 48.8584, "lon": 2.2945},
    {"name": "Big Ben", "city": "London", "lat": 51.5007, "lon": -0.1246},
    {"name": "The Shard", "city": "London", "lat": 51.5045, "lon": -0.0865},
    {"name": "Sydney Opera House", "city": "Sydney", "lat": -33.8568, "lon": 151.2153},
    {"name": "Burj Khalifa", "city": "Dubai", "lat": 25.1972, "lon": 55.2744},
    {"name": "Colosseum", "city": "Rome", "lat": 41.8902, "lon": 12.4922},
    {"name": "Sagrada Familia", "city": "Barcelona", "lat": 41.4036, "lon": 2.1744},
    {"name": "St Basils Cathedral", "city": "Moscow", "lat": 55.7525, "lon": 37.6231},
    {"name": "Taj Mahal", "city": "Agra", "lat": 27.1751, "lon": 78.0421},
    {"name": "CN Tower", "city": "Toronto", "lat": 43.6426, "lon": -79.3871},
    {"name": "Gateway Arch", "city": "St Louis", "lat": 38.6247, "lon": -90.1848},
    {"name": "Leaning Tower of Pisa", "city": "Pisa", "lat": 43.7230, "lon": 10.3966},
    {"name": "Brandenburg Gate", "city": "Berlin", "lat": 52.5163, "lon": 13.3777},
    {"name": "Petronas Towers", "city": "Kuala Lumpur", "lat": 3.1578, "lon": 101.7117},
    {"name": "Tokyo Tower", "city": "Tokyo", "lat": 35.6586, "lon": 139.7454},
]

OUTPUT_DIR = Path(__file__).parent / "output"
REPORT_PATH = OUTPUT_DIR / "famous_buildings_report.json"


def safe_filename(name: str) -> str:
    return name.lower().replace(" ", "-").replace("'", "").replace(",", "")


def build_one(entry: dict) -> dict:
    """Build a single building model. Returns a result dict."""
    name = entry["name"]
    city = entry["city"]
    lat, lon = entry["lat"], entry["lon"]
    filename = f"{safe_filename(name)}.glb"
    result = {
        "name": name,
        "city": city,
        "filename": filename,
        "status": "pending",
        "issues": [],
        "duration_s": 0,
    }

    bbox = BoundingBox(
        north=lat + OFFSET,
        south=lat - OFFSET,
        east=lon + OFFSET,
        west=lon - OFFSET,
    )

    logger.info(f"=== Building: {name} ({city}) ===")
    t0 = time.time()

    try:
        builder = CityBuilder()
        city_id = asyncio.run(builder.process_city(bbox))
        glb_path = builder.generate_glb(city_id, filename)
        elapsed = time.time() - t0
        result["duration_s"] = round(elapsed, 1)
        result["status"] = "success"
        result["glb_path"] = str(glb_path)
        result["glb_size_kb"] = round(Path(glb_path).stat().st_size / 1024, 1)
        logger.info(f"  -> OK in {elapsed:.1f}s, {result['glb_size_kb']} KB")

        # Basic sanity checks
        if result["glb_size_kb"] < 5:
            result["issues"].append("Very small file — may be empty or missing geometry")
        if result["glb_size_kb"] > 50000:
            result["issues"].append("Very large file — may have excessive geometry")

    except Exception as exc:
        elapsed = time.time() - t0
        result["duration_s"] = round(elapsed, 1)
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  -> FAILED: {exc}")

    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []

    for i, entry in enumerate(BUILDINGS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(BUILDINGS)}] {entry['name']} ({entry['city']})")
        logger.info(f"{'='*60}")

        result = build_one(entry)
        results.append(result)

        # Save intermediate report after each build
        with open(REPORT_PATH, "w") as f:
            json.dump(results, f, indent=2)

        # Pause between builds to avoid Overpass rate limiting
        if i < len(BUILDINGS) - 1:
            logger.info("  Pausing 5s before next build...")
            time.sleep(5)

    # Print summary
    print("\n" + "=" * 70)
    print("FAMOUS BUILDINGS REPORT")
    print("=" * 70)

    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    with_issues = [r for r in succeeded if r["issues"]]

    print(f"\nTotal: {len(results)}  |  Success: {len(succeeded)}  |  Failed: {len(failed)}")
    print()

    for r in results:
        status = "OK" if r["status"] == "success" else "FAIL"
        size = f'{r.get("glb_size_kb", 0):.0f} KB' if r["status"] == "success" else r.get("error", "unknown")
        issues = "; ".join(r.get("issues", []))
        flag = f"  !! {issues}" if issues else ""
        print(f"  [{status:4s}] {r['name']:30s} ({r['city']:15s}) {r['duration_s']:6.1f}s  {size}{flag}")

    if with_issues:
        print(f"\nBuildings with issues ({len(with_issues)}):")
        for r in with_issues:
            print(f"  - {r['name']}: {'; '.join(r['issues'])}")

    if failed:
        print(f"\nFailed builds ({len(failed)}):")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'unknown')}")

    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
