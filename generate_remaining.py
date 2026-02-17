"""Generate the 4 remaining famous buildings that were missed due to rate limiting."""
import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from citybuilder import CityBuilder, BoundingBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OFFSET = 0.0015

REMAINING = [
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

    # Load existing results
    existing = []
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            existing = json.load(f)

    new_results = []
    for i, entry in enumerate(REMAINING):
        logger.info(f"\n[{i+1}/{len(REMAINING)}] {entry['name']} ({entry['city']})")
        result = build_one(entry)
        new_results.append(result)

        if i < len(REMAINING) - 1:
            logger.info("  Pausing 10s before next build...")
            time.sleep(10)

    # Also fix the 16 existing results — they were marked "failed" due to the str bug
    # but the files actually exist. Correct them.
    fixed_existing = []
    for r in existing:
        glb_file = OUTPUT_DIR / r["filename"]
        if r["status"] == "failed" and glb_file.exists():
            r["status"] = "success"
            r["glb_path"] = str(glb_file)
            r["glb_size_kb"] = round(glb_file.stat().st_size / 1024, 1)
            r.pop("error", None)
            r.pop("traceback", None)
            if r["glb_size_kb"] < 5:
                r["issues"].append("Very small file — may be empty or missing geometry")
        fixed_existing.append(r)

    # Merge: existing (fixed) + new
    all_results = fixed_existing + new_results

    with open(REPORT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("FAMOUS BUILDINGS REPORT (ALL 20)")
    print("=" * 70)

    succeeded = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "failed"]
    with_issues = [r for r in succeeded if r["issues"]]

    print(f"\nTotal: {len(all_results)}  |  Success: {len(succeeded)}  |  Failed: {len(failed)}")
    print()

    for r in all_results:
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
