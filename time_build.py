"""Time each component of a CityBuilder build."""

import asyncio
import logging
import time
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from citybuilder.builder import CityBuilder
from citybuilder.models import BoundingBox


async def timed_build(name: str, bbox: BoundingBox, use_cache: bool = True):
    builder = CityBuilder(use_cache=use_cache)
    timings = {}

    # Phase 1: process_city (OSM download + DB store)
    t0 = time.perf_counter()
    city_id = await builder.process_city(bbox)
    timings["1. OSM download + DB store"] = time.perf_counter() - t0

    # Phase 2: GLB generation (mesh building, terrain, vegetation, roads, etc.)
    output_path = f"{name}.glb"

    t0 = time.perf_counter()
    builder.generate_glb(city_id, output_path)
    timings["2. GLB generation (total)"] = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"BUILD COMPLETE: {name}")
    print("=" * 60)
    total = 0
    for label, dur in timings.items():
        print(f"  {label}: {dur:.1f}s")
        total += dur
    print(f"  TOTAL: {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    no_cache = "--no-cache" in sys.argv

    # Denver metro area — will be clamped to ~5km by the builder
    bbox = BoundingBox(
        north=39.77,
        south=39.71,
        east=-104.96,
        west=-105.02,
    )
    asyncio.run(timed_build("denver", bbox, use_cache=not no_cache))
