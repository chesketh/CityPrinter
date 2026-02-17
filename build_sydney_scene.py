"""Build full Sydney Opera House scene (buildings, water, terrain + shells)."""
import asyncio, sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from citybuilder import CityBuilder, BoundingBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OFFSET = 0.0015
lat, lon = -33.8568, 151.2153

bbox = BoundingBox(
    north=lat + OFFSET,
    south=lat - OFFSET,
    east=lon + OFFSET,
    west=lon - OFFSET,
)

t0 = time.time()
builder = CityBuilder()
city_id = asyncio.run(builder.process_city(bbox))
out = builder.generate_glb(city_id, "sydney-opera-house-scene.glb")
print(f"\nDone in {time.time()-t0:.1f}s -> {out}")
