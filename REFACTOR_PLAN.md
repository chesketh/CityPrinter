# Refactor `citybuilder.py` into a Package — Updated Plan

## Context

`citybuilder.py` is a ~3,925-line monolith containing 7 classes (not 6 — `BuildingGeometry` was missed in the original count), configuration constants, CLI commands, and the full data pipeline. Splitting it into focused modules will make it easier to navigate, modify, and test.

## Backward Compatibility

Three files import from the module — all use `from citybuilder import CityBuilder, BoundingBox`:
- `backend/jobs.py` (lazy import inside `_sync_build`)
- `generate_famous.py` (top-level)
- `generate_remaining.py` (top-level)

Converting `citybuilder.py` into a `citybuilder/` package with `__init__.py` re-exports preserves this exactly.

Two other scripts (`query_db.py`, `testdb.py`) do **not** import from `citybuilder` — they use `sqlite3` directly and are unaffected.

## Target Structure

```
CityBuilder/
├── citybuilder/                  # Package (replaces citybuilder.py)
│   ├── __init__.py               # Re-exports CityBuilder, BoundingBox; imports constants first
│   ├── __main__.py               # Enables `python -m citybuilder build ...`
│   ├── constants.py              # Monkey-patch, FEATURE_CATEGORIES, OSM_TAG_OVERRIDES, paths
│   ├── models.py                 # PathManager, BoundingBox, BuildingGeometry
│   ├── database.py               # CityDatabase class
│   ├── ai_helper.py              # AIHelper class
│   ├── lidar.py                  # LidarProcessor class
│   ├── geometry.py               # Transform functions, extrusion, roof generators
│   ├── glb.py                    # generate_glb, generate_stl, _geometry_to_mesh
│   ├── osm_data.py               # download_osm_data + process_features
│   ├── builder.py                # CityBuilder orchestrator (thin)
│   └── cli.py                    # Click commands (build, build_bbox)
├── backend/...
├── frontend/...
```

---

## Actual `self.*` Dependencies (verified against source)

Understanding what each method *actually* accesses on `self` is critical for correct extraction. The original plan overstated dependencies in several places.

| Method | `self.*` attributes used | Lines |
|--------|--------------------------|-------|
| `_download_osm_data` | `self._process_features` (calls it at the end) | 1931 |
| `_process_features` | `self.db`, `self._transform_geometry` | various |
| `generate_glb` | `self.db`, `self._ROOF_SHAPES_SUPPORTED`, `self._generate_roof_mesh`, `self._extrude_watertight` | various |
| `generate_stl` | `self.db`, `self._geometry_to_mesh` | various |
| `_transform_*` | `self.transformer` (the pyproj Transformer) | 2841, 2854, 2879 |
| `_extrude_watertight` | None (pure geometry, calls no other self methods) | — |
| `_generate_roof_mesh` | `self._roof_*` methods (dispatches by shape) | 3581-3592 |
| `_roof_gabled/hipped/skillion` | `self._roof_ridge_info` | 3720, 3758, 3817 |
| `_roof_pyramidal/dome/ridge_info` | None (stateless) | — |
| `approximate_local_xy` | `self.coordinate_scale`, `self.center_lat`, `self.center_lon` | 3840-3843 |

**Key corrections from original plan:**
- `_process_features` does **NOT** use `lidar_processor`, `center_lat`, `center_lon`, or `coordinate_scale`
- `generate_glb` does **NOT** use `center_lat`, `center_lon`, or `coordinate_scale`
- `_download_osm_data` only touches `self` to call `self._process_features`

### Module-level constants used inside methods

| Constant | Used in | Must be imported by |
|----------|---------|---------------------|
| `FEATURE_CATEGORIES` | `generate_glb` (lines 3243-3245) | `glb.py` |
| `OSM_TAG_OVERRIDES` | `_process_features` (line 2064) | `osm_data.py` |
| `BASE_DIR` | `cli.py` build command (line 3892) | `cli.py` |

---

## Implementation Steps

### Step 1: Create `citybuilder/constants.py`

Move from `citybuilder.py`:
- Monkey-patch for osmnx `_overpass_request` (lines 1-6 imports + lines 8-35)
- `OSM_TAG_OVERRIDES` dict (lines 68-86)
- `FEATURE_CATEGORIES` dict (lines 88-197)
- Path constants: `BASE_DIR`, `DATA_DIR`, `LIDAR_DIR`, `OUTPUT_DIR` + directory creation (lines 199-211)
- `load_dotenv()` call (line 200)
- Logging configuration (lines 214-215)

**Why monkey-patch lives here:** It must execute before any osmnx usage. `__init__.py` will import `constants` first to guarantee this.

### Step 2: Create `citybuilder/models.py`

Move from `citybuilder.py`:
- `PathManager` class (lines 217-233) — depends on `LIDAR_DIR`, `OUTPUT_DIR`, `DATA_DIR` from constants
- `BoundingBox` dataclass (lines 235-248) — depends on `shapely.geometry.Polygon`
- `BuildingGeometry` class (lines 1095-1105) — no external dependencies

### Step 3: Create `citybuilder/database.py`

Move from `citybuilder.py`:
- `CityDatabase` class (lines 250-978) — entirely self-contained
- Imports: `sqlite3`, `json`, `logging`, `pathlib`, `typing`, `pyproj`
- Import `BoundingBox` from `.models`
- Import `DATA_DIR` from `.constants` (for default db_path)

### Step 4: Create `citybuilder/ai_helper.py`

Move from `citybuilder.py`:
- `AIHelper` class (lines 980-1094)
- Imports: `json`, `logging`, `openai.OpenAI`
- Import `CityDatabase` from `.database` (type hint / constructor arg)

### Step 5: Create `citybuilder/lidar.py`

Move from `citybuilder.py`:
- `LidarProcessor` class (lines 1107-1638)
- Imports: `numpy`, `laspy`, `scipy.spatial.cKDTree`, `scipy.interpolate.griddata`, `logging`, `pathlib`, `tqdm`, `pyproj`
- Import `CityDatabase` from `.database`
- Import `BuildingGeometry` from `.models`
- Import `LIDAR_DIR` from `.constants`

### Step 6: Create `citybuilder/geometry.py`

Extract from `CityBuilder` class as **standalone functions** (not a class):

| Current method | New function signature |
|----------------|----------------------|
| `_transform_geometry(self, geom)` | `transform_geometry(geom, transformer)` |
| `_transform_polygon(self, polygon)` | `transform_polygon(polygon, transformer)` |
| `_transform_linestring(self, line)` | `transform_linestring(line, transformer)` |
| `_extrude_watertight(self, geometry, height, ...)` | `extrude_watertight(geometry, height, ...)` |
| `_generate_roof_mesh(self, geometry, roof_shape, ...)` | `generate_roof_mesh(geometry, roof_shape, ...)` |
| `_roof_pyramidal(self, coords, ...)` | `_roof_pyramidal(coords, ...)` |
| `_roof_dome(self, coords, ...)` | `_roof_dome(coords, ...)` |
| `_roof_ridge_info(self, polygon)` | `_roof_ridge_info(polygon)` |
| `_roof_gabled(self, coords, ...)` | `_roof_gabled(coords, ...)` |
| `_roof_hipped(self, coords, ...)` | `_roof_hipped(coords, ...)` |
| `_roof_skillion(self, coords, ...)` | `_roof_skillion(coords, ...)` |

Also move:
- `_ROOF_SHAPES_SUPPORTED` frozenset → module-level `ROOF_SHAPES_SUPPORTED`

**Design rationale (changed from original plan):** The original plan proposed a `GeometryEngine` class. This is unnecessary — the only shared state is `transformer`, which is only used by the 3 transform functions and can be passed as an argument. The roof/extrusion functions are already stateless (they only call each other, with no external state). Plain functions are simpler, more testable, and avoid an artificial class hierarchy.

Internal call graph (all stay in `geometry.py`):
- `transform_geometry` → calls `transform_polygon`, `transform_linestring`
- `generate_roof_mesh` → dispatches to `_roof_pyramidal`, `_roof_gabled`, `_roof_hipped`, `_roof_dome`, `_roof_skillion`
- `_roof_gabled`, `_roof_hipped`, `_roof_skillion` → call `_roof_ridge_info`

### Step 7: Create `citybuilder/glb.py`

Extract from `CityBuilder` class as functions:

| Current method | New function signature |
|----------------|----------------------|
| `generate_glb(self, city_id, output_path, ...)` | `generate_glb(db, city_id, output_path, ...)` |
| `generate_stl(self, city_id, output_path)` | `generate_stl(db, city_id, output_path)` |
| `_geometry_to_mesh(self, geometry, height)` | `_geometry_to_mesh(geometry, height)` |

Imports from other modules:
- `from .constants import FEATURE_CATEGORIES` (used in road width lookup, lines 3243-3245)
- `from .geometry import extrude_watertight, generate_roof_mesh, ROOF_SHAPES_SUPPORTED`
- `from .models import PathManager` (for output path resolution)
- `from .database import CityDatabase` (type hint)

**Note:** `generate_glb` currently returns the absolute path (a `str`). The `builder.py` wrapper must preserve this return value — `backend/jobs.py` (line 51) depends on it.

### Step 8: Create `citybuilder/osm_data.py`

Extract from `CityBuilder` class as functions:

| Current method | New function signature |
|----------------|----------------------|
| `_download_osm_data(self, bbox, city_id, ...)` | `download_osm_data(db, bbox, city_id, transformer, ...)` |
| `_process_features(self, buildings, ...)` | `process_features(db, transformer, buildings, ...)` |

Imports from other modules:
- `from .constants import OSM_TAG_OVERRIDES` (used line 2064 for per-building tag overrides)
- `from .geometry import transform_geometry` (called throughout `_process_features`)
- `from .database import CityDatabase` (type hint)

**Dependency correction:** The original plan listed `lidar_processor, center_lat, center_lon, coordinate_scale` as needed dependencies. This is wrong — `_process_features` only accesses `self.db` and `self._transform_geometry`. The `transformer` is needed indirectly (passed through to `transform_geometry`).

`download_osm_data` calls `process_features` at the end (line 1931), so both being in the same module keeps this simple — it's a direct function call.

### Step 9: Create `citybuilder/builder.py`

The `CityBuilder` class becomes a thin orchestrator:

```python
class CityBuilder:
    def __init__(self, coordinate_scale=111320.0, center_lat=47.6062, center_lon=-122.3321):
        self.coordinate_scale = coordinate_scale
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.db = CityDatabase()
        self.ai_helper = AIHelper(self.db)
        self.transformer = None
        self.lidar_processor = LidarProcessor(self.db)

    async def process_city(self, location, progress_callback=None) -> int:
        # Setup transformer, load lidar, then delegate:
        osm_data.download_osm_data(self.db, bbox, city_id, self.transformer, ...)

    def generate_glb(self, city_id, output_path, **kwargs) -> str:
        return glb.generate_glb(self.db, city_id, output_path, **kwargs)

    def generate_stl(self, city_id, output_path):
        return glb.generate_stl(self.db, city_id, output_path)

    # Keep these utility methods on the class (they use self.coordinate_scale etc.):
    def approximate_local_xy(self, lat, lon) -> (float, float)
    def approximate_local_bbox(self, north, south, east, west) -> dict
    def get_lidar_files_for_approx_bbox(self, latlon_bbox) -> list
```

### Step 10: Create `citybuilder/cli.py`

Move Click commands:
- `cli()` group (line 3880)
- `build()` command (line 3884)
- `build_bbox()` command (line 3902)
- `async_build()` helper (line 3914)
- `if __name__ == "__main__": cli()` guard

Imports:
- `from .builder import CityBuilder`
- `from .models import BoundingBox`
- `from .constants import BASE_DIR`
- `from .models import PathManager`

### Step 11: Create `citybuilder/__main__.py` (missing from original plan)

```python
from citybuilder.cli import cli
cli()
```

This enables `python -m citybuilder build "Space Needle"`. The original plan listed this as a verification target but didn't include the file needed to make it work.

### Step 12: Create `citybuilder/__init__.py`

```python
# Import constants FIRST to ensure the osmnx monkey-patch is applied
# before any other module imports osmnx.
from citybuilder import constants as _constants  # noqa: F401

from citybuilder.builder import CityBuilder
from citybuilder.models import BoundingBox
```

### Step 13: Delete old `citybuilder.py` and clean up `__pycache__`

- Delete `citybuilder.py`
- Delete `__pycache__/citybuilder.cpython-313.pyc` — **critical**: stale bytecode from the old module file will shadow the new package directory and cause confusing import failures

---

## Key Design Decisions

### Extract as functions, not new classes
`_process_features`, `generate_glb`, and `download_osm_data` become module-level functions rather than methods on some new class. This avoids creating artificial class hierarchies. Dependencies are passed as arguments.

### No GeometryEngine class (changed from original plan)
The original plan proposed a `GeometryEngine` class for roof/extrusion functions. After verifying the actual `self.*` usage:
- Transform functions need only `transformer` → pass it as an argument
- Roof/extrusion functions are fully stateless → plain functions
- A class would add complexity with no benefit

### Pass dependencies explicitly
Instead of relying on `self.db`, functions take `db` as an argument. Similarly, `transform_geometry` takes `transformer` as an argument. This makes dependencies visible and testable.

### Monkey-patch stays in constants.py
It's imported first via `__init__.py`, ensuring the patch applies before any osmnx usage anywhere in the package.

### `__main__.py` is required (added to original plan)
Without it, `python -m citybuilder build` would fail. The original plan listed this CLI invocation as a verification target but didn't include the file.

---

## Files Created/Modified

| File | Action |
|------|--------|
| `citybuilder/` | New package directory |
| `citybuilder/__init__.py` | New — re-exports + monkey-patch import |
| `citybuilder/__main__.py` | **New** — enables `python -m` invocation |
| `citybuilder/constants.py` | New — config, monkey-patch, paths |
| `citybuilder/models.py` | New — PathManager, BoundingBox, BuildingGeometry |
| `citybuilder/database.py` | New — CityDatabase |
| `citybuilder/ai_helper.py` | New — AIHelper |
| `citybuilder/lidar.py` | New — LidarProcessor |
| `citybuilder/geometry.py` | New — transforms + extrusion + roofs (as functions) |
| `citybuilder/glb.py` | New — GLB/STL export |
| `citybuilder/osm_data.py` | New — download + process |
| `citybuilder/builder.py` | New — CityBuilder orchestrator |
| `citybuilder/cli.py` | New — Click commands |
| `citybuilder.py` | **Deleted** |
| `__pycache__/citybuilder.cpython-313.pyc` | **Deleted** |

---

## Verification

1. `from citybuilder import CityBuilder, BoundingBox` still works
2. `python -m citybuilder build "Space Needle"` CLI still works
3. `backend/jobs.py` deferred import still works
4. `generate_famous.py` runs successfully
5. Rebuild one building (e.g., Space Needle) and verify GLB output is identical
6. No `__pycache__` artifacts from old module remain

---

## Changes from Original Plan

| # | What changed | Why |
|---|-------------|-----|
| 1 | **Removed `GeometryEngine` class** — all geometry code is plain functions | Roof/extrusion methods are stateless. Only `transformer` is needed by transform functions, and it's simpler to pass as an argument than wrap in a class. |
| 2 | **Corrected dependency list for `osm_data.py`** | `_process_features` only uses `db` + `transform_geometry`. It does NOT use `lidar_processor`, `center_lat`, `center_lon`, or `coordinate_scale`. |
| 3 | **Corrected dependency list for `glb.py`** | `generate_glb` only uses `db` + geometry functions. It does NOT use `center_lat`, `center_lon`, or `coordinate_scale`. |
| 4 | **Added `__main__.py`** | Required for `python -m citybuilder` to work. Was listed as a verification target but missing from the file list. |
| 5 | **Added `__pycache__` cleanup step** | Stale `.pyc` from the old `citybuilder.py` module will shadow the new `citybuilder/` package. |
| 6 | **Noted `FEATURE_CATEGORIES` import in `glb.py`** | Used at lines 3243-3245 for road width lookup — must be imported from constants. |
| 7 | **Noted `OSM_TAG_OVERRIDES` import in `osm_data.py`** | Used at line 2064 for per-building tag overrides. |
| 8 | **Corrected class count** | 7 classes total (PathManager, BoundingBox, CityDatabase, AIHelper, BuildingGeometry, LidarProcessor, CityBuilder), not 6. |
| 9 | **Added explicit `constants` import in `__init__.py`** | Ensures monkey-patch runs before any module-level osmnx imports in other submodules. |
| 10 | **Noted `generate_glb` return value** | `backend/jobs.py` line 51 depends on the return value (`glb_path`). The `builder.py` wrapper must preserve this. |
