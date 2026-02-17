import sys
import pathlib

# Ensure the project root (parent of backend/) is on sys.path so that
# ``import citybuilder`` resolves to CityBuilder/citybuilder.py.
_project_root = str(pathlib.Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.routers import build, geocode, models

app = FastAPI(
    title="CityBuilder API",
    description="Backend API for the CityBuilder 3-D city model generator",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS -- allow the Vite dev server on localhost:5173
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(build.router)
app.include_router(geocode.router)
app.include_router(models.router)

# ---------------------------------------------------------------------------
# Static files -- serve generated GLB/STL assets
# ---------------------------------------------------------------------------
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(config.OUTPUT_DIR)), name="output")


@app.get("/")
async def root():
    return {"status": "ok", "service": "CityBuilder API"}
