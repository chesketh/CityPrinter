from pydantic import BaseModel
from typing import Optional


class BuildByNameRequest(BaseModel):
    location: str
    output_format: str = "glb"  # "glb" or "ply"
    scale: float = 1.0          # scale factor for PLY (e.g. 0.001 for m→mm)


class BuildByBboxRequest(BaseModel):
    north: float
    south: float
    east: float
    west: float
    output_format: str = "glb"
    scale: float = 1.0


class GeocodeResponse(BaseModel):
    north: float
    south: float
    east: float
    west: float
    display_name: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[dict] = None
    bbox: Optional[dict] = None


class ModelInfo(BaseModel):
    name: str
    filename: str
    bbox: Optional[dict] = None
