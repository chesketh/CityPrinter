import asyncio
import logging

from fastapi import APIRouter, HTTPException

from backend.geocoder import GeocoderService
from backend.jobs import job_manager
from backend.models import BuildByNameRequest, BuildByBboxRequest, JobResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/build", tags=["build"])

geocoder = GeocoderService()


@router.post("/name", response_model=JobResponse)
async def build_by_name(request: BuildByNameRequest):
    """Start a city build by location name.

    The location is first geocoded (DB lookup then Nominatim) to obtain a
    bounding box, which is then handed to the CityBuilder pipeline.  The
    heavy lifting runs in a background task; the caller receives a job ID
    immediately and can poll ``/status/{job_id}`` for progress.
    """
    try:
        bbox = geocoder.get_bbox(request.location)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    job = job_manager.create_job()
    asyncio.create_task(job_manager.run_build(
        job, bbox, request.location,
        output_format=request.output_format,
        scale=request.scale))

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        result=job.result,
        bbox=bbox,
    )


@router.post("/bbox", response_model=JobResponse)
async def build_by_bbox(request: BuildByBboxRequest):
    """Start a city build from explicit bounding-box coordinates."""
    bbox = {
        "north": request.north,
        "south": request.south,
        "east": request.east,
        "west": request.west,
    }
    name = f"custom-{bbox['north']:.4f}-{bbox['west']:.4f}"

    job = job_manager.create_job()
    asyncio.create_task(job_manager.run_build(
        job, bbox, name,
        output_format=request.output_format,
        scale=request.scale))

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        result=job.result,
        bbox=bbox,
    )


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_build_status(job_id: str):
    """Poll the status of a running or completed build job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        result=job.result,
    )
