import asyncio
import logging
import sys
import pathlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

# Ensure the project root is importable so we can reach citybuilder.py
_project_root = str(pathlib.Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    message: str = "Queued"
    result: Optional[dict] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _sync_build(north: float, south: float, east: float, west: float,
                output_filename: str,
                output_format: str = "glb",
                scale: float = 1.0,
                progress_callback=None) -> dict:
    """Run the full CityBuilder pipeline in a dedicated thread.

    A *new* event loop is created inside the thread (via ``asyncio.run``)
    because ``CityBuilder.process_city`` is declared ``async`` even though,
    when given a ``BoundingBox``, it never awaits anything that truly yields
    to the event loop (the heavy work is synchronous OSMnx / LiDAR I/O).
    """
    from citybuilder import CityBuilder, BoundingBox  # noqa: import here to avoid top-level side-effects
    from backend import config as _cfg

    bbox = BoundingBox(north=north, south=south, east=east, west=west)
    builder = CityBuilder(use_cache=_cfg.USE_CACHE)
    city_id = asyncio.run(builder.process_city(bbox, progress_callback=progress_callback))

    if output_format == "ply":
        safe_name = output_filename.replace('.glb', '').replace('.ply', '')
        ply_result = builder.generate_ply_single(
            city_id, output_filename,
            name=safe_name, scale=scale,
            progress_callback=progress_callback)
        ply_filename = safe_name + '.ply'
        return {
            "city_id": city_id,
            "format": "ply",
            "ply_path": ply_result['output_path'],
            "model_url": f"/output/{ply_filename}",
            "faces": ply_result['faces'],
            "vertices": ply_result['vertices'],
            "watertight": ply_result['watertight'],
            "size_mb": ply_result['size_mb'],
        }
    else:
        glb_path = builder.generate_glb(city_id, output_filename,
                                        progress_callback=progress_callback)
        return {
            "city_id": city_id,
            "format": "glb",
            "glb_path": str(glb_path),
            "model_url": f"/output/{output_filename}",
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def create_job(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def run_build(self, job: Job, bbox: dict, name: str,
                        output_format: str = "glb", scale: float = 1.0) -> None:
        """Execute the build pipeline, updating *job* with progress."""
        try:
            job.status = JobStatus.running
            job.progress = 5.0
            job.message = "Preparing workspace..."

            # Derive a filename-safe string from the location name
            safe_name = (
                name.lower()
                .replace(" ", "-")
                .replace(",", "")
                .replace("'", "")
            )
            output_filename = f"{safe_name}.glb"

            def _update_progress(pct: float, msg: str) -> None:
                job.progress = pct
                job.message = msg

            result = await asyncio.to_thread(
                _sync_build,
                bbox["north"],
                bbox["south"],
                bbox["east"],
                bbox["west"],
                output_filename,
                output_format=output_format,
                scale=scale,
                progress_callback=_update_progress,
            )

            job.progress = 100.0
            job.message = "Build complete"
            job.status = JobStatus.completed
            job.result = result

        except Exception as exc:
            logger.exception("Build failed for job %s", job.id)
            job.status = JobStatus.failed
            job.progress = 0.0
            job.message = f"Build failed: {exc}"


# Singleton instance used across the application
job_manager = JobManager()
