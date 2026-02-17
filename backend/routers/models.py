import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend import config
from backend.models import ModelInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=List[ModelInfo])
async def list_models():
    """Return metadata for every ``.glb`` file in the output directory."""
    output_dir: Path = config.OUTPUT_DIR
    if not output_dir.exists():
        return []

    models: list[ModelInfo] = []
    for glb_file in sorted(output_dir.glob("*.glb")):
        models.append(
            ModelInfo(
                name=glb_file.stem.replace("-", " ").title(),
                filename=glb_file.name,
                bbox=None,
            )
        )
    return models


@router.get("/{filename}")
async def get_model(filename: str):
    """Serve a specific GLB file from the output directory."""
    file_path = config.OUTPUT_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        path=str(file_path),
        media_type="model/gltf-binary",
        filename=filename,
    )
