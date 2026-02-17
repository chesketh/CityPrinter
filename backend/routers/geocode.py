import logging

from fastapi import APIRouter, HTTPException, Query

from backend.geocoder import GeocoderService
from backend.models import GeocodeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["geocode"])

geocoder = GeocoderService()


@router.get("/geocode", response_model=GeocodeResponse)
def geocode_location(query: str = Query(..., min_length=1)):
    """Resolve a location name to a bounding box.

    Checks the local DB (landmarks, neighborhoods) first, then falls back
    to Nominatim.  Uses ``def`` (not ``async def``) so FastAPI runs it in
    a threadpool — the blocking Nominatim HTTP call won't stall the event loop.
    """
    try:
        bbox = geocoder.get_bbox(query)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return GeocodeResponse(
        north=bbox["north"],
        south=bbox["south"],
        east=bbox["east"],
        west=bbox["west"],
        display_name=bbox.get("display_name"),
    )
