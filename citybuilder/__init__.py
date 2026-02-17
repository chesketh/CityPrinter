"""CityBuilder package — 3D city model generation from OpenStreetMap data.

Import constants FIRST to ensure the osmnx monkey-patch is applied
before any other module imports osmnx.
"""

from citybuilder import constants as _constants  # noqa: F401

from citybuilder.builder import CityBuilder
from citybuilder.models import BoundingBox
