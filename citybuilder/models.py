"""Data classes and path management."""

import pathlib
from dataclasses import dataclass

from shapely.geometry import Polygon, box

from .constants import OUTPUT_DIR, DATA_DIR


class PathManager:
    """Manage paths relative to the CityBuilder directory."""

    @staticmethod
    def get_output_path(filename: str) -> pathlib.Path:
        """Get the output file path."""
        return OUTPUT_DIR / filename

    @staticmethod
    def get_data_path(filename: str) -> pathlib.Path:
        """Get the data file path."""
        return DATA_DIR / filename


@dataclass
class BoundingBox:
    north: float
    south: float
    east: float
    west: float

    def to_polygon(self) -> Polygon:
        """Convert bounding box to shapely polygon."""
        return box(self.west, self.south, self.east, self.north)

    def get_city_name(self) -> str:
        """Return a human-readable city name based on the bounding box."""
        return f"{self.north:.2f}, {self.west:.2f} to {self.south:.2f}, {self.east:.2f}"


class BuildingGeometry:
    """Store detailed building geometry information."""
    def __init__(self):
        self.height = None
        self.ground_level = None
        self.roof_type = None
        self.roof_angle = None
        self.roof_points = None
        self.wall_points = None
        self.footprint_points = None
        self.confidence = 0.0  # 0-1 score of geometry confidence
