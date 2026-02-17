"""LidarProcessor — LiDAR point cloud processing for building geometry."""

import pathlib
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import laspy
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point, MultiPoint
from tqdm import tqdm

from .constants import BASE_DIR, LIDAR_DIR
from .database import CityDatabase
from .models import BoundingBox, BuildingGeometry

logger = logging.getLogger(__name__)


class LidarProcessor:
    """Handle LiDAR data processing for building heights and geometry."""

    # LAS classification codes
    GROUND_CLASSES = {2}  # Ground
    BUILDING_CLASSES = {6}  # Building
    NOISE_CLASSES = {7}  # Noise
    WATER_CLASSES = {9}  # Water

    # Roof type patterns
    ROOF_PATTERNS = {
        'flat': {'max_angle': 5},
        'gabled': {'min_angle': 15, 'max_angle': 50, 'symmetry': True},
        'hipped': {'min_angle': 15, 'max_angle': 50, 'corners': 4},
        'mansard': {'min_angle': 60, 'max_angle': 85, 'levels': 2},
        'pyramidal': {'min_angle': 15, 'max_angle': 50, 'symmetry': True, 'corners': 4}
    }

    def __init__(self, db: CityDatabase):
        self.db = db
        self.point_cloud = None
        self.classified_clouds = {}
        self.tree = None
        self.ground_model = None
        self.coord_transformer = None
        self.heights_in_feet = False

    def _scan_lidar_directory(self, city_dir: pathlib.Path, city_name: str, coord_system: str) -> List[pathlib.Path]:
        """
        Scan the given directory for LAZ files, read bounding boxes in the file's
        native coordinate reference system (e.g., EPSG:2927 in feet),
        and store them directly in the database without transformation.
        """
        point_cloud_files = sorted(city_dir.glob("*.laz"))
        for file_path in point_cloud_files:
            try:
                with laspy.open(str(file_path)) as f:
                    header = f.header
                    min_x, min_y, min_z = header.mins
                    max_x, max_y, max_z = header.maxs
                    point_count = header.point_count

                # Raw bounding box from the LAZ file (still in EPSG:2927 if QGIS says "Washington South")
                west = min_x
                south = min_y
                east = max_x
                north = max_y

                # Change the coord_system to the correct reference
                file_info = {
                    'file_path': str(file_path.relative_to(BASE_DIR)),
                    'city_name': city_name,
                    # If QGIS says "NAD83 / Washington South" use "EPSG:2927" (or your custom PROJ string)
                    'coord_system': "EPSG:2927",
                    'bbox_north': north,
                    'bbox_south': south,
                    'bbox_east': east,
                    'bbox_west': west,
                    'min_z': min_z,
                    'max_z': max_z,
                    'point_count': point_count,
                    'has_classifications': True,
                    'classification_counts': {},
                    'last_modified': file_path.stat().st_mtime
                }

                self.db.store_lidar_metadata(file_info)

            except Exception as exc:
                logger.exception(f"Failed to process LAZ file {file_path}: {exc}")

        return point_cloud_files

    def load_data(self, bbox: BoundingBox) -> bool:
        """
        Load LiDAR data for the given bounding box if available.
        Convert from lat/lon (EPSG:4326) to State Plane (EPSG:2927) if it's Seattle.
        """
        # If bounding box is in the Seattle area (lat ~47.59-47.65, lon ~-122.38- -122.31),
        # we'll use an EPSG:2927 transform
        if (47.59 < bbox.south < bbox.north < 47.65) \
           and (-122.38 < bbox.west < bbox.east < -122.31):
            city_dir = LIDAR_DIR / "seattle"
            city_name = "seattle"

            # Create a transformer from lat/lon (EPSG:4326) to WA State Plane North (EPSG:2927)
            # If the LAZ offset or local usage differs, replace "EPSG:2927" with a custom string
            self.coord_transformer = Transformer.from_crs(
                "EPSG:4326",
                "EPSG:2927",
                always_xy=True
            )
            self.heights_in_feet = True
            logger.info("Using Washington State Plane coordinates (EPSG:2927) for Seattle")

            # Transform bounding box corners from lat/lon to feet
            west, south = self.coord_transformer.transform(bbox.west, bbox.south)
            east, north = self.coord_transformer.transform(bbox.east, bbox.north)

            # Build an expanded bounding box in State Plane coords
            buffer_distance = 1000  # 1000 feet buffer
            self.state_plane_bbox = BoundingBox(
                north=north + buffer_distance,
                south=south - buffer_distance,
                east=east + buffer_distance,
                west=west - buffer_distance
            )

            logger.info(f"Original BBox (EPSG:4326): "
                        f"N={bbox.north:.6f}, S={bbox.south:.6f}, "
                        f"E={bbox.east:.6f}, W={bbox.west:.6f}")
            logger.info(f"Transformed BBox (EPSG:2927): "
                        f"N={self.state_plane_bbox.north:.1f}, S={self.state_plane_bbox.south:.1f}, "
                        f"E={self.state_plane_bbox.east:.1f}, W={self.state_plane_bbox.west:.1f}")

            # Query DB for LiDAR files that intersect this bounding box
            relevant_files = self.db.get_lidar_files_for_bbox(self.state_plane_bbox, "EPSG:2927")
            if not relevant_files:
                logger.info("No files found on first pass; scanning directory for new LAZ files...")
                self._scan_lidar_directory(city_dir, city_name, "EPSG:2927")
                relevant_files = self.db.get_lidar_files_for_bbox(self.state_plane_bbox, "EPSG:2927")

            if not relevant_files:
                logger.info("Still no LiDAR files found after directory scan.")
                return False

            # Convert file_path to absolute paths, load LAZ data, etc.
            pc_files = [pathlib.Path(BASE_DIR / f['file_path']) for f in relevant_files]
            if pc_files:
                logger.info(f"Loading {len(pc_files)} LAZ files for bounding box in Seattle.")
                self._load_point_clouds(pc_files, self.state_plane_bbox)
                return True
            else:
                logger.info("No valid point cloud files to load.")
                return False

        else:
            # Logic for bounding boxes outside Seattle or no LiDAR coverage
            logger.info("Outside Seattle bounding box or no coverage. Skipping LiDAR load.")
            return False

    def _load_point_clouds(self, files: List[pathlib.Path], bbox: BoundingBox):
        """Load and merge multiple LAZ/LAS files within the bounding box."""
        logger.info("Loading and merging point clouds...")
        logger.info(f"Searching in bounding box: N={bbox.north:.1f}, S={bbox.south:.1f}, E={bbox.east:.1f}, W={bbox.west:.1f}")
        classified_points = {}
        total_points = 0

        for file in files:
            try:
                logger.info(f"Processing {file.name}")
                with laspy.open(str(file)) as f:
                    las = f.read()

                # Log coordinate ranges in file
                x_min, y_min = las.x.min(), las.y.min()
                x_max, y_max = las.x.max(), las.y.max()
                logger.info(f"File bounds: N={y_max:.1f}, S={y_min:.1f}, E={x_max:.1f}, W={x_min:.1f}")

                # Filter points within bbox
                mask = ((las.x >= bbox.west) & (las.x <= bbox.east) &
                       (las.y >= bbox.south) & (las.y <= bbox.north))

                points_in_bbox = mask.sum()
                if points_in_bbox == 0:
                    logger.debug(f"No points found in bounding box for {file.name}")
                    continue
                else:
                    logger.info(f"Found {points_in_bbox} points in bounding box")

                # Get all available point attributes
                point_data = {
                    'x': las.x[mask],
                    'y': las.y[mask],
                    'z': las.z[mask],
                    'classification': las.classification[mask],
                    'intensity': las.intensity[mask] if hasattr(las, 'intensity') else None,
                    'return_number': las.return_number[mask] if hasattr(las, 'return_number') else None,
                    'number_of_returns': las.number_of_returns[mask] if hasattr(las, 'number_of_returns') else None
                }

                # Convert heights from feet to meters if needed
                if self.heights_in_feet:
                    point_data['z'] = point_data['z'] * 0.3048  # Convert feet to meters

                # Organize points by classification
                unique_classes = np.unique(point_data['classification'])
                for class_code in unique_classes:
                    class_mask = point_data['classification'] == class_code
                    points = np.vstack((
                        point_data['x'][class_mask],
                        point_data['y'][class_mask],
                        point_data['z'][class_mask]
                    )).transpose()

                    if class_code not in classified_points:
                        classified_points[class_code] = []
                    classified_points[class_code].append(points)

                total_points += len(point_data['x'])
                logger.info(f"Added {len(point_data['x'])} points from {file.name}")
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                continue

        if total_points == 0:
            logger.warning("No points found within the specified bounding box")
            self.point_cloud = None
            self.tree = None
            return

        # Merge points by classification
        self.classified_clouds = {}
        for class_code, point_lists in classified_points.items():
            if point_lists:
                self.classified_clouds[class_code] = np.vstack(point_lists)
                logger.info(f"Class {class_code}: {len(self.classified_clouds[class_code])} points")

        if not self.classified_clouds:
            logger.warning("No valid points after classification")
            self.point_cloud = None
            self.tree = None
            return

        # Create ground model from ground points
        self._create_ground_model()

        # Create KD-tree for all points
        try:
            all_points = np.vstack([points for points in self.classified_clouds.values() if len(points) > 0])
            if len(all_points) > 0:
                self.point_cloud = all_points
                self.tree = cKDTree(all_points[:, :2])
                logger.info(f"Successfully loaded {len(all_points)} total points")
            else:
                logger.warning("No valid points to create KD-tree")
                self.point_cloud = None
                self.tree = None
        except Exception as e:
            logger.error(f"Error creating point cloud: {e}")
            self.point_cloud = None
            self.tree = None

    def _create_ground_model(self):
        """Create a ground model from classified ground points."""
        logger.info("Starting ground model creation...")
        ground_points = self.classified_clouds.get(2)  # Class 2 is ground
        if ground_points is not None and len(ground_points) > 0:
            try:
                logger.info(f"Processing {len(ground_points)} ground points...")

                # Create grid for ground model
                grid_size = 1.0  # 1 meter grid for higher detail
                x_min, y_min = ground_points[:, :2].min(axis=0)
                x_max, y_max = ground_points[:, :2].max(axis=0)

                logger.info(f"Ground point bounds: X={x_min:.1f} to {x_max:.1f}, Y={y_min:.1f} to {y_max:.1f}")

                x_grid = np.arange(x_min, x_max, grid_size)
                y_grid = np.arange(y_min, y_max, grid_size)
                logger.info(f"Creating grid with size {len(x_grid)}x{len(y_grid)}")

                x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

                # Interpolate ground heights with 'linear' method for better accuracy
                logger.info("Interpolating ground heights...")
                self.ground_model = griddata(
                    ground_points[:, :2],
                    ground_points[:, 2],
                    (x_mesh, y_mesh),
                    method='linear'
                )

                self.ground_grid = {
                    'x': x_grid,
                    'y': y_grid,
                    'grid_size': grid_size
                }

                logger.info("Ground model creation complete")

            except Exception as e:
                logger.error(f"Error creating ground model: {e}")
                self.ground_model = None
                self.ground_grid = None
        else:
            logger.warning("No ground points available for ground model")
            self.ground_model = None
            self.ground_grid = None

    def get_building_geometry(self, geometry: Polygon, city_id: int, osm_id: str) -> BuildingGeometry:
        """Get detailed building geometry from LiDAR data."""
        building_geom = BuildingGeometry()

        # If no LiDAR data is available, return empty geometry
        if self.tree is None or self.point_cloud is None:
            logger.debug("No LiDAR data available for building geometry analysis")
            return building_geom

        try:
            # Check if we have cached scan data
            cached_scan = self.db.get_building_scan(city_id, osm_id)
            if cached_scan:
                logger.info(f"Using cached LiDAR scan for building {osm_id}")
                building_geom.height = cached_scan['height']
                building_geom.ground_level = cached_scan['ground_level']
                building_geom.roof_type = cached_scan['roof_type']
                building_geom.roof_angle = cached_scan['roof_angle']
                building_geom.confidence = cached_scan['confidence']
                return building_geom

            logger.info("Starting building geometry analysis...")
            # Transform geometry to State Plane coordinates if needed
            if self.coord_transformer is not None:
                logger.info("Transforming coordinates...")
                # Transform the geometry coordinates
                transformed_coords = []
                for x, y in geometry.exterior.coords:
                    tx, ty = self.coord_transformer.transform(x, y)
                    transformed_coords.append((tx, ty))
                geometry = Polygon(transformed_coords)
                logger.info("Coordinate transformation complete")

            # Get points within the building footprint with a larger buffer for tall buildings
            bounds = geometry.bounds
            buffer_distance = max(5.0, (bounds[2] - bounds[0]) * 0.1)  # At least 5m or 10% of building width
            buffered_geometry = geometry.buffer(buffer_distance)
            buffered_bounds = buffered_geometry.bounds

            logger.info(f"Building bounds: {bounds}")
            center_point = np.array([(bounds[0] + bounds[2])/2, (bounds[1] + bounds[3])/2])
            radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2 + buffer_distance
            logger.info(f"Searching points within radius {radius:.1f}m of center {center_point}")

            # Get all points within the buffer
            try:
                logger.info("Querying KD-tree...")
                indices = self.tree.query_ball_point(center_point, radius)
                points = self.point_cloud[indices] if indices else np.array([])
                logger.info(f"Found {len(points)} points in initial search")
            except Exception as e:
                logger.error(f"KD-tree query failed: {e}")
                return building_geom

            if len(points) == 0:
                logger.debug(f"No points found for building at {bounds}")
                return building_geom

            # Filter points within the actual polygon
            xy_points = points[:, :2]
            points_shapely = [Point(x, y) for x, y in xy_points]
            mask = np.array([buffered_geometry.contains(p) for p in points_shapely])
            points = points[mask]

            if len(points) == 0:
                logger.debug("No points found within building polygon")
                return building_geom

            # Get ground level
            building_geom.ground_level = self._get_ground_level(geometry)
            logger.info(f"Ground level: {building_geom.ground_level:.2f}m")

            # Remove noise and get building points
            z_values = points[:, 2]
            z_mean = np.mean(z_values)
            z_std = np.std(z_values)
            filtered_points = points[(z_values > z_mean - 3*z_std) & (z_values < z_mean + 3*z_std)]
            building_points = filtered_points[filtered_points[:, 2] > building_geom.ground_level + 0.3]

            if len(building_points) == 0:
                logger.debug(f"No points above ground level")
                return building_geom

            # Analyze building sections
            z_values = building_points[:, 2] - building_geom.ground_level
            height_range = np.max(z_values) - np.min(z_values)
            num_sections = max(1, int(height_range / 10))  # One section per ~10m of height

            # Create height sections
            section_heights = np.linspace(0, height_range, num_sections + 1)
            sections = []

            logger.info(f"Analyzing {num_sections} building sections...")

            for i in range(num_sections):
                z_min = section_heights[i]
                z_max = section_heights[i + 1]

                # Get points in this section
                section_mask = (z_values >= z_min) & (z_values < z_max)
                section_points = building_points[section_mask]

                if len(section_points) < 10:  # Skip sections with too few points
                    continue

                # Calculate section properties
                xy_points = section_points[:, :2]
                try:
                    section_poly = MultiPoint(xy_points).convex_hull
                    if section_poly.area < 1:  # Skip tiny sections
                        continue

                    section_info = {
                        'height_range': (z_min + building_geom.ground_level, z_max + building_geom.ground_level),
                        'points': section_points,
                        'area': section_poly.area,
                        'centroid': section_poly.centroid,
                        'polygon': section_poly
                    }
                    sections.append(section_info)
                except Exception as e:
                    logger.warning(f"Error processing section {i}: {e}")
                    continue

            # Analyze sections for features
            if sections:
                # Get overall building height from highest section
                top_section = sections[-1]
                building_geom.height = float(top_section['height_range'][1] - building_geom.ground_level)

                # Get roof points (top 20% of building)
                roof_section_count = max(1, int(len(sections) * 0.2))
                roof_sections = sections[-roof_section_count:]
                roof_points = np.vstack([s['points'] for s in roof_sections])

                # Analyze roof structure
                building_geom.roof_type, building_geom.roof_angle = self._analyze_roof_structure(roof_points, building_geom.ground_level)
                logger.info(f"Detected roof type: {building_geom.roof_type} with angle {building_geom.roof_angle:.1f}°")

                # Calculate confidence based on point distribution
                point_counts = [len(s['points']) for s in sections]
                height_coverage = len(sections) / num_sections
                point_density = np.mean(point_counts) / geometry.area
                building_geom.confidence = min(1.0, (height_coverage * point_density) / 10)

                # Store scan data in cache
                scan_data = {
                    'height': building_geom.height,
                    'ground_level': building_geom.ground_level,
                    'roof_type': building_geom.roof_type,
                    'roof_angle': building_geom.roof_angle,
                    'confidence': building_geom.confidence,
                    'point_count': len(building_points),
                    'points': building_points.tolist()  # Convert to list for storage
                }
                self.db.store_building_scan(city_id, osm_id, scan_data)
                logger.info(f"Cached LiDAR scan for building {osm_id}")

                logger.info(f"Building height: {building_geom.height:.1f}m")
                logger.info(f"Confidence: {building_geom.confidence:.2f}")

        except Exception as e:
            logger.error(f"Error extracting building geometry: {e}")

        return building_geom

    def _get_ground_level(self, geometry: Polygon) -> float:
        """Get ground level for a building footprint."""
        if self.ground_model is not None:
            # Get ground heights from the ground model
            centroid = geometry.centroid
            x_idx = np.searchsorted(self.ground_grid['x'], centroid.x)
            y_idx = np.searchsorted(self.ground_grid['y'], centroid.y)

            # Get average of nearby grid cells
            x_start = max(0, x_idx - 2)
            x_end = min(len(self.ground_grid['x']), x_idx + 3)
            y_start = max(0, y_idx - 2)
            y_end = min(len(self.ground_grid['y']), y_idx + 3)

            ground_heights = self.ground_model[y_start:y_end, x_start:x_end]
            return float(np.nanmean(ground_heights))

        # Fallback to using point cloud ground points
        ground_points = self.classified_clouds.get(2)  # Class 2 is ground
        if ground_points is not None:
            bounds = geometry.bounds
            min_point = np.array([bounds[0], bounds[1]])  # minx, miny
            max_point = np.array([bounds[2], bounds[3]])  # maxx, maxy
            box_bounds = np.vstack((min_point, max_point))
            indices = self.tree.query_ball_box(box_bounds)

            if indices:
                points = ground_points[indices]
                if len(points) > 0:
                    return float(np.percentile(points[:, 2], 10))

        return 0.0

    def _analyze_roof_structure(self, roof_points: np.ndarray, ground_level: float) -> Tuple[str, float]:
        """Analyze roof structure to determine type and angle."""
        try:
            from sklearn.decomposition import PCA

            # Normalize points relative to their centroid
            centroid = roof_points.mean(axis=0)
            normalized_points = roof_points - centroid

            # Perform PCA to find principal components
            pca = PCA(n_components=3)
            pca.fit(normalized_points)

            # Calculate angles between components and vertical
            vertical = np.array([0, 0, 1])
            angles = np.degrees(np.arccos(np.abs(np.dot(pca.components_, vertical))))

            # Analyze the distribution of points along the principal components
            transformed = pca.transform(normalized_points)

            # Calculate various metrics for roof classification
            height_range = np.ptp(roof_points[:, 2])
            xy_ratio = np.ptp(transformed[:, 0]) / np.ptp(transformed[:, 1])
            point_distribution = np.histogram(roof_points[:, 2], bins=20)[0]

            # Classify roof type based on metrics
            if angles[0] < 5:
                return 'flat', 0
            elif 0.9 < xy_ratio < 1.1 and angles[0] > 15:
                return 'pyramidal', float(angles[0])
            elif angles[0] > 15 and np.std(point_distribution) < np.mean(point_distribution) * 0.5:
                return 'gabled', float(angles[0])
            elif angles[0] > 60 and height_range > 5:
                return 'mansard', float(angles[0])
            elif angles[0] > 15 and np.std(point_distribution) > np.mean(point_distribution) * 0.5:
                return 'hipped', float(angles[0])

            return 'unknown', float(angles[0])

        except Exception as e:
            logger.error(f"Error analyzing roof structure: {e}")
            return 'unknown', 0
