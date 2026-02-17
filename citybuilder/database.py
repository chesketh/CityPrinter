"""CityDatabase — SQLite storage for cities, features, LiDAR metadata."""

import sqlite3
import json
import logging
import pickle
from typing import Any, Dict, List, Tuple, Optional, Union

from pyproj import Transformer
from shapely.geometry import Polygon, MultiPolygon, LineString

from .models import BoundingBox, PathManager

logger = logging.getLogger(__name__)


class CityDatabase:
    def __init__(self, db_path: str = "cities.db"):
        self.db_path = PathManager.get_data_path(db_path)
        self.init_database()

        # Set default connection settings
        self.connection_settings = {
            'timeout': 20,
            'isolation_level': None  # Autocommit mode
        }

        # Initialize database with optimized settings
        with sqlite3.connect(self.db_path, **self.connection_settings) as conn:
            conn.execute("PRAGMA busy_timeout = 10000")  # 10 second timeout
            conn.execute("PRAGMA journal_mode = WAL")    # Write-Ahead Logging
            conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes
            conn.commit()

        # Create the transformer once if we always assume EPSG:2285 for Seattle
        self.to_state_plane = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:2285",
            always_xy=True
        )

    def _get_connection(self):
        """Get a database connection with optimized settings."""
        conn = sqlite3.connect(self.db_path, **self.connection_settings)
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def init_database(self):
        """Initialize the database with necessary tables."""
        self.db_path.parent.mkdir(exist_ok=True)

        # Wait for any existing connections to close
        for _ in range(3):  # Try 3 times
            try:
                with sqlite3.connect(self.db_path, timeout=20) as conn:
                    conn.execute("PRAGMA busy_timeout = 10000")
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.execute("PRAGMA synchronous = NORMAL")

                    cursor = conn.cursor()

                    # Check if tables exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    existing_tables = {row[0] for row in cursor.fetchall()}

                    if not existing_tables:  # Only create tables if none exist
                        # Drop existing tables and indices
                        cursor.execute("DROP INDEX IF EXISTS idx_features_city_id")
                        cursor.execute("DROP INDEX IF EXISTS idx_features_type")
                        cursor.execute("DROP INDEX IF EXISTS idx_lidar_bbox")
                        cursor.execute("DROP INDEX IF EXISTS idx_log_city_id")
                        cursor.execute("DROP INDEX IF EXISTS idx_landmarks_name")
                        cursor.execute("DROP INDEX IF EXISTS idx_neighborhoods_city_id")
                        cursor.execute("DROP TABLE IF EXISTS processing_log")
                        cursor.execute("DROP TABLE IF EXISTS features")
                        cursor.execute("DROP TABLE IF EXISTS lidar_files")
                        cursor.execute("DROP TABLE IF EXISTS landmarks")
                        cursor.execute("DROP TABLE IF EXISTS neighborhoods")
                        cursor.execute("DROP TABLE IF EXISTS cities")
                        cursor.execute("DROP TABLE IF EXISTS building_scans")

                        # Create cities table
                        cursor.execute("""
                            CREATE TABLE cities (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name TEXT NOT NULL,
                                bbox_north REAL,
                                bbox_south REAL,
                                bbox_east REAL,
                                bbox_west REAL,
                                coord_system TEXT DEFAULT 'EPSG:4326',
                                has_lidar BOOLEAN DEFAULT FALSE,
                                has_osm BOOLEAN DEFAULT FALSE,
                                has_buildings BOOLEAN DEFAULT FALSE,
                                has_roads BOOLEAN DEFAULT FALSE,
                                has_water BOOLEAN DEFAULT FALSE,
                                last_updated TIMESTAMP,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)

                        # Create neighborhoods table (attached to cities)
                        cursor.execute("""
                            CREATE TABLE neighborhoods (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                city_id INTEGER NOT NULL,
                                name TEXT NOT NULL,
                                bbox_north REAL NOT NULL,
                                bbox_south REAL NOT NULL,
                                bbox_east REAL NOT NULL,
                                bbox_west REAL NOT NULL,
                                description TEXT,
                                coord_system TEXT DEFAULT 'EPSG:4326',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (city_id) REFERENCES cities (id)
                            )
                        """)

                        # Create landmarks table (for significant buildings)
                        cursor.execute("""
                            CREATE TABLE landmarks (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                city_id INTEGER NOT NULL,
                                name TEXT NOT NULL,
                                osm_id TEXT,
                                building_type TEXT,
                                height REAL,
                                ground_level REAL,
                                num_floors INTEGER,
                                year_built INTEGER,
                                architect TEXT,
                                center_lat REAL NOT NULL,
                                center_lon REAL NOT NULL,
                                bbox_radius REAL NOT NULL,
                                description TEXT,
                                coord_system TEXT DEFAULT 'EPSG:4326',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (city_id) REFERENCES cities (id)
                            )
                        """)

                        # Insert known Seattle neighborhoods
                        seattle_neighborhoods = [
                            ('Downtown Core', 47.6120, 47.5980, -122.3280, -122.3380, 'Central business district including major retail and office towers'),
                            ('South Lake Union', 47.6260, 47.6180, -122.3280, -122.3380, 'Tech hub and urban neighborhood'),
                            ('Pioneer Square', 47.6020, 47.5980, -122.3280, -122.3380, 'Historic district and original downtown')
                        ]

                        # Insert Seattle first to get its ID
                        cursor.execute("""
                            INSERT INTO cities (name, bbox_north, bbox_south, bbox_east, bbox_west)
                            VALUES ('Seattle', 47.642711, 47.598576, -122.317543, -122.371531)
                        """)
                        seattle_id = cursor.lastrowid

                        # Insert neighborhoods
                        cursor.executemany("""
                            INSERT INTO neighborhoods (city_id, name, bbox_north, bbox_south, bbox_east, bbox_west, description)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [(seattle_id, *neighborhood) for neighborhood in seattle_neighborhoods])

                        # Insert significant buildings as landmarks
                        landmarks = [
                            (seattle_id, 'Columbia Center', '123456', 'skyscraper', 284.2, 23.0, 76, 1985, 'Chester L. Lindsey',
                             47.6045, -122.3314, 300,  # Center point and 300m radius (about 3-4 blocks)
                             'Tallest building in Seattle (937 ft), distinctive stepped design with setbacks at floors 40 and 76')
                        ]

                        cursor.executemany("""
                            INSERT INTO landmarks (
                                city_id, name, osm_id, building_type, height, ground_level,
                                num_floors, year_built, architect, center_lat, center_lon,
                                bbox_radius, description
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, landmarks)

                        # Create features table
                        cursor.execute("""
                            CREATE TABLE features (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                city_id INTEGER NOT NULL,
                                feature_type TEXT NOT NULL,
                                osm_id TEXT,
                                geometry BLOB NOT NULL,
                                geometry_coord_system TEXT NOT NULL DEFAULT 'EPSG:4326',
                                properties TEXT,
                                source TEXT DEFAULT 'osm',
                                confidence REAL DEFAULT 1.0,
                                last_updated TIMESTAMP,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (city_id) REFERENCES cities (id)
                            )
                        """)

                        # Create building_scans table to cache LiDAR analysis results
                        cursor.execute("""
                            CREATE TABLE building_scans (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                city_id INTEGER NOT NULL,
                                osm_id TEXT NOT NULL,
                                height REAL,
                                ground_level REAL,
                                roof_type TEXT,
                                roof_angle REAL,
                                confidence REAL,
                                point_count INTEGER,
                                scan_data BLOB,
                                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (city_id) REFERENCES cities (id),
                                UNIQUE(city_id, osm_id)
                            )
                        """)

                        # Create lidar_files table
                        cursor.execute("""
                            CREATE TABLE lidar_files (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                file_path TEXT NOT NULL UNIQUE,
                                city_name TEXT NOT NULL,
                                coord_system TEXT NOT NULL,
                                bbox_north DOUBLE PRECISION NOT NULL,
                                bbox_south DOUBLE PRECISION NOT NULL,
                                bbox_east DOUBLE PRECISION NOT NULL,
                                bbox_west DOUBLE PRECISION NOT NULL,
                                min_z DOUBLE PRECISION,
                                max_z DOUBLE PRECISION,
                                point_count INTEGER,
                                has_classifications BOOLEAN DEFAULT FALSE,
                                classification_counts TEXT,
                                last_modified TIMESTAMP,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)

                        # Create processing_log table
                        cursor.execute("""
                            CREATE TABLE processing_log (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                city_id INTEGER NOT NULL,
                                operation TEXT NOT NULL,
                                status TEXT NOT NULL,
                                details TEXT,
                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                FOREIGN KEY (city_id) REFERENCES cities (id)
                            )
                        """)

                        # Create indices without IF NOT EXISTS
                        cursor.execute("CREATE INDEX idx_features_city_id ON features(city_id)")
                        cursor.execute("CREATE INDEX idx_features_type ON features(feature_type)")
                        cursor.execute("CREATE INDEX idx_lidar_bbox ON lidar_files(bbox_north, bbox_south, bbox_east, bbox_west)")
                        cursor.execute("CREATE INDEX idx_log_city_id ON processing_log(city_id)")
                        cursor.execute("CREATE INDEX idx_landmarks_name ON landmarks(name)")
                        cursor.execute("CREATE INDEX idx_neighborhoods_city_id ON neighborhoods(city_id)")

                        conn.commit()
                    break  # If we get here, everything worked
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:  # Don't sleep on last try
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise  # Re-raise the error if we've tried enough times

    def add_city(self, name: str, bbox: BoundingBox, coord_system: str = "EPSG:4326") -> int:
        """Add a new city to the database and return its ID."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # Check if city already exists with similar bbox
                    cursor.execute("""
                        SELECT id FROM cities
                        WHERE name = ? AND
                              ABS(bbox_north - ?) < 0.001 AND
                              ABS(bbox_south - ?) < 0.001 AND
                              ABS(bbox_east - ?) < 0.001 AND
                              ABS(bbox_west - ?) < 0.001
                    """, (name, bbox.north, bbox.south, bbox.east, bbox.west))

                    existing_id = cursor.fetchone()
                    if existing_id:
                        return existing_id[0]

                    # Add new city
                    cursor.execute("""
                        INSERT INTO cities (
                            name, bbox_north, bbox_south, bbox_east, bbox_west,
                            coord_system, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (name, bbox.north, bbox.south, bbox.east, bbox.west, coord_system))

                    city_id = cursor.lastrowid
                    self.log_operation(city_id, "create", "success", "City created")
                    conn.commit()
                    return city_id
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise
        raise Exception("Failed to add city after multiple retries")

    def store_features(self, city_id: int, feature_type: str, geometry: Union[Polygon, MultiPolygon, LineString],
                      properties: Dict, source: str = "osm", confidence: float = 1.0, osm_id: str = None,
                      geometry_coord_system: str = "EPSG:4326"):
        """Store features in the database with metadata."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # Check if feature already exists
                    cursor.execute("""
                        SELECT id FROM features
                        WHERE city_id = ? AND feature_type = ? AND osm_id = ?
                    """, (city_id, feature_type, osm_id))

                    if cursor.fetchone():
                        # Feature already exists, skip it
                        return

                    # Store feature with WKB geometry
                    cursor.execute("""
                        INSERT INTO features (
                            city_id, feature_type, osm_id, geometry, geometry_coord_system,
                            properties, source, confidence, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        city_id, feature_type, osm_id, geometry.wkb,
                        geometry_coord_system, json.dumps(properties), source, confidence
                    ))

                    # Update city processing state
                    if feature_type == 'building':
                        cursor.execute("UPDATE cities SET has_buildings = TRUE WHERE id = ?", (city_id,))
                    elif feature_type == 'road':
                        cursor.execute("UPDATE cities SET has_roads = TRUE WHERE id = ?", (city_id,))
                    elif feature_type == 'water':
                        cursor.execute("UPDATE cities SET has_water = TRUE WHERE id = ?", (city_id,))

                    cursor.execute("UPDATE cities SET last_updated = CURRENT_TIMESTAMP WHERE id = ?", (city_id,))
                    conn.commit()
                    break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise

    def store_features_batch(self, city_id: int, feature_type: str,
                             items: list, source: str = 'osm') -> int:
        """Store many features in one transaction (much faster than
        individual ``store_features`` calls).

        *items* is a list of ``(geometry, properties, confidence, osm_id)``
        tuples.  Returns the number of rows actually inserted.

        Skips the per-row duplicate check — callers should ensure
        ``clear_city_features`` was called first.
        """
        if not items:
            return 0
        rows = []
        for geometry, properties, confidence, osm_id in items:
            if geometry is None:
                continue
            rows.append((
                city_id, feature_type, osm_id, geometry.wkb,
                'projected', json.dumps(properties), source, confidence,
            ))
        if not rows:
            return 0
        for _ in range(3):
            try:
                with self._get_connection() as conn:
                    conn.execute("PRAGMA synchronous = OFF")
                    conn.executemany("""
                        INSERT OR IGNORE INTO features (
                            city_id, feature_type, osm_id, geometry,
                            geometry_coord_system, properties, source,
                            confidence, last_updated
                        ) VALUES (?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)
                    """, rows)
                    if feature_type == 'road':
                        conn.execute(
                            "UPDATE cities SET has_roads=TRUE WHERE id=?",
                            (city_id,))
                    elif feature_type == 'building':
                        conn.execute(
                            "UPDATE cities SET has_buildings=TRUE WHERE id=?",
                            (city_id,))
                    conn.execute(
                        "UPDATE cities SET last_updated=CURRENT_TIMESTAMP "
                        "WHERE id=?", (city_id,))
                    conn.commit()
                    conn.execute("PRAGMA synchronous = NORMAL")
                    return len(rows)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time as _t
                    _t.sleep(5)
                else:
                    raise
        return 0

    def restore_features_batch(self, city_id: int, features: list) -> int:
        """Bulk-restore pre-processed feature rows from cache.

        *features* is a list of (feature_type, osm_id, geometry_wkb,
        properties_json, source, confidence) tuples — the same format
        returned by ``get_city_features()``.
        """
        if not features:
            return 0
        rows = [(city_id, ft, osm_id, geom_wkb, 'projected',
                 props_json, src, conf)
                for ft, osm_id, geom_wkb, props_json, src, conf in features]
        has_buildings = any(ft == 'building' for ft, *_ in features)
        has_roads = any(ft == 'road' for ft, *_ in features)
        has_water = any(ft == 'water' for ft, *_ in features)
        for _ in range(3):
            try:
                with self._get_connection() as conn:
                    conn.execute("PRAGMA synchronous = OFF")
                    conn.executemany("""
                        INSERT OR IGNORE INTO features (
                            city_id, feature_type, osm_id, geometry,
                            geometry_coord_system, properties, source,
                            confidence, last_updated
                        ) VALUES (?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)
                    """, rows)
                    if has_buildings:
                        conn.execute(
                            "UPDATE cities SET has_buildings=TRUE WHERE id=?",
                            (city_id,))
                    if has_roads:
                        conn.execute(
                            "UPDATE cities SET has_roads=TRUE WHERE id=?",
                            (city_id,))
                    if has_water:
                        conn.execute(
                            "UPDATE cities SET has_water=TRUE WHERE id=?",
                            (city_id,))
                    conn.execute(
                        "UPDATE cities SET has_osm=TRUE, "
                        "last_updated=CURRENT_TIMESTAMP WHERE id=?",
                        (city_id,))
                    conn.commit()
                    conn.execute("PRAGMA synchronous = NORMAL")
                    return len(rows)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time as _t
                    _t.sleep(5)
                else:
                    raise
        return 0

    def clear_city_features(self, city_id: int) -> None:
        """Delete all features for a city so it can be rebuilt from scratch."""
        for _ in range(3):
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM features WHERE city_id = ?", (city_id,))
                    deleted = cursor.rowcount
                    # Reset processing flags
                    cursor.execute("""
                        UPDATE cities SET has_buildings = FALSE, has_roads = FALSE,
                               has_water = FALSE, has_osm = FALSE,
                               last_updated = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (city_id,))
                    conn.commit()
                    logger.info(f"Cleared {deleted} features for city_id={city_id}")
                    return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)
                else:
                    raise

    def update_city_state(self, city_id: int, **kwargs):
        """Update city processing state."""
        valid_fields = {'has_lidar', 'has_osm', 'has_buildings', 'has_roads', 'has_water'}
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}

        if not updates:
            return

        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    set_clause = ", ".join(f"{k} = ?" for k in updates)
                    query = f"UPDATE cities SET {set_clause}, last_updated = CURRENT_TIMESTAMP WHERE id = ?"
                    cursor.execute(query, list(updates.values()) + [city_id])
                    conn.commit()
                    break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise

    def log_operation(self, city_id: int, operation: str, status: str, details: str = None):
        """Log a processing operation."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO processing_log (city_id, operation, status, details)
                        VALUES (?, ?, ?, ?)
                    """, (city_id, operation, status, details))
                    conn.commit()
                    break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise

    def get_city_state(self, city_id: int) -> Dict:
        """Get the current processing state of a city."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT name, bbox_north, bbox_south, bbox_east, bbox_west,
                               coord_system, has_lidar, has_osm, has_buildings,
                               has_roads, has_water, last_updated
                        FROM cities WHERE id = ?
                    """, (city_id,))
                    row = cursor.fetchone()
                    if row:
                        return {
                            'name': row[0],
                            'bbox': BoundingBox(
                                north=row[1],
                                south=row[2],
                                east=row[3],
                                west=row[4]
                            ),
                            'coord_system': row[5],
                            'has_lidar': row[6],
                            'has_osm': row[7],
                            'has_buildings': row[8],
                            'has_roads': row[9],
                            'has_water': row[10],
                            'last_updated': row[11]
                        }
                    return None
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise
        return None

    def get_city_features(self, city_id: int, feature_type: str = None) -> List[Tuple]:
        """Retrieve features for a given city with optional type filter."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    if feature_type:
                        cursor.execute("""
                            SELECT feature_type, osm_id, geometry, properties, source, confidence
                            FROM features
                            WHERE city_id = ? AND feature_type = ?
                        """, (city_id, feature_type))
                    else:
                        cursor.execute("""
                            SELECT feature_type, osm_id, geometry, properties, source, confidence
                            FROM features
                            WHERE city_id = ?
                        """, (city_id,))
                    return cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise
        return []

    def get_processing_history(self, city_id: int) -> List[Dict]:
        """Get the processing history for a city."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT operation, status, details, timestamp
                        FROM processing_log
                        WHERE city_id = ?
                        ORDER BY timestamp DESC
                    """, (city_id,))
                    return [{
                        'operation': row[0],
                        'status': row[1],
                        'details': row[2],
                        'timestamp': row[3]
                    } for row in cursor.fetchall()]
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise
        return []

    def store_building_scan(self, city_id: int, osm_id: str, scan_data: Dict) -> bool:
        """Store LiDAR scan results for a building."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO building_scans (
                            city_id, osm_id, height, ground_level, roof_type,
                            roof_angle, confidence, point_count, scan_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        city_id,
                        osm_id,
                        scan_data.get('height'),
                        scan_data.get('ground_level'),
                        scan_data.get('roof_type'),
                        scan_data.get('roof_angle'),
                        scan_data.get('confidence', 0.0),
                        scan_data.get('point_count', 0),
                        pickle.dumps(scan_data.get('points', []))
                    ))
                    conn.commit()
                    return True
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise
        return False

    def get_building_scan(self, city_id: int, osm_id: str) -> Optional[Dict]:
        """Retrieve cached LiDAR scan results for a building."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT height, ground_level, roof_type, roof_angle,
                               confidence, point_count, scan_data
                        FROM building_scans
                        WHERE city_id = ? AND osm_id = ?
                    """, (city_id, osm_id))
                    row = cursor.fetchone()
                    if row:
                        return {
                            'height': row[0],
                            'ground_level': row[1],
                            'roof_type': row[2],
                            'roof_angle': row[3],
                            'confidence': row[4],
                            'point_count': row[5],
                            'points': pickle.loads(row[6]) if row[6] else []
                        }
                    return None
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise
        return None

    def store_lidar_metadata(self, file_info: Dict) -> int:
        """Store LiDAR file metadata in the database."""
        for _ in range(3):  # Try 3 times
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO lidar_files (
                            file_path, city_name, coord_system,
                            bbox_north, bbox_south, bbox_east, bbox_west,
                            min_z, max_z, point_count, has_classifications,
                            classification_counts, last_modified
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_info['file_path'],
                        file_info['city_name'],
                        file_info['coord_system'],
                        file_info['bbox_north'],
                        file_info['bbox_south'],
                        file_info['bbox_east'],
                        file_info['bbox_west'],
                        file_info['min_z'],
                        file_info['max_z'],
                        file_info['point_count'],
                        file_info['has_classifications'],
                        json.dumps(file_info.get('classification_counts', {})),
                        file_info['last_modified']
                    ))
                    conn.commit()
                    return cursor.lastrowid
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and _ < 2:
                    import time
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    raise
        raise Exception("Failed to store LiDAR metadata after multiple retries")

    def get_lidar_files_for_bbox(self, bbox: BoundingBox, user_coord_system: str = "EPSG:4326") -> List[Dict]:
        """
        Retrieve LiDAR files that intersect the given bounding box.
        We assume we've stored each file's bounding box in EPSG:2927 (feet).
        If the incoming bbox is lat/lon, we transform it to EPSG:2927,
        so the query matches properly.
        """
        if user_coord_system == "EPSG:4326":
            transformer = Transformer.from_crs(
                "EPSG:4326",
                "EPSG:2927",
                always_xy=True
            )
            min_x, min_y = transformer.transform(bbox.west, bbox.south)
            max_x, max_y = transformer.transform(bbox.east, bbox.north)
            search_bbox = BoundingBox(north=max_y, south=min_y, east=max_x, west=min_x)
            db_coord_system = "EPSG:2927"
        else:
            search_bbox = bbox
            db_coord_system = user_coord_system

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Example small buffer in feet
            buffer_ft = 200.0
            query = """
                SELECT file_path, city_name, coord_system,
                       bbox_north, bbox_south, bbox_east, bbox_west,
                       min_z, max_z, point_count, has_classifications,
                       classification_counts, last_modified
                FROM lidar_files
                WHERE coord_system = ?
                  AND bbox_north >= ?
                  AND bbox_south <= ?
                  AND bbox_east >= ?
                  AND bbox_west <= ?
            """

            params = [
                db_coord_system,
                search_bbox.south - buffer_ft,
                search_bbox.north + buffer_ft,
                search_bbox.west - buffer_ft,
                search_bbox.east + buffer_ft
            ]

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                logger.info("No LiDAR files found matching that bounding box.")
                return []

            results = []
            for row in rows:
                results.append({
                    'file_path': row[0],
                    'city_name': row[1],
                    'coord_system': row[2],
                    'bbox_north': float(row[3]),
                    'bbox_south': float(row[4]),
                    'bbox_east': float(row[5]),
                    'bbox_west': float(row[6]),
                    'min_z': row[7],
                    'max_z': row[8],
                    'point_count': row[9],
                    'has_classifications': row[10],
                    'classification_counts': json.loads(row[11]) if row[11] else {},
                    'last_modified': row[12]
                })

            return results
