"""AIHelper — GPT-4 powered city/landmark lookup with database caching."""

import json
import math
import logging

from openai import OpenAI
from typing import Dict

from .database import CityDatabase

logger = logging.getLogger(__name__)


class AIHelper:
    def __init__(self, db: CityDatabase):
        self.client = OpenAI()
        self.model = "gpt-4"
        self.db = db

    async def get_city_info(self, city_name: str) -> Dict:
        """Query database or AI to get information about a city's downtown area."""
        try:
            # First check if this is a landmark building
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT center_lat, center_lon, bbox_radius, description
                    FROM landmarks
                    WHERE LOWER(name) = LOWER(?)
                """, (city_name,))

                result = cursor.fetchone()
                if result:
                    logger.info(f"Found landmark building {city_name}: {result[3]}")
                    center_lat, center_lon, radius = result[0], result[1], result[2]
                    # Convert radius from meters to degrees (approximate)
                    lat_radius = radius / 111320.0  # 1 degree latitude ≈ 111.32 km
                    lon_radius = radius / (111320.0 * math.cos(math.radians(center_lat)))
                    return {
                        "north": center_lat + lat_radius,
                        "south": center_lat - lat_radius,
                        "east": center_lon + lon_radius,
                        "west": center_lon - lon_radius
                    }

                # If not a landmark, check if it's a neighborhood
                cursor.execute("""
                    SELECT bbox_north, bbox_south, bbox_east, bbox_west, description
                    FROM neighborhoods
                    WHERE LOWER(name) = LOWER(?)
                """, (city_name,))

                result = cursor.fetchone()
                if result:
                    logger.info(f"Found neighborhood {city_name}: {result[4]}")
                    return {
                        "north": result[0],
                        "south": result[1],
                        "east": result[2],
                        "west": result[3]
                    }

            # If not in database, use AI to get coordinates
            logger.info(f"Location {city_name} not found in database, querying AI...")
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "You are a geographic expert. Provide bounding box coordinates."
                },
                {
                    "role": "user",
                    "content": f"What are the bounding box coordinates for {city_name}? "
                             f"Respond in JSON format with keys: north, south, east, west"
                }]
            )

            bbox_data = json.loads(completion.choices[0].message.content)

            # Store the AI-provided coordinates in neighborhoods table
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                # Get or create city
                cursor.execute("SELECT id FROM cities WHERE LOWER(name) = LOWER(?)", (city_name.split()[0],))
                city_result = cursor.fetchone()
                if not city_result:
                    cursor.execute("""
                        INSERT INTO cities (name, bbox_north, bbox_south, bbox_east, bbox_west)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        city_name.split()[0],
                        bbox_data["north"],
                        bbox_data["south"],
                        bbox_data["east"],
                        bbox_data["west"]
                    ))
                    city_id = cursor.lastrowid
                else:
                    city_id = city_result[0]

                # Store as neighborhood
                cursor.execute("""
                    INSERT INTO neighborhoods (
                        city_id, name, bbox_north, bbox_south, bbox_east, bbox_west, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    city_id,
                    city_name,
                    bbox_data["north"],
                    bbox_data["south"],
                    bbox_data["east"],
                    bbox_data["west"],
                    f"AI-generated bounding box for {city_name}"
                ))
                conn.commit()

            return bbox_data

        except Exception as e:
            logger.error(f"Error getting location info: {e}")
            # For any error, return a default bounding box
            return {
                "north": 0.1,
                "south": -0.1,
                "east": 0.1,
                "west": -0.1
            }
