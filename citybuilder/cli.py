"""Click CLI commands for CityBuilder."""

import asyncio
import logging
from typing import Union

import click

from .builder import CityBuilder
from .models import BoundingBox

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """CityBuilder CLI for generating 3D city models from OpenStreetMap data."""
    pass


@cli.command()
@click.argument('city_name')
@click.option('--output', '-o', default='city.stl', help='Output STL file path')
def build(city_name: str, output: str):
    """Build a 3D city model from a city name."""
    builder = CityBuilder()
    asyncio.run(async_build(builder, city_name, output))


@cli.command()
@click.argument('north', type=float)
@click.argument('south', type=float)
@click.argument('east', type=float)
@click.argument('west', type=float)
@click.option('--output', '-o', default='city.stl', help='Output STL file path')
def build_bbox(north: float, south: float, east: float, west: float, output: str):
    """Build a 3D city model from a bounding box."""
    bbox = BoundingBox(north=north, south=south, east=east, west=west)
    builder = CityBuilder()
    asyncio.run(async_build(builder, bbox, output))


@cli.command()
@click.argument('north', type=float)
@click.argument('south', type=float)
@click.argument('east', type=float)
@click.argument('west', type=float)
@click.option('--output', '-o', default='city.glb', help='Output GLB file path')
@click.option('--name', '-n', default='city', help='Name prefix for PLY layer files')
@click.option('--scale', '-s', default=1.0, help='Scale factor (e.g. 0.001 for m→mm)')
def print_bbox(north: float, south: float, east: float, west: float,
               output: str, name: str, scale: float):
    """Generate 3D-printable PLY layers from a bounding box.

    Outputs separate colored PLY files for each layer (base, terrain,
    buildings, roads, water, vegetation) that fit together for
    multi-color 3D printing.
    """
    bbox = BoundingBox(north=north, south=south, east=east, west=west)
    builder = CityBuilder()
    asyncio.run(async_print(builder, bbox, output, name, scale))


async def async_build(builder: CityBuilder, location: Union[str, BoundingBox], output: str):
    """Async helper function for building city models."""
    try:
        city_id = await builder.process_city(location)
        builder.generate_stl(city_id, output)
        logger.info(f"Successfully generated 3D model: {output}")
    except Exception as e:
        logger.error(f"Error building city model: {e}")
        raise click.ClickException(str(e))


async def async_print(builder: CityBuilder, bbox: BoundingBox,
                      output: str, name: str, scale: float):
    """Async helper for PLY print generation."""
    try:
        def _progress(pct, msg):
            click.echo(f"[{pct:3.0f}%] {msg}")

        city_id = await builder.process_city(bbox, progress_callback=_progress)
        result = builder.generate_ply(city_id, output, name=name,
                                      scale=scale, progress_callback=_progress)

        click.echo(f"\n{'='*50}")
        click.echo(f"Generated {len(result['layers'])} print layers:")
        for layer in result['layers']:
            wt = '✓' if layer['watertight'] else '✗'
            click.echo(f"  [{wt}] {layer['file']}: {layer['faces']} faces, "
                       f"color={layer['color_rgb']}")
        click.echo(f"\nManifest: {result['manifest_path']}")
        click.echo(f"{'='*50}")
    except Exception as e:
        logger.error(f"Error generating print layers: {e}")
        raise click.ClickException(str(e))
