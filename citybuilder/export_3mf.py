"""3MF multi-body export for multi-color 3D printing.

Generates a single .3mf file containing one solid body per layer,
each with a distinct material color. Compatible with PrusaSlicer and
Bambu Studio for multi-material / filament-swap printing.

Usage:
    from citybuilder.export_3mf import generate_3mf
    result = generate_3mf(glb_path, output_path, name)
"""

import io
import logging
import pathlib
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional

import numpy as np
import trimesh

from .ply import (PRINT_LAYERS, _solidify_surface, _make_watertight,
                  _assign_solid_color)

logger = logging.getLogger(__name__)

# 3MF namespace constants
NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_MAT  = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0"
    Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _mesh_to_3mf_object(mesh: trimesh.Trimesh, obj_id: int,
                        name: str, color_hex: str,
                        mat_id: int) -> ET.Element:
    """Build a 3MF <object> element for a single mesh body."""
    obj = ET.Element('object', {
        'id': str(obj_id),
        'name': name,
        'type': 'model',
        'm:pid': str(mat_id),   # material group id
        'm:pindex': '0',        # index into that group
    })
    mesh_el = ET.SubElement(obj, 'mesh')

    verts_el = ET.SubElement(mesh_el, 'vertices')
    for v in mesh.vertices:
        ET.SubElement(verts_el, 'vertex', {
            'x': f'{v[0]:.6f}',
            'y': f'{v[1]:.6f}',
            'z': f'{v[2]:.6f}',
        })

    tris_el = ET.SubElement(mesh_el, 'triangles')
    for f in mesh.faces:
        ET.SubElement(tris_el, 'triangle', {
            'v1': str(f[0]),
            'v2': str(f[1]),
            'v3': str(f[2]),
        })

    return obj


def _rgb_to_hex(r, g, b) -> str:
    return f'#{r:02X}{g:02X}{b:02X}FF'


def generate_3mf(glb_path: str, output_path: str,
                 name: str = "city",
                 scale: float = 1.0,
                 progress_callback=None) -> dict:
    """Generate a multi-body 3MF from a GLB for multi-color printing.

    Each geographic layer (terrain, buildings, roads, water, vegetation,
    details) becomes a separate body in the 3MF with its own material color.
    All bodies are solid and watertight. Load the result directly into
    PrusaSlicer or Bambu Studio — each body will be assigned to a filament.

    Parameters
    ----------
    glb_path : str
        Source GLB file from CityPrinter pipeline.
    output_path : str
        Output .3mf file path.
    name : str
        Model name.
    scale : float
        Scale factor (1.0 = meters; use 0.001 for mm if needed).
    progress_callback : callable
        Optional (pct, msg).

    Returns
    -------
    dict with output_path, layers, total_faces, watertight_count.
    """
    def _prog(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    t0 = time.perf_counter()
    _prog(0, "Loading GLB...")

    scene = trimesh.load(glb_path, force='scene')
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(geometry={'model': scene})

    # Build group_name → color map
    group_to_color = {}
    group_to_layer = {}
    for layer_name, layer_def in PRINT_LAYERS.items():
        for grp in layer_def.get('glb_groups', []):
            group_to_color[grp] = layer_def['color']
            group_to_layer[grp] = layer_name

    _prog(5, "Processing layers...")
    total_layers = len(PRINT_LAYERS)

    # Merge GLB groups into layer bodies
    layer_meshes = {}  # layer_name → trimesh.Trimesh
    for geom_name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        layer_name = group_to_layer.get(geom_name, 'details')
        if layer_name not in layer_meshes:
            layer_meshes[layer_name] = []
        layer_meshes[layer_name].append(geom.copy())

    # Process each layer into a solid watertight body
    bodies = []  # list of {name, mesh, color, layer_def}
    for i, (layer_name, layer_def) in enumerate(
            sorted(PRINT_LAYERS.items(), key=lambda x: x[1]['order'])):

        if layer_name == 'base':
            continue  # base plate is optional, skip for now

        pct = 5 + int(70 * i / total_layers)
        _prog(pct, f"Solidifying {layer_name}...")

        parts = layer_meshes.get(layer_name, [])
        if not parts:
            logger.info(f"  {layer_name}: no geometry, skipping")
            continue

        # Merge parts
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0].copy()

        # Solidify open surfaces
        if not mesh.is_watertight:
            mesh = _solidify_surface(mesh, thickness=2.0)

        # Repair
        mesh = _make_watertight(mesh)
        if mesh is None or len(mesh.faces) == 0:
            logger.warning(f"  {layer_name}: empty after repair, skipping")
            continue

        # Scale
        if scale != 1.0:
            mesh.apply_scale(scale)

        color = layer_def['color']
        logger.info(f"  {layer_name}: {len(mesh.faces)} faces, "
                    f"watertight={mesh.is_watertight}, color={color}")

        bodies.append({
            'name': layer_name,
            'mesh': mesh,
            'color': color,
            'watertight': mesh.is_watertight,
            'faces': len(mesh.faces),
        })

    if not bodies:
        raise ValueError("No geometry produced")

    _prog(80, "Building 3MF...")

    # ── Build 3MF XML ──────────────────────────────────────────────
    ET.register_namespace('', NS_CORE)
    ET.register_namespace('m', NS_MAT)
    ET.register_namespace('p', NS_PROD)

    model = ET.Element('model', {
        'xmlns': NS_CORE,
        'xmlns:m': NS_MAT,
        'xmlns:p': NS_PROD,
        'unit': 'millimeter',
        'xml:lang': 'en-US',
    })

    resources = ET.SubElement(model, 'resources')

    # Add material color groups — one group per body
    mat_id_start = 100
    obj_id_start = 1

    for i, body in enumerate(bodies):
        mat_id = mat_id_start + i
        r, g, b = body['color']
        color_hex = _rgb_to_hex(r, g, b)

        # Material color group
        cg = ET.SubElement(resources, 'm:colorgroup', {'id': str(mat_id)})
        ET.SubElement(cg, 'm:color', {'color': color_hex})

    # Add mesh objects
    build_items = []
    for i, body in enumerate(bodies):
        obj_id = obj_id_start + i
        mat_id = mat_id_start + i
        obj_el = _mesh_to_3mf_object(
            body['mesh'], obj_id, body['name'],
            _rgb_to_hex(*body['color']), mat_id)
        resources.append(obj_el)
        build_items.append(obj_id)

    # Build section
    build = ET.SubElement(model, 'build')
    for obj_id in build_items:
        ET.SubElement(build, 'item', {
            'objectid': str(obj_id),
            'transform': '1 0 0 0 1 0 0 0 1 0 0 0',  # identity
        })

    # Serialize XML
    xml_bytes = ET.tostring(model, encoding='utf-8', xml_declaration=True)

    # ── Write 3MF zip ──────────────────────────────────────────────
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(out), 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', CONTENT_TYPES)
        zf.writestr('_rels/.rels', RELS)
        zf.writestr('3D/3dmodel.model', xml_bytes)

    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / 1024 / 1024
    watertight_count = sum(1 for b in bodies if b['watertight'])
    total_faces = sum(b['faces'] for b in bodies)

    result = {
        'output_path': str(out),
        'layers': [{'name': b['name'], 'color': b['color'],
                    'faces': b['faces'], 'watertight': b['watertight']}
                   for b in bodies],
        'total_faces': total_faces,
        'watertight_count': watertight_count,
        'total_layers': len(bodies),
        'size_mb': round(size_mb, 2),
        'elapsed_seconds': round(elapsed, 1),
    }

    logger.info(f"3MF: {len(bodies)} bodies, {total_faces:,} faces, "
                f"{watertight_count}/{len(bodies)} watertight, "
                f"{size_mb:.1f} MB, {elapsed:.1f}s")
    logger.info(f"Output: {out}")

    _prog(100, "3MF ready!")
    return result
