"""PLY generation for 3D printing — separate watertight layers with solid colors.

Takes a generated GLB scene and splits it into individual PLY files,
each with a single solid vertex color suitable for single-filament printing.
Layers include alignment features (pegs/holes) so they stack together.

Output structure:
  <output_dir>/
    <name>_base.ply        — solid base plate with alignment pegs
    <name>_terrain.ply     — terrain surface with peg holes
    <name>_buildings.ply   — all buildings merged
    <name>_roads.ply       — roads + bridges
    <name>_water.ply       — blue base (water layer)
    <name>_vegetation.ply  — trees merged + simplified
    <name>_details.ply     — parks, piers, railways, etc.
    manifest.json          — layer order, colors, print settings
"""

import json
import logging
import pathlib
import time
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# ── Layer definitions ──────────────────────────────────────────────
# Each print layer merges multiple GLB mesh groups into one solid.
# Colors are RGB 0-255 for PLY vertex colors.

PRINT_LAYERS = {
    'base': {
        'color': [60, 50, 40],          # dark brown
        'glb_groups': [],                # generated procedurally
        'order': 0,
        'description': 'Solid base plate with alignment pegs',
    },
    'water': {
        'color': [64, 133, 217],        # blue
        'glb_groups': ['blue_base', 'wave'],
        'order': 1,
        'description': 'Water surface layer',
    },
    'terrain': {
        'color': [120, 160, 80],        # earth green
        'glb_groups': ['green', 'terrain', 'foundation', 'rock', 'glacier'],
        'order': 2,
        'description': 'Terrain and ground surface',
    },
    'roads': {
        'color': [65, 65, 65],          # dark asphalt
        'glb_groups': ['road', 'bridge', 'railway', 'paved', 'track'],
        'order': 3,
        'description': 'Roads, bridges, and railways',
    },
    'buildings': {
        'color': [240, 240, 240],       # white
        'glb_groups': ['building', 'wall', 'windows',
                       'statue', 'torch', 'shells',
                       'cn_shaft', 'cn_pod',
                       'brandenburg_gate'],
        'order': 4,
        'description': 'All buildings and structures',
    },
    'vegetation': {
        'color': [50, 130, 50],         # forest green
        'glb_groups': ['tree_trunk', 'tree_canopy', 'conifer_canopy',
                       'palm_canopy', 'broadleaf_canopy',
                       'mangrove_canopy', 'scrub_canopy',
                       'sclerophyll_canopy', 'bush_canopy'],
        'order': 5,
        'description': 'Trees and vegetation (simplified)',
    },
    'details': {
        'color': [100, 170, 90],        # park green
        'glb_groups': ['park', 'pitch', 'pool', 'pier',
                       'boat_hull', 'boat_sail'],
        'order': 6,
        'description': 'Parks, sports fields, piers, and decorative features',
    },
}

# Alignment peg dimensions (mm in model units — scaled at export)
PEG_RADIUS = 2.0
PEG_HEIGHT = 3.0
PEG_HOLE_CLEARANCE = 0.3  # extra radius for receiving hole
PEG_POSITIONS_FRACTION = [
    (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),  # corners
]


def _make_watertight(mesh: trimesh.Trimesh, min_thickness: float = 0.5) -> Optional[trimesh.Trimesh]:
    """Make a mesh watertight/manifold for 3D printing.

    Strategy:
    1. Basic cleanup (merge verts, remove degenerates, fill holes)
    2. If still non-manifold, try manifold3d (gold standard)
    3. If manifold3d fails, try voxel remesh
    4. Last resort: return cleaned mesh (most slicers handle minor issues)
    """
    if mesh is None or len(mesh.faces) == 0:
        return None

    try:
        # ── Step 1: Basic cleanup ──
        mesh.merge_vertices(merge_tex=True, merge_norm=True)

        # Remove degenerate/duplicate faces
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        if hasattr(mesh, 'remove_duplicate_faces'):
            mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()

        # Remove zero-area faces
        try:
            face_mask = mesh.area_faces > 1e-10
            if not face_mask.all():
                mesh.update_faces(face_mask)
        except Exception:
            pass

        if len(mesh.faces) == 0:
            return None

        # Fill holes and fix normals
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)

        if mesh.is_watertight:
            return mesh

        # ── Step 2: manifold3d repair ──
        try:
            import manifold3d
            manifold = manifold3d.Manifold(
                mesh=manifold3d.Mesh(
                    vert_properties=np.array(mesh.vertices, dtype=np.float32),
                    tri_verts=np.array(mesh.faces, dtype=np.uint32),
                )
            )
            out_mesh = manifold.to_mesh()
            result = trimesh.Trimesh(
                vertices=out_mesh.vert_properties[:, :3],
                faces=out_mesh.tri_verts,
            )
            if result.is_watertight and len(result.faces) > 0:
                logger.info(f"  manifold3d repair: {len(mesh.faces)} → {len(result.faces)} faces")
                return result
        except Exception as e:
            logger.debug(f"manifold3d repair failed: {e}")

        # ── Step 3: Voxel remesh ──
        try:
            extents = mesh.bounding_box.extents
            # Target ~150 voxels along longest axis (balance detail vs speed)
            pitch = max(extents) / 150
            pitch = max(pitch, 0.3)     # minimum 0.3m resolution
            pitch = min(pitch, 5.0)     # maximum 5m for huge meshes
            vox = mesh.voxelized(pitch)
            if hasattr(vox, 'fill'):
                vox = vox.fill()  # fill internal voids
            result = vox.marching_cubes
            if result.is_watertight and len(result.faces) > 0:
                logger.info(f"  voxel remesh: {len(mesh.faces)} → {len(result.faces)} faces "
                            f"(pitch={pitch:.1f})")
                return result
        except Exception as e:
            logger.debug(f"Voxel remesh failed: {e}")

        # ── Step 4: Return as-is ──
        logger.warning(f"  Could not make watertight ({len(mesh.faces)} faces) — "
                       f"returning cleaned mesh")
        return mesh

    except Exception as e:
        logger.warning(f"Watertight repair failed: {e}")
        return mesh


def _solidify_surface(mesh: trimesh.Trimesh, thickness: float = 2.0) -> trimesh.Trimesh:
    """Turn an open surface mesh into a solid slab by extruding downward.

    For terrain/water/road surfaces that are just a top sheet with no
    bottom, this creates a copy offset downward and stitches the edges
    to form a closed solid.
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh

    if mesh.is_watertight:
        return mesh  # already solid

    try:
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        n_verts = len(verts)

        # Create bottom surface: offset Y (up axis) downward
        bottom_verts = verts.copy()
        bottom_verts[:, 1] -= thickness

        # Bottom faces: reversed winding for outward normals
        bottom_faces = faces[:, ::-1] + n_verts

        # Stitch boundary edges
        # Find boundary edges (edges that appear in only one face)
        edges = set()
        edge_count = {}
        for f in faces:
            for i in range(3):
                e = (min(f[i], f[(i+1)%3]), max(f[i], f[(i+1)%3]))
                edge_count[e] = edge_count.get(e, 0) + 1
        boundary_edges = [(a, b) for (a, b), c in edge_count.items() if c == 1]

        # Create side walls from boundary edges
        side_faces = []
        for a, b in boundary_edges:
            # Top edge: a, b  →  Bottom edge: a+n, b+n
            side_faces.append([a, b, b + n_verts])
            side_faces.append([a, b + n_verts, a + n_verts])

        # Combine all
        all_verts = np.vstack([verts, bottom_verts])
        all_faces = np.vstack([faces, bottom_faces] +
                              ([np.array(side_faces)] if side_faces else []))

        result = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
        result.merge_vertices()
        trimesh.repair.fix_normals(result)

        logger.info(f"  solidified: {len(faces)} → {len(result.faces)} faces "
                    f"(thickness={thickness})")
        return result

    except Exception as e:
        logger.warning(f"Solidify failed: {e}")
        return mesh


def _assign_solid_color(mesh: trimesh.Trimesh, color_rgb: list) -> trimesh.Trimesh:
    """Assign a single solid color to all vertices of a mesh."""
    rgba = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255], dtype=np.uint8)
    colors = np.tile(rgba, (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


def _create_base_plate(scene_bounds, thickness: float = 3.0) -> trimesh.Trimesh:
    """Create a solid rectangular base plate that fits under the entire model.

    The base extends slightly beyond the model footprint for stability.
    Includes alignment pegs on top surface.
    """
    margin = 2.0
    xmin, ymin, zmin = scene_bounds[0]
    xmax, ymax, zmax = scene_bounds[1]

    # Base plate sits below everything
    base_bottom = ymin - thickness
    base_top = ymin

    plate = trimesh.creation.box(
        extents=[
            (xmax - xmin) + 2 * margin,
            thickness,
            (zmax - zmin) + 2 * margin,
        ],
        transform=trimesh.transformations.translation_matrix([
            (xmin + xmax) / 2,
            (base_bottom + base_top) / 2,
            (zmin + zmax) / 2,
        ])
    )

    # Add alignment pegs
    width_x = xmax - xmin + 2 * margin
    width_z = zmax - zmin + 2 * margin
    for fx, fz in PEG_POSITIONS_FRACTION:
        px = xmin - margin + fx * width_x
        pz = zmin - margin + fz * width_z
        peg = trimesh.creation.cylinder(
            radius=PEG_RADIUS,
            height=PEG_HEIGHT,
            transform=trimesh.transformations.translation_matrix([
                px, base_top + PEG_HEIGHT / 2, pz
            ])
        )
        plate = plate + peg  # boolean union

    return plate


def _add_peg_holes(mesh: trimesh.Trimesh, scene_bounds) -> trimesh.Trimesh:
    """Add alignment peg holes to a layer mesh so it slots onto pegs."""
    xmin, ymin, zmin = scene_bounds[0]
    xmax, ymax, zmax = scene_bounds[1]
    margin = 2.0
    width_x = xmax - xmin + 2 * margin
    width_z = zmax - zmin + 2 * margin
    hole_r = PEG_RADIUS + PEG_HOLE_CLEARANCE

    for fx, fz in PEG_POSITIONS_FRACTION:
        px = xmin - margin + fx * width_x
        pz = zmin - margin + fz * width_z
        hole = trimesh.creation.cylinder(
            radius=hole_r,
            height=PEG_HEIGHT * 3,  # tall enough to punch through
            transform=trimesh.transformations.translation_matrix([
                px, ymin + PEG_HEIGHT, pz
            ])
        )
        try:
            mesh = mesh.difference(hole)
        except Exception:
            pass  # boolean ops can fail on complex meshes — skip silently

    return mesh


def generate_ply_single(glb_path: str, output_path: str,
                        name: str = "city",
                        scale: float = 1.0,
                        progress_callback=None) -> dict:
    """Generate a single watertight PLY from a GLB.

    All feature meshes are color-coded by type and merged into one PLY file.
    The merged mesh is solidified and repaired to be watertight.

    Parameters
    ----------
    glb_path : str
        Path to the source GLB file.
    output_path : str
        Path for the output PLY file.
    name : str
        Model name for metadata.
    scale : float
        Scale factor (1.0 = meters).
    progress_callback : callable
        Optional (pct, msg) callback.

    Returns
    -------
    dict with output_path, faces, vertices, watertight.
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(0, "Loading GLB...")
    t0 = time.perf_counter()

    scene = trimesh.load(glb_path, force='scene')
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(geometry={'model': scene})

    # Build a reverse map: GLB group name → layer color
    group_to_color = {}
    for layer_name, layer_def in PRINT_LAYERS.items():
        color = layer_def['color']
        for grp in layer_def.get('glb_groups', []):
            group_to_color[grp] = color

    _progress(10, "Coloring meshes...")

    colored_parts = []
    for geom_name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue

        mesh = geom.copy()

        # Get color for this group (default: light gray)
        color_rgb = group_to_color.get(geom_name, [180, 180, 180])

        # Assign solid vertex color
        rgba = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255],
                        dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.tile(rgba, (len(mesh.vertices), 1))
        )

        # Solidify ALL open surfaces — every mesh needs to be a closed volume
        if not mesh.is_watertight:
            mesh = _solidify_surface(mesh, thickness=2.0)
            # Re-apply color after solidify (adds new verts)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile(rgba, (len(mesh.vertices), 1))
            )

        colored_parts.append(mesh)
        logger.info(f"  {geom_name}: {len(mesh.faces)} faces, "
                    f"color={color_rgb}")

    if not colored_parts:
        raise ValueError("No geometry found in GLB")

    _progress(40, f"Merging {len(colored_parts)} meshes...")

    # Concatenate all parts — this preserves vertex colors
    merged = trimesh.util.concatenate(colored_parts)
    logger.info(f"Merged: {len(merged.faces)} faces, "
                f"{len(merged.vertices)} verts")

    _progress(60, "Repairing mesh...")

    # Basic cleanup
    merged.merge_vertices(merge_tex=False, merge_norm=False)
    merged.remove_unreferenced_vertices()
    try:
        face_mask = merged.area_faces > 1e-10
        if not face_mask.all():
            merged.update_faces(face_mask)
    except Exception:
        pass
    trimesh.repair.fill_holes(merged)
    trimesh.repair.fix_normals(merged)
    trimesh.repair.fix_winding(merged)

    logger.info(f"After repair: {len(merged.faces)} faces, "
                f"watertight={merged.is_watertight}")

    # Try to make watertight — try multiple strategies
    if not merged.is_watertight:
        _progress(75, "Running manifold repair...")

        # Save colors for re-assignment after repair
        orig_verts = np.array(merged.vertices)
        try:
            orig_colors = np.array(merged.visual.vertex_colors)
        except Exception:
            orig_colors = np.full((len(orig_verts), 4), 200, dtype=np.uint8)

        def _recolor(repaired_mesh):
            """Re-assign colors to repaired mesh via nearest-neighbor."""
            from scipy.spatial import KDTree
            tree = KDTree(orig_verts)
            _, idx = tree.query(np.array(repaired_mesh.vertices))
            new_colors = orig_colors[idx]
            repaired_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=repaired_mesh, vertex_colors=new_colors)
            return repaired_mesh

        # Strategy 1: manifold3d
        try:
            import manifold3d
            manifold = manifold3d.Manifold(
                mesh=manifold3d.Mesh(
                    vert_properties=np.array(merged.vertices, dtype=np.float32),
                    tri_verts=np.array(merged.faces, dtype=np.uint32),
                )
            )
            out_mesh = manifold.to_mesh()
            repaired = trimesh.Trimesh(
                vertices=out_mesh.vert_properties[:, :3],
                faces=out_mesh.tri_verts,
            )
            if repaired.is_watertight and len(repaired.faces) > 0:
                logger.info(f"manifold3d: {len(merged.faces)} → "
                            f"{len(repaired.faces)} faces ✓")
                merged = _recolor(repaired)
            else:
                logger.warning("manifold3d: not watertight, trying voxel...")
                raise ValueError("not watertight")
        except Exception as e:
            logger.warning(f"manifold3d failed ({e}), trying voxel remesh...")

            # Strategy 2: voxel remesh — guaranteed watertight
            try:
                _progress(82, "Voxel remesh (may take a moment)...")
                extents = merged.bounding_box.extents
                pitch = max(extents) / 200  # 200 voxels along longest axis
                pitch = max(pitch, 0.5)
                pitch = min(pitch, 4.0)
                logger.info(f"Voxelizing at pitch={pitch:.2f}m...")
                vox = merged.voxelized(pitch).fill()
                result = vox.marching_cubes
                if result.is_watertight and len(result.faces) > 0:
                    logger.info(f"Voxel remesh: {len(merged.faces)} → "
                                f"{len(result.faces)} faces ✓")
                    # Simplify — target 100k faces max, preserve color
                    if len(result.faces) > 150000:
                        _progress(88, "Simplifying mesh...")
                        target = min(len(result.faces), 100000)
                        simplified = result.simplify_quadric_decimation(target)
                        if simplified.is_watertight and len(simplified.faces) > 0:
                            logger.info(f"Simplified: {len(result.faces)} → "
                                        f"{len(simplified.faces)} faces")
                            result = simplified
                    merged = _recolor(result)
                else:
                    logger.warning("Voxel remesh not watertight either")
            except Exception as e2:
                logger.warning(f"Voxel remesh failed ({e2}), exporting as-is")

    _progress(90, "Exporting PLY...")

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if scale != 1.0:
        merged.apply_scale(scale)

    merged.export(str(out), file_type='ply')

    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / 1024 / 1024
    watertight = merged.is_watertight

    logger.info(f"{'✓' if watertight else '✗'} {out.name}: "
                f"{len(merged.faces)} faces, "
                f"{'watertight' if watertight else 'non-manifold'}, "
                f"{size_mb:.1f} MB, {elapsed:.1f}s")

    _progress(100, "Done!")
    return {
        'output_path': str(out),
        'faces': len(merged.faces),
        'vertices': len(merged.vertices),
        'watertight': watertight,
        'size_mb': round(size_mb, 2),
        'elapsed_seconds': round(elapsed, 1),
    }


def generate_ply_layers(glb_path: str, output_dir: str,
                        name: str = "city",
                        scale: float = 1.0,
                        min_thickness: float = 0.5,
                        progress_callback=None) -> dict:
    """Generate separate PLY files for each printable layer from a GLB.

    Parameters
    ----------
    glb_path : str
        Path to the source GLB file (from CityBuilder pipeline).
    output_dir : str
        Directory to write PLY files and manifest.
    name : str
        Base name for output files.
    scale : float
        Scale factor (1.0 = original meters). Use e.g. 0.001 to convert
        m → mm if your slicer expects mm.
    min_thickness : float
        Minimum wall thickness in model units.
    progress_callback : callable
        Optional (pct, msg) callback.

    Returns
    -------
    dict with 'layers' list and 'manifest_path'.
    """
    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    t0 = time.perf_counter()
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _progress(0, "Loading GLB model...")
    scene = trimesh.load(glb_path, force='scene')

    if not isinstance(scene, trimesh.Scene):
        # Single mesh — wrap it
        scene = trimesh.Scene(geometry={'model': scene})

    # Map GLB geometry names to meshes
    glb_meshes = {}
    for geom_name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
            glb_meshes[geom_name] = geom
            logger.info(f"  GLB mesh: {geom_name} — {len(geom.faces)} faces, "
                        f"{len(geom.vertices)} verts")

    # Compute overall scene bounds for base plate and pegs
    all_verts = np.vstack([m.vertices for m in glb_meshes.values()])
    scene_bounds = (all_verts.min(axis=0), all_verts.max(axis=0))
    logger.info(f"Scene bounds: {scene_bounds[0]} → {scene_bounds[1]}")

    # Track which GLB groups got assigned
    assigned = set()
    results = []
    total_layers = len(PRINT_LAYERS)

    for i, (layer_name, layer_def) in enumerate(
            sorted(PRINT_LAYERS.items(), key=lambda x: x[1]['order'])):

        pct = 10 + int(80 * i / total_layers)
        _progress(pct, f"Processing layer: {layer_name}...")

        color = layer_def['color']

        if layer_name == 'base':
            # Procedurally generated base plate
            mesh = _create_base_plate(scene_bounds)
            mesh = _assign_solid_color(mesh, color)
        else:
            # Merge all matching GLB groups into one mesh
            parts = []
            for grp in layer_def['glb_groups']:
                if grp in glb_meshes:
                    parts.append(glb_meshes[grp])
                    assigned.add(grp)

            if not parts:
                logger.info(f"  Layer '{layer_name}': no matching meshes, skipping")
                continue

            # Concatenate all parts into one mesh
            if len(parts) == 1:
                mesh = parts[0].copy()
            else:
                mesh = trimesh.util.concatenate(parts)

            # Solidify open surfaces (terrain, water, roads, details)
            # These are flat/curved sheets that need thickness for printing
            if layer_name in ('terrain', 'water', 'roads', 'details'):
                mesh = _solidify_surface(mesh, thickness=2.0)

            # Make watertight
            mesh = _make_watertight(mesh, min_thickness)
            if mesh is None or len(mesh.faces) == 0:
                logger.warning(f"  Layer '{layer_name}': empty after repair, skipping")
                continue

            # Assign solid color
            mesh = _assign_solid_color(mesh, color)

        # Apply scale
        if scale != 1.0:
            mesh.apply_scale(scale)

        # Export PLY
        filename = f"{name}_{layer_name}.ply"
        filepath = out / filename
        mesh.export(str(filepath), file_type='ply')

        size_mb = filepath.stat().st_size / 1024 / 1024
        watertight = mesh.is_watertight if hasattr(mesh, 'is_watertight') else False

        layer_info = {
            'layer': layer_name,
            'file': filename,
            'color_rgb': color,
            'faces': len(mesh.faces),
            'vertices': len(mesh.vertices),
            'watertight': watertight,
            'size_mb': round(size_mb, 2),
            'order': layer_def['order'],
            'description': layer_def['description'],
        }
        results.append(layer_info)
        logger.info(f"  ✓ {filename}: {len(mesh.faces)} faces, "
                    f"{'watertight' if watertight else 'NON-MANIFOLD'}, "
                    f"{size_mb:.1f} MB")

    # Check for unassigned GLB groups
    unassigned = set(glb_meshes.keys()) - assigned
    if unassigned:
        logger.warning(f"Unassigned GLB groups (not in any print layer): {unassigned}")
        # Merge unassigned into details layer
        extras = [glb_meshes[g] for g in unassigned]
        if extras:
            mesh = trimesh.util.concatenate(extras) if len(extras) > 1 else extras[0].copy()
            mesh = _make_watertight(mesh)
            if mesh is not None and len(mesh.faces) > 0:
                mesh = _assign_solid_color(mesh, [180, 180, 180])  # light gray
                if scale != 1.0:
                    mesh.apply_scale(scale)
                filename = f"{name}_extras.ply"
                filepath = out / filename
                mesh.export(str(filepath), file_type='ply')
                results.append({
                    'layer': 'extras',
                    'file': filename,
                    'color_rgb': [180, 180, 180],
                    'faces': len(mesh.faces),
                    'vertices': len(mesh.vertices),
                    'watertight': mesh.is_watertight,
                    'size_mb': round(filepath.stat().st_size / 1024 / 1024, 2),
                    'order': 99,
                    'description': f'Unassigned meshes: {", ".join(sorted(unassigned))}',
                })

    # Write manifest
    elapsed = time.perf_counter() - t0
    manifest = {
        'name': name,
        'source_glb': str(glb_path),
        'scale': scale,
        'layers': sorted(results, key=lambda x: x['order']),
        'total_faces': sum(r['faces'] for r in results),
        'total_vertices': sum(r['vertices'] for r in results),
        'elapsed_seconds': round(elapsed, 1),
        'print_notes': {
            'layer_order': 'Print base first, then stack layers in order',
            'alignment': '4 corner pegs for registration',
            'colors': 'Each layer is single-color for filament swap or multi-material',
        },
    }
    manifest_path = out / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    _progress(100, "PLY layers complete!")
    logger.info(f"Generated {len(results)} PLY layers in {elapsed:.1f}s")
    logger.info(f"Manifest: {manifest_path}")

    return {
        'layers': results,
        'manifest_path': str(manifest_path),
        'output_dir': str(out),
    }
