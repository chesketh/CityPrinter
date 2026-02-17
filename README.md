# CityPrinter 🏗️🖨️

Generate 3D-printable city models from OpenStreetMap data. Forked from [CityBuilder](https://github.com/chesketh/CityBuilder).

## What's Different

CityBuilder outputs beautiful GLB models for web viewing. **CityPrinter** takes those same models and splits them into **separate colored layers** designed for multi-color 3D printing.

Each layer is a single solid color, watertight PLY file. Print each layer with a different filament color, and they stack together with alignment pegs.

## Print Layers

| Order | Layer | Color | Contents |
|-------|-------|-------|----------|
| 0 | Base | Dark brown | Solid base plate with alignment pegs |
| 1 | Water | Blue | Ocean, rivers, lakes |
| 2 | Terrain | Earth green | Ground surface, rocks, glaciers |
| 3 | Roads | Dark gray | Roads, bridges, railways |
| 4 | Buildings | White | All structures |
| 5 | Vegetation | Forest green | Trees (simplified) |
| 6 | Details | Park green | Parks, piers, sports fields |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate print layers for a bounding box (Manhattan example)
python -m citybuilder print-bbox 40.758 40.748 -73.975 -73.990 \
    --name manhattan --output manhattan.glb

# Output: manhattan_print/ directory with PLY files + manifest.json
```

## API

```bash
# Start the backend
uvicorn backend.app:app --reload

# Build with PLY output
curl -X POST http://localhost:8000/api/build/bbox \
  -H "Content-Type: application/json" \
  -d '{"north": 40.758, "south": 40.748, "east": -73.975, "west": -73.990,
       "output_format": "ply", "scale": 1.0}'

# Poll status
curl http://localhost:8000/api/build/status/{job_id}
```

## Frontend

Same interface as CityBuilder with an added toggle for print mode:

```bash
cd frontend
npm install
npm run dev
```

## Print Tips

- **Layer order**: Print base first, stack layers 1→6
- **Alignment**: 4 corner pegs keep layers registered
- **Scale**: Use `--scale 0.001` if your slicer expects millimeters (model is in meters)
- **Watertight**: All meshes are repaired for manifold geometry
- **Color**: Each PLY has vertex colors embedded for slicer preview, but designed for single-filament-per-layer printing

## Requirements

- Python 3.10+
- trimesh, numpy, scipy, shapely, pyproj
- Internet connection (downloads OSM + elevation data)

## License

Same as CityBuilder.
