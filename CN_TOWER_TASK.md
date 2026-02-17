# CN Tower Landmark — Iterative GLB Builder Task

## Your Mission

Create a high-quality 3D GLB model of landmarks and buildings (CN tower used an example) using an iterative build → render → evaluate loop. Target **80%+ geometric realism** as scored by GPT-5.2 Vision.

## Project Location

```
c:\Users\chesk\.cursor-tutor\CityBuilder
```

## How the Iteration Loop Works

1. **Edit `build_cn_tower.py`** — modify geometry parameters
2. **Run `python build_cn_tower.py`** — generates GLB in ~1-2 seconds
3. **Run `python eval_cn_tower.py`** — renders to PNG, sends to GPT-5.2 Vision for scoring
4. **Read the GPT feedback**, make changes, repeat

**Iterate quickly.** Each build takes <2 seconds. Don't over-analyze — build, evaluate, adjust, repeat. Make **big** changes between iterations, not tiny tweaks. You have a fast engine — use it.

## Key Architecture & Available Primitives

### Coordinate System
- **Y-up** (Three.js/glTF convention): vertices are `[easting, height, northing]`
- trimesh passes vertices through as-is (no auto conversion)

### Core Functions (import from `citybuilder/glb.py`)

```python
from citybuilder.glb import (
    _make_tapered_prism,    # frustum: cx, cz, y_bot, y_top, r_bot, r_top, nsides, rotation
    _add_prism_to_group,    # append verts/faces to a mesh group dict
    _build_profile_slices,  # stack of tapered prisms from (height, radius) profile
)
```

**`_make_tapered_prism(cx, cz, y_bot, y_top, r_bot, r_top, nsides=8, rotation=0.0)`**
- Creates a frustum (cone section) with nsides polygonal cross-section
- Returns (verts, faces) lists
- Use nsides=24+ for smooth cylinders, nsides=3 for triangular, nsides=6 for hex

**`_build_profile_slices(cx, cz, base_y, profile, group, nsides=12)`**
- Takes a list of `(height_above_base, radius)` tuples
- Stacks tapered prisms to build a profile
- Perfect for towers with varying cross-section

**`_add_prism_to_group(group, verts, faces)`**
- Appends geometry to a group dict `{'verts': [], 'faces': [], 'offset': 0}`

### Materials
```python
import trimesh
mat = trimesh.visual.material.PBRMaterial(
    baseColorFactor=[r, g, b, a],  # 0-1 range
    roughnessFactor=0.4,
    metallicFactor=0.5,
    doubleSided=True
)
mesh.visual = trimesh.visual.TextureVisuals(material=mat)
```

### Scene Export
```python
scene = trimesh.Scene()
scene.add_geometry(mesh, geom_name='tower')
scene.export('output/cn-tower.glb')
```

## CN Tower Real Dimensions

- **Total height**: 553.3 m (with antenna)
- **Main deck (SkyPod)**: at ~346 m — hexagonal shape, ~36 m diameter
- **Restaurant level**: at ~351 m — rotating restaurant, cylindrical
- **Observation deck (LookOut)**: at ~342 m
- **SkyPod (upper observation)**: at ~447 m — smaller cylinder, ~12 m diameter
- **Antenna/mast**: from ~447 m to 553.3 m, tapers from ~5 m to ~1 m
- **Main shaft**: Y-shaped (3 hollow tubes) tapering from ~28 m at base to ~7 m at main deck
  - The Y-shape is distinctive: 3 pillars that merge into one round column around 335 m
- **Base**: 3 curved buttresses meeting the ground

### Simplified Geometry Approach (recommended starting point)
For a low-poly model, represent the CN Tower as:
1. **Base section (0-335m)**: 3 tapered pillars (or a single tapered hexagonal prism)
2. **Main pod (335-360m)**: Wide disc/donut shape
3. **Upper shaft (360-447m)**: Thin cylinder
4. **Sky Pod (447-460m)**: Small disc
5. **Antenna (460-553m)**: Very thin cone

## Files to Create

### `build_cn_tower.py`
```python
"""Fast standalone CN Tower GLB builder for rapid iteration."""
import math, sys, time
import numpy as np
import trimesh

sys.path.insert(0, '.')
from citybuilder.glb import (
    _make_tapered_prism, _add_prism_to_group, _build_profile_slices,
)

groups = {}
groups['tower'] = {'verts': [], 'faces': [], 'offset': 0}
groups['pod']   = {'verts': [], 'faces': [], 'offset': 0}

# ── TOWER SHAFT ──
# Use _build_profile_slices or stacked _make_tapered_prism calls
# Profile: (height, radius) — shaft tapers from base to main deck
shaft_profile = [
    (  0.0, 14.0),   # base
    ( 50.0, 13.0),
    (100.0, 11.5),
    (150.0, 10.0),
    (200.0,  8.5),
    (250.0,  7.0),
    (300.0,  5.5),
    (335.0,  4.5),   # just below main pod
]
_build_profile_slices(0, 0, 0, shaft_profile, groups['tower'], nsides=24)

# ── MAIN POD (SkyPod / observation deck) ──
pod_profile = [
    (335.0,  4.5),
    (338.0, 14.0),   # pod flares out
    (342.0, 18.0),   # widest — observation deck
    (348.0, 17.0),   # restaurant level
    (353.0, 14.0),   # narrows above restaurant
    (358.0,  5.0),   # back to shaft width
]
_build_profile_slices(0, 0, 0, pod_profile, groups['pod'], nsides=24)

# ── UPPER SHAFT ──
upper_profile = [
    (358.0, 4.0),
    (440.0, 3.5),
    (447.0, 3.0),    # below sky pod
]
_build_profile_slices(0, 0, 0, upper_profile, groups['tower'], nsides=24)

# ── SKY POD ──
skypod_profile = [
    (447.0, 3.0),
    (449.0, 6.0),    # flare out
    (453.0, 5.5),    # sky pod
    (456.0, 3.0),    # narrows
]
_build_profile_slices(0, 0, 0, skypod_profile, groups['pod'], nsides=24)

# ── ANTENNA ──
antenna_profile = [
    (456.0, 2.5),
    (480.0, 2.0),
    (510.0, 1.5),
    (540.0, 0.8),
    (553.0, 0.3),
]
_build_profile_slices(0, 0, 0, antenna_profile, groups['tower'], nsides=12)

# ── EXPORT ──
t0 = time.perf_counter()
scene = trimesh.Scene()
for name, g in groups.items():
    if not g['verts']:
        continue
    mesh = trimesh.Trimesh(
        vertices=np.array(g['verts'], dtype=np.float64),
        faces=np.array(g['faces'], dtype=np.int32),
        process=False)
    mesh.fix_normals()
    if name == 'tower':
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.78, 0.78, 0.78, 1.0],  # concrete grey
            roughnessFactor=0.6, metallicFactor=0.1, doubleSided=True)
    else:
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.85, 0.85, 0.85, 1.0],  # lighter grey for pods
            roughnessFactor=0.3, metallicFactor=0.3, doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(mesh, geom_name=name)

import glob as _glob
existing = _glob.glob('output/cn-tower-v*.glb')
nums = [int(f.split('-v')[-1].split('.')[0]) for f in existing if f.split('-v')[-1].split('.')[0].isdigit()]
ver = max(nums, default=0) + 1
out = 'output/cn-tower.glb'
vout = f'output/cn-tower-v{ver}.glb'
scene.export(out)
import shutil; shutil.copy2(out, vout)
dt = time.perf_counter() - t0
print(f"v{ver} exported in {dt:.2f}s -> {vout}")
```

### `eval_cn_tower.py`
```python
"""Evaluate CN Tower model using GPT-5.2 Vision."""
import sys, os, base64, math, io
import numpy as np
import trimesh
from PIL import Image

API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith('OPENAI_API_KEY='):
                API_KEY = line.split('=', 1)[1].strip()

GLB_PATH = 'output/cn-tower.glb'
RENDER_DIR = 'output'

# Reference: public domain image of CN Tower
REFERENCE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Toronto_-_ON_-_Toronto_Skyline2.jpg/1280px-Toronto_-_ON_-_Toronto_Skyline2.jpg"


def render_with_pyrender(glb_path):
    """Render GLB with pyrender for proper PBR lighting."""
    import pyrender

    tm_scene = trimesh.load(glb_path)
    center = tm_scene.centroid
    ext = max(tm_scene.extents)

    pr_scene = pyrender.Scene(
        bg_color=[0.6, 0.75, 0.9, 1.0],
        ambient_light=[0.3, 0.3, 0.3],
    )

    for name, geom in tm_scene.geometry.items():
        xform = tm_scene.graph.get(name)[0] if name in tm_scene.graph.nodes else np.eye(4)
        geom.fix_normals()
        if not hasattr(geom.visual, 'material') or geom.visual.material is None:
            color = [0.8, 0.8, 0.8, 1.0]
        else:
            mat = geom.visual.material
            color = list(mat.baseColorFactor) if hasattr(mat, 'baseColorFactor') and mat.baseColorFactor is not None else [0.8, 0.8, 0.8, 1.0]

        pr_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color, metallicFactor=0.1, roughnessFactor=0.5, doubleSided=True)
        pr_mesh = pyrender.Mesh.from_trimesh(geom, material=pr_mat, smooth=True)
        pr_scene.add(pr_mesh, pose=xform)

    # Lighting
    up = np.array([0, 1, 0])
    sun = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=4.0)
    sun_pose = np.eye(4)
    sun_dir = np.array([0.5, -0.8, -0.3])
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    right = np.cross(-sun_dir, up); right = right / np.linalg.norm(right)
    true_up = np.cross(right, -sun_dir)
    sun_pose[:3, 0] = right; sun_pose[:3, 1] = true_up; sun_pose[:3, 2] = -sun_dir
    sun_pose[:3, 3] = center + np.array([0, ext, 0])
    pr_scene.add(sun, pose=sun_pose)

    fill = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.5)
    fill_pose = np.eye(4)
    fill_dir = np.array([-0.3, -0.5, 0.5]); fill_dir = fill_dir / np.linalg.norm(fill_dir)
    fill_right = np.cross(-fill_dir, up); fill_right = fill_right / np.linalg.norm(fill_right)
    fill_up = np.cross(fill_right, -fill_dir)
    fill_pose[:3, 0] = fill_right; fill_pose[:3, 1] = fill_up; fill_pose[:3, 2] = -fill_dir
    pr_scene.add(fill, pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(1280, 960)

    def _look_at(eye, target):
        fwd = np.array(target) - np.array(eye)
        fwd = fwd / np.linalg.norm(fwd)
        r = np.cross(fwd, np.array([0, 1, 0])); r = r / np.linalg.norm(r)
        u = np.cross(r, fwd)
        pose = np.eye(4)
        pose[:3, 0] = r; pose[:3, 1] = u; pose[:3, 2] = -fwd; pose[:3, 3] = eye
        return pose

    cam = pyrender.PerspectiveCamera(yfov=math.radians(45))
    cam_node = pr_scene.add(cam)

    dist = ext * 1.8
    paths = []

    # View 1: Ground-level looking up (classic tourist angle)
    elev1 = math.radians(25)
    azim1 = math.radians(-30)
    eye1 = center + np.array([
        dist * math.cos(elev1) * math.sin(azim1),
        dist * math.sin(elev1),
        dist * math.cos(elev1) * math.cos(azim1),
    ])
    pr_scene.set_pose(cam_node, _look_at(eye1, center))
    color1, _ = renderer.render(pr_scene)
    p1 = os.path.join(RENDER_DIR, 'cn-tower-render-1.png')
    Image.fromarray(color1).save(p1); paths.append(p1)
    print(f"  Rendered {p1}")

    # View 2: Elevated side view
    elev2 = math.radians(15)
    azim2 = math.radians(90)
    eye2 = center + np.array([
        dist * math.cos(elev2) * math.sin(azim2),
        dist * math.sin(elev2),
        dist * math.cos(elev2) * math.cos(azim2),
    ])
    pr_scene.set_pose(cam_node, _look_at(eye2, center))
    color2, _ = renderer.render(pr_scene)
    p2 = os.path.join(RENDER_DIR, 'cn-tower-render-2.png')
    Image.fromarray(color2).save(p2); paths.append(p2)
    print(f"  Rendered {p2}")

    # View 3: From above
    eye3 = center + np.array([dist * 0.3, dist * 0.9, dist * 0.3])
    pr_scene.set_pose(cam_node, _look_at(eye3, center))
    color3, _ = renderer.render(pr_scene)
    p3 = os.path.join(RENDER_DIR, 'cn-tower-render-3.png')
    Image.fromarray(color3).save(p3); paths.append(p3)
    print(f"  Rendered {p3}")

    renderer.delete()
    return paths


def evaluate_with_gpt(render_paths, reference_url):
    """Send rendered images + reference to GPT-5.2 for evaluation."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    content = [{"type": "text", "text": """You are evaluating a 3D model of the CN Tower in Toronto.

Image 1: 3D model rendered from ground-level angle
Image 2: 3D model rendered from elevated side view
Image 3: 3D model rendered from above
Image 4: Reference photo of the real CN Tower

This is a LOW-POLY architectural model. Score it for SHAPE, PROPORTIONS, and SILHOUETTE.

Score the model 0-100% on geometric realism. Focus on:
1. Overall proportions — is the height-to-width ratio correct (very tall and slender)?
2. Main observation pod shape — is it the correct disc/donut shape at ~63% height?
3. Shaft taper — does it taper smoothly from wide base to narrow top?
4. SkyPod — is there a smaller bulge above the main pod?
5. Antenna — tall thin spire from SkyPod to tip?
6. Base — does it show the characteristic Y-shape or buttress structure?

SCORE: XX%
Then explain what's right, what's wrong, and the single most impactful geometry change."""}]

    for p in render_paths:
        img = Image.open(p)
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{b64}", "detail": "high"
        }})

    content.append({"type": "image_url", "image_url": {
        "url": reference_url, "detail": "high"
    }})

    response = client.chat.completions.create(
        model="gpt-5.2-chat-latest",
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=1000,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable or create .env file")
        sys.exit(1)

    print("Rendering model...")
    try:
        paths = render_with_pyrender(GLB_PATH)
    except Exception as e:
        print(f"  pyrender failed ({e}), using fallback...")
        # Basic trimesh fallback could go here
        raise

    print("Sending to GPT-5.2 for evaluation...")
    result = evaluate_with_gpt(paths, REFERENCE_URL)
    print("\n" + "=" * 60)
    print("GPT-5.2 EVALUATION")
    print("=" * 60)
    print(result.encode('ascii', 'replace').decode('ascii'))
```

## Lessons Learned from Opera House Iteration (apply these!)

### Speed & Strategy
- **Iterate fast**: build+eval takes ~15 seconds total. Don't spend 5 minutes analyzing before each change.
- **Make BIG changes** between iterations. Don't tweak a value by 2%. Change the whole approach.
- **Read the GPT feedback carefully** — it tells you exactly what to fix.
- **Look at the renders yourself** (read the PNG files) to see what GPT sees.

### Technical Tips
- **PBR materials** via `trimesh.visual.material.PBRMaterial` for reliable GLB color
- **`trimesh.Scene`** with separate meshes per part (tower, pod, antenna) for distinct colors
- **Image resize** before sending to GPT: `img.thumbnail((1024, 1024))` then JPEG quality 85
- **Version numbering**: auto-increment from existing files in output/
- **Always use `process=False`** in `trimesh.Trimesh()` to prevent trimesh from mangling geometry
- **`mesh.fix_normals()`** to ensure correct shading

### Common Pitfalls
- Don't add too many small details — GPT evaluates overall SHAPE not fine detail
- The profile approach (`_build_profile_slices`) is the easiest way to build tower shapes
- Camera distance = `ext * 1.8` works well for tall structures (ext = max extent of model)
- For tall structures, ground-level camera angles are most recognizable
- GPT-5.2 scores have ~±10% variance between runs. Don't panic over small drops.

### GPT Evaluation Variance
- Same geometry can score ±10-15% across runs
- If something scores 72%, it might score 68% or 78% on another run
- Track the general trend, not individual scores
- 3 consecutive 70%+ scores = solid geometry

## Environment Notes
- Working directory: `c:\Users\chesk\.cursor-tutor\CityBuilder`
- Python with trimesh, numpy, pyrender, PIL already installed
- OpenAI API key in `.env` file (OPENAI_API_KEY=...)
- Output directory: `output/` (already exists)
- Model: `gpt-5.2-chat-latest` for evaluation

## Your Workflow

### Step 0: Discover Which Landmarks Need Custom Models

Run `python eval_city_builds.py` to evaluate ALL famous building GLBs with GPT-5.2 Vision. This script:
1. Loads each GLB from `output/famous_buildings_report.json`
2. Renders 2 views per building with pyrender
3. Sends renders to GPT-5.2 asking "does this look like [landmark name]?"
4. Scores each 0-100% on recognizability
5. Outputs a ranked report (worst → best) to `output/landmark_evaluation.json`

**Rebuild any landmark scoring below 55%.** Skip landmarks that already have custom models in `_LANDMARKS` dict. Landmarks scoring 55%+ are acceptable as-is from OSM data.

### Step 1-6: Build → Evaluate → Iterate

1. Create `build_<landmark>.py` and `eval_<landmark>.py` (starter code above)
2. Run `python build_<landmark>.py` to generate v1
3. Run `python eval_<landmark>.py` to evaluate
4. Read the GPT feedback
5. Modify the geometry in `build_<landmark>.py`
6. Repeat from step 2

### Step 7: CRITICAL — Integrate into Landmark System

See integration checklist below. **Without this step, the standalone GLB is useless.**

### Step 8: Loop Back to Step 0

Move on to the next poorly-rendered landmark. Re-run `eval_city_builds.py` periodically to re-assess.

Target: **Repeatedly scoring 70%+ from GPT-5.2** is sufficient. Scores have ~±10% variance — if you consistently hit 70%+, stop iterating and move to integration. Only rebuild landmarks that score below 55% in the city eval.

## CRITICAL: Integration into Landmark System

**The build/eval loop is NOT the final step.** After achieving the target score, you MUST integrate the geometry into `citybuilder/glb.py` so it replaces the bad OSM render during city builds. Without this step, the standalone GLB is useless.

### Integration Checklist

1. **Add to `_LANDMARKS` dict** (~line 304 in `glb.py`):
   ```python
   'landmark_name': {'lat': XX.XXXX, 'lon': YY.YYYY, 'exclude_radius': NN},
   ```
   The `exclude_radius` caps nearby OSM buildings to 8m so they don't clash.

2. **Create `_generate_<landmark>()` function** in `glb.py`:
   - Signature: `def _generate_<name>(cx_utm, cz_utm, terrain_elev, groups, type_colors)`
   - Use `cx_utm, cz_utm` as the center position (UTM coordinates)
   - Use `terrain_elev` as the ground height (Y offset)
   - Create new mesh groups: `groups['<name>_part'] = {'verts': [], 'faces': [], 'offset': 0}`
   - Set colors: `type_colors['<name>_part'] = [r, g, b, a]`
   - Call `_build_profile_slices(cx_utm, cz_utm, terrain_elev, ...)` with UTM center

3. **Add handler in `_inject_landmarks()`** (~line 746):
   ```python
   elif name == 'landmark_name':
       _generate_landmark_name(x_utm, y_utm, terrain_elev, groups, type_colors)
   ```

4. **Clear `__pycache__`** after modifying glb.py

5. **Test** by building the area around the landmark and verifying it renders

### What Makes a Good Landmark Candidate

Look for landmarks that OSM represents poorly — typically:
- Towers/spires (rendered as flat buildings or thin rectangles)
- Domes and curved structures (rendered as blocky polygons)
- Monuments and sculptures (missing entirely or just a footprint)
- Bridges and arches (rendered as flat surfaces)

Check the `generate_famous.py` list for known landmarks, or look at existing city GLBs.
