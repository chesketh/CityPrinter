"""Build and evaluate diverse downtown city renders with GPT-5.2 Vision."""
import asyncio, sys, os, base64, math, io, json, time
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from citybuilder import CityBuilder, BoundingBox

API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith('OPENAI_API_KEY='):
                API_KEY = line.split('=', 1)[1].strip()

RENDER_DIR = 'output/downtown-renders'
RESULTS_PATH = 'output/downtown_evaluation.json'

# Notable downtowns NOT in the famous buildings list
OFFSET = 0.003  # ~330m — decent downtown block
DOWNTOWNS = [
    {"name": "Downtown Miami (Brickell)", "lat": 25.7617, "lon": -80.1918},
    {"name": "Downtown Singapore (Marina Bay)", "lat": 1.2838, "lon": 103.8591},
    {"name": "Downtown Hong Kong (Central)", "lat": 22.2830, "lon": 114.1580},
    {"name": "Downtown Nashville", "lat": 36.1627, "lon": -86.7816},
    {"name": "Downtown Denver", "lat": 39.7439, "lon": -104.9880},
    {"name": "Downtown Boston (Financial)", "lat": 42.3554, "lon": -71.0564},
]


def build_downtown(entry):
    """Build a downtown area, return GLB path or None."""
    lat, lon = entry["lat"], entry["lon"]
    slug = entry["name"].lower().replace(" ", "-").replace("(", "").replace(")", "")
    filename = f"downtown-{slug}.glb"
    glb_path = os.path.join("output", filename)

    bbox = BoundingBox(
        north=lat + OFFSET, south=lat - OFFSET,
        east=lon + OFFSET, west=lon - OFFSET,
    )
    try:
        builder = CityBuilder()
        city_id = asyncio.run(builder.process_city(bbox))
        result_path = builder.generate_glb(city_id, filename)
        return str(result_path)
    except Exception as e:
        print(f"  BUILD FAILED: {e}")
        return None


def render_glb(glb_path, name_slug):
    """Render GLB from 2 angles, return paths to PNGs."""
    import pyrender

    tm_scene = trimesh.load(glb_path)
    if not tm_scene.geometry:
        return []

    center = tm_scene.centroid
    ext = max(tm_scene.extents)

    pr_scene = pyrender.Scene(
        bg_color=[0.6, 0.75, 0.9, 1.0],
        ambient_light=[0.3, 0.3, 0.3],
    )

    for geom_name, geom in tm_scene.geometry.items():
        try:
            xform = tm_scene.graph.get(geom_name)[0] if geom_name in tm_scene.graph.nodes else np.eye(4)
        except Exception:
            xform = np.eye(4)
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

    dist = ext * 1.4
    paths = []

    # View 1: elevated 3/4 view
    elev1 = math.radians(30)
    azim1 = math.radians(-35)
    eye1 = center + np.array([
        dist * math.cos(elev1) * math.sin(azim1),
        dist * math.sin(elev1),
        dist * math.cos(elev1) * math.cos(azim1),
    ])
    pr_scene.set_pose(cam_node, _look_at(eye1, center))
    color1, _ = renderer.render(pr_scene)
    p1 = os.path.join(RENDER_DIR, f'{name_slug}-1.png')
    Image.fromarray(color1).save(p1); paths.append(p1)

    # View 2: lower angle, opposite side
    elev2 = math.radians(18)
    azim2 = math.radians(120)
    eye2 = center + np.array([
        dist * math.cos(elev2) * math.sin(azim2),
        dist * math.sin(elev2),
        dist * math.cos(elev2) * math.cos(azim2),
    ])
    pr_scene.set_pose(cam_node, _look_at(eye2, center))
    color2, _ = renderer.render(pr_scene)
    p2 = os.path.join(RENDER_DIR, f'{name_slug}-2.png')
    Image.fromarray(color2).save(p2); paths.append(p2)

    renderer.delete()
    return paths


def evaluate_downtown(name, render_paths):
    """Send renders to GPT-5.2 for overall city scene quality assessment."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    content = [{"type": "text", "text": f"""You are evaluating a 3D city model of "{name}".

Image 1: Elevated 3/4 view of the downtown area
Image 2: Lower angle view from opposite side

This model was auto-generated from OpenStreetMap data + Overture building heights + satellite imagery.
Score the OVERALL quality of this downtown scene 0-100%.

Evaluate:
1. Building heights — do buildings have varied, realistic heights or all flat?
2. Building density — does it look like a real downtown?
3. Terrain/ground — is there a ground plane, roads, green areas?
4. Vegetation — are there trees/bushes visible?
5. Overall impression — would someone recognize this as a real city skyline?
6. Any visual glitches — floating geometry, Z-fighting, missing ground, etc?

Respond in EXACTLY this format:
SCORE: XX%
STRENGTHS: <brief list of what looks good>
WEAKNESSES: <brief list of what needs improvement>
GLITCHES: <any visual bugs or none>
OVERALL: <one sentence summary>"""}]

    for p in render_paths:
        img = Image.open(p)
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{b64}", "detail": "high"
        }})

    response = client.chat.completions.create(
        model="gpt-5.2-chat-latest",
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=500,
    )
    return response.choices[0].message.content


def parse_response(text):
    result = {'raw': text, 'score': None, 'strengths': '', 'weaknesses': '', 'glitches': '', 'overall': ''}
    for line in text.split('\n'):
        line = line.strip()
        if line.upper().startswith('SCORE:'):
            try:
                result['score'] = int(line.split(':')[1].strip().replace('%', ''))
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith('STRENGTHS:'):
            result['strengths'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('WEAKNESSES:'):
            result['weaknesses'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('GLITCHES:'):
            result['glitches'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('OVERALL:'):
            result['overall'] = line.split(':', 1)[1].strip()
    return result


if __name__ == '__main__':
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable or create .env file")
        sys.exit(1)

    os.makedirs(RENDER_DIR, exist_ok=True)
    results = []

    for i, entry in enumerate(DOWNTOWNS):
        name = entry["name"]
        slug = name.lower().replace(" ", "-").replace("(", "").replace(")", "")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(DOWNTOWNS)}] {name}")
        print(f"{'='*60}")

        # Build
        print("  Building...")
        t0 = time.time()
        glb_path = build_downtown(entry)
        build_time = time.time() - t0
        if not glb_path:
            continue
        size_kb = os.path.getsize(glb_path) / 1024
        print(f"  Built in {build_time:.1f}s ({size_kb:.0f} KB)")

        # Render
        print("  Rendering...")
        try:
            render_paths = render_glb(glb_path, slug)
            if not render_paths:
                print("  SKIP - no geometry")
                continue
            print(f"  Rendered {len(render_paths)} views")
        except Exception as e:
            print(f"  RENDER FAILED: {e}")
            continue

        # Evaluate
        print("  Evaluating with GPT-5.2...")
        try:
            gpt_text = evaluate_downtown(name, render_paths)
            parsed = parse_response(gpt_text)
            parsed['name'] = name
            parsed['lat'] = entry['lat']
            parsed['lon'] = entry['lon']
            parsed['glb_size_kb'] = round(size_kb, 1)
            parsed['build_time_s'] = round(build_time, 1)
            results.append(parsed)
            print(f"  Score: {parsed['score']}%")
            print(f"  Strengths: {parsed['strengths'][:80]}")
            print(f"  Weaknesses: {parsed['weaknesses'][:80]}")
            print(f"  Glitches: {parsed['glitches'][:80]}")
        except Exception as e:
            print(f"  EVAL FAILED: {e}")
            continue

        # Save intermediate
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

        if i < len(DOWNTOWNS) - 1:
            time.sleep(2)

    # Summary
    print(f"\n{'='*70}")
    print("DOWNTOWN EVALUATION REPORT")
    print(f"{'='*70}")
    results.sort(key=lambda r: r.get('score') or 0, reverse=True)
    for r in results:
        score = f"{r['score']:3d}%" if r['score'] is not None else " N/A"
        print(f"  {score}  {r['name']:40s}  {r.get('overall', '')[:60]}")

    print(f"\nFull results saved to: {RESULTS_PATH}")
