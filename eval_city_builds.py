"""Evaluate famous building GLBs with GPT-5.2 Vision to find poorly-rendered landmarks."""
import sys, os, base64, math, io, json, time
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

REPORT_PATH = 'output/famous_buildings_report.json'
RENDER_DIR = 'output/eval-renders'
RESULTS_PATH = 'output/landmark_evaluation.json'


def render_glb(glb_path, name_slug, center_on_tallest=True):
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

    renderer = pyrender.OffscreenRenderer(1024, 768)

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

    dist = ext * 1.6
    paths = []

    # View 1: ~30 degree elevation, angled
    elev1 = math.radians(25)
    azim1 = math.radians(-30)
    eye1 = center + np.array([
        dist * math.cos(elev1) * math.sin(azim1),
        dist * math.sin(elev1),
        dist * math.cos(elev1) * math.cos(azim1),
    ])
    pr_scene.set_pose(cam_node, _look_at(eye1, center))
    color1, _ = renderer.render(pr_scene)
    p1 = os.path.join(RENDER_DIR, f'{name_slug}-1.png')
    Image.fromarray(color1).save(p1); paths.append(p1)

    # View 2: side view
    elev2 = math.radians(15)
    azim2 = math.radians(90)
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


def evaluate_one(name, city, render_paths):
    """Send renders to GPT-5.2 for evaluation of a single landmark."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    content = [{"type": "text", "text": f"""You are evaluating an OSM-based 3D city model that should contain "{name}" in {city}.

Image 1: 3D model from angled elevated view
Image 2: 3D model from side view

This model was generated from OpenStreetMap data. OSM only has building footprints and heights, so complex shapes get simplified to extruded polygons.

Score the landmark's representation 0-100%:
- Does the landmark look recognizable as {name}?
- Are the key architectural features visible (shape, proportions, distinctive elements)?
- Or does it look like a generic box/polygon that could be any building?

Respond in EXACTLY this format:
SCORE: XX%
RECOGNIZABLE: yes/no
ISSUES: <one line describing the main visual problem>
DIFFICULTY: easy/medium/hard (how hard would it be to create a custom 3D model replacement)
IMPACT: low/medium/high (how much would a custom model improve the city view)"""}]

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
        max_completion_tokens=300,
    )
    return response.choices[0].message.content


def parse_gpt_response(text):
    """Extract structured fields from GPT response."""
    result = {'raw': text, 'score': None, 'recognizable': None, 'issues': '', 'difficulty': '', 'impact': ''}
    for line in text.split('\n'):
        line = line.strip()
        if line.upper().startswith('SCORE:'):
            try:
                result['score'] = int(line.split(':')[1].strip().replace('%', ''))
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith('RECOGNIZABLE:'):
            val = line.split(':')[1].strip().lower()
            result['recognizable'] = val.startswith('yes')
        elif line.upper().startswith('ISSUES:'):
            result['issues'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('DIFFICULTY:'):
            result['difficulty'] = line.split(':', 1)[1].strip().lower()
        elif line.upper().startswith('IMPACT:'):
            result['impact'] = line.split(':', 1)[1].strip().lower()
    return result


if __name__ == '__main__':
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable or create .env file")
        sys.exit(1)

    os.makedirs(RENDER_DIR, exist_ok=True)

    # Load the build report
    with open(REPORT_PATH) as f:
        builds = json.load(f)

    successful = [b for b in builds if b['status'] == 'success']
    # Skip CN Tower (already has custom landmark)
    skip = {'cn-tower.glb'}
    to_eval = [b for b in successful if b['filename'] not in skip]

    print(f"Evaluating {len(to_eval)} landmarks with GPT-5.2 Vision...\n")
    results = []

    for i, b in enumerate(to_eval):
        name = b['name']
        city = b['city']
        glb_path = b['glb_path']
        slug = b['filename'].replace('.glb', '')

        print(f"[{i+1}/{len(to_eval)}] {name} ({city})...")

        if not os.path.exists(glb_path):
            print(f"  SKIP - GLB not found: {glb_path}")
            continue

        try:
            render_paths = render_glb(glb_path, slug)
            if not render_paths:
                print(f"  SKIP - no geometry")
                continue
            print(f"  Rendered {len(render_paths)} views")
        except Exception as e:
            print(f"  RENDER FAILED: {e}")
            continue

        try:
            gpt_text = evaluate_one(name, city, render_paths)
            parsed = parse_gpt_response(gpt_text)
            parsed['name'] = name
            parsed['city'] = city
            parsed['filename'] = b['filename']
            parsed['glb_size_kb'] = b.get('glb_size_kb', 0)
            results.append(parsed)
            print(f"  Score: {parsed['score']}% | Recognizable: {parsed['recognizable']} | Impact: {parsed['impact']}")
        except Exception as e:
            print(f"  EVAL FAILED: {e}")
            continue

        # Save intermediate results
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

        # Brief pause to avoid rate limiting
        if i < len(to_eval) - 1:
            time.sleep(2)

    # Sort by score ascending (worst first)
    results.sort(key=lambda r: r.get('score') or 999)

    print("\n" + "=" * 70)
    print("LANDMARK EVALUATION REPORT (sorted worst to best)")
    print("=" * 70)
    for r in results:
        score = f"{r['score']:3d}%" if r['score'] is not None else " N/A"
        recog = "Y" if r.get('recognizable') else "N"
        impact = r.get('impact', '?')[:4]
        diff = r.get('difficulty', '?')[:4]
        print(f"  {score} [{recog}] {r['name']:30s} impact={impact:4s} diff={diff:4s} | {r.get('issues', '')[:50]}")

    # Top candidates for custom landmarks
    candidates = [r for r in results if r.get('score') is not None and r['score'] < 60]
    if candidates:
        print(f"\n--- TOP CANDIDATES FOR CUSTOM LANDMARKS (score < 60%) ---")
        for r in candidates:
            print(f"  {r['score']}% {r['name']} ({r['city']}) - {r['issues']}")

    print(f"\nFull results saved to: {RESULTS_PATH}")
