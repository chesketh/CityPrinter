"""Evaluate Brandenburg Gate model using GPT-5.2 Vision."""
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

GLB_PATH = 'output/brandenburg-gate.glb'
RENDER_DIR = 'output'


def render_with_pyrender(glb_path):
    import pyrender
    tm_scene = trimesh.load(glb_path)
    center = tm_scene.centroid
    ext = max(tm_scene.extents)

    pr_scene = pyrender.Scene(bg_color=[0.6, 0.75, 0.9, 1.0], ambient_light=[0.3, 0.3, 0.3])
    for name, geom in tm_scene.geometry.items():
        try:
            xform = tm_scene.graph.get(name)[0] if name in tm_scene.graph.nodes else np.eye(4)
        except Exception:
            xform = np.eye(4)
        geom.fix_normals()
        mat = geom.visual.material if hasattr(geom.visual, 'material') and geom.visual.material else None
        color = list(mat.baseColorFactor) if mat and hasattr(mat, 'baseColorFactor') and mat.baseColorFactor is not None else [0.8, 0.8, 0.8, 1.0]
        pr_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=0.1, roughnessFactor=0.5, doubleSided=True)
        pr_mesh = pyrender.Mesh.from_trimesh(geom, material=pr_mat, smooth=True)
        pr_scene.add(pr_mesh, pose=xform)

    up = np.array([0, 1, 0])
    sun = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=4.0)
    sun_pose = np.eye(4)
    sun_dir = np.array([0.5, -0.8, -0.3]); sun_dir = sun_dir / np.linalg.norm(sun_dir)
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
    dist = ext * 2.0
    paths = []

    # View 1: Front view (looking at the gate face)
    eye1 = center + np.array([0, dist * 0.3, dist])
    pr_scene.set_pose(cam_node, _look_at(eye1, center))
    color1, _ = renderer.render(pr_scene)
    p1 = os.path.join(RENDER_DIR, 'brandenburg-gate-render-1.png')
    Image.fromarray(color1).save(p1); paths.append(p1)
    print(f"  Rendered {p1}")

    # View 2: 3/4 angled view
    eye2 = center + np.array([dist * 0.6, dist * 0.4, dist * 0.6])
    pr_scene.set_pose(cam_node, _look_at(eye2, center))
    color2, _ = renderer.render(pr_scene)
    p2 = os.path.join(RENDER_DIR, 'brandenburg-gate-render-2.png')
    Image.fromarray(color2).save(p2); paths.append(p2)
    print(f"  Rendered {p2}")

    # View 3: Side view (showing depth)
    eye3 = center + np.array([dist, dist * 0.3, 0])
    pr_scene.set_pose(cam_node, _look_at(eye3, center))
    color3, _ = renderer.render(pr_scene)
    p3 = os.path.join(RENDER_DIR, 'brandenburg-gate-render-3.png')
    Image.fromarray(color3).save(p3); paths.append(p3)
    print(f"  Rendered {p3}")

    renderer.delete()
    return paths


def evaluate_with_gpt(render_paths):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    content = [{"type": "text", "text": """You are evaluating a 3D model of the Brandenburg Gate in Berlin, Germany.

Image 1: Front view (facing the gate)
Image 2: 3/4 angled view
Image 3: Side view (showing depth)

This is a LOW-POLY architectural model. Score it for SHAPE, PROPORTIONS, and SILHOUETTE.

Score the model 0-100% on geometric realism. Focus on:
1. Column arrangement — 12 Doric columns (6 per row, 2 rows) creating 5 passageways
2. Proportions — ~65m wide, ~26m tall, relatively shallow depth (~11m)
3. Entablature — horizontal beam across the top of the columns
4. Attic — raised central section above the entablature
5. Quadriga — chariot/statue element on top (can be a simplified block)
6. Overall gate-like appearance — does it read as a gateway, not a building?

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
    paths = render_with_pyrender(GLB_PATH)
    print("Sending to GPT-5.2 for evaluation...")
    result = evaluate_with_gpt(paths)
    print("\n" + "=" * 60)
    print("GPT-5.2 EVALUATION")
    print("=" * 60)
    print(result.encode('ascii', 'replace').decode('ascii'))
