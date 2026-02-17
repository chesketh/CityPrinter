"""Evaluate Opera House model using GPT-5.2 Vision.

Renders the GLB to PNGs from multiple angles using pyrender (PBR lighting),
then sends them to GPT-5.2 alongside a reference photo for comparison.
"""
import sys, os, base64, math, io
import numpy as np
import trimesh
from PIL import Image

# API key from environment or .env
API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith('OPENAI_API_KEY='):
                API_KEY = line.split('=', 1)[1].strip()

GLB_PATH = 'output/sydney-opera-house.glb'
RENDER_DIR = 'output'

# Reference image (public domain Wikimedia)
REFERENCE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Sydney_Australia._%2821339175489%29.jpg/1280px-Sydney_Australia._%2821339175489%29.jpg"


def render_with_pyrender(glb_path):
    """Render GLB with pyrender for proper PBR lighting."""
    import pyrender

    # Load the GLB as a trimesh scene
    tm_scene = trimesh.load(glb_path)
    center = tm_scene.centroid
    ext = max(tm_scene.extents)

    # Convert trimesh scene to pyrender scene
    pr_scene = pyrender.Scene(
        bg_color=[0.6, 0.75, 0.9, 1.0],  # light blue sky
        ambient_light=[0.3, 0.3, 0.3],
    )

    # Add each mesh from the trimesh scene
    for name, geom in tm_scene.geometry.items():
        # Get the transform for this geometry
        node_names = [n for n in tm_scene.graph.nodes if n == name]
        xform = tm_scene.graph.get(name)[0] if name in tm_scene.graph.nodes else np.eye(4)

        # Create pyrender mesh with smooth normals
        geom.fix_normals()
        if not hasattr(geom.visual, 'material') or geom.visual.material is None:
            color = [0.8, 0.8, 0.8, 1.0]
        else:
            mat = geom.visual.material
            if hasattr(mat, 'baseColorFactor') and mat.baseColorFactor is not None:
                color = list(mat.baseColorFactor)
            else:
                color = [0.8, 0.8, 0.8, 1.0]

        pr_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.4,
            doubleSided=True,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(geom, material=pr_mat, smooth=True)
        pr_scene.add(pr_mesh, pose=xform)

    # Add strong directional light (sun from upper-left)
    sun = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=4.0)
    sun_pose = np.eye(4)
    # Light direction: from upper-left-front
    sun_dir = np.array([0.5, -0.8, -0.3])
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    # Build rotation matrix for the light
    up = np.array([0, 1, 0])
    right = np.cross(-sun_dir, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, -sun_dir)
    sun_pose[:3, 0] = right
    sun_pose[:3, 1] = true_up
    sun_pose[:3, 2] = -sun_dir
    sun_pose[:3, 3] = center + np.array([0, ext, 0])
    pr_scene.add(sun, pose=sun_pose)

    # Add fill light from opposite side
    fill = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.5)
    fill_pose = np.eye(4)
    fill_dir = np.array([-0.3, -0.5, 0.5])
    fill_dir = fill_dir / np.linalg.norm(fill_dir)
    fill_right = np.cross(-fill_dir, up)
    fill_right = fill_right / np.linalg.norm(fill_right)
    fill_up = np.cross(fill_right, -fill_dir)
    fill_pose[:3, 0] = fill_right
    fill_pose[:3, 1] = fill_up
    fill_pose[:3, 2] = -fill_dir
    pr_scene.add(fill, pose=fill_pose)

    # Set up offscreen renderer
    renderer = pyrender.OffscreenRenderer(1280, 960)

    def _look_at(eye, target):
        """Build camera pose matrix."""
        fwd = np.array(target) - np.array(eye)
        fwd = fwd / np.linalg.norm(fwd)
        r = np.cross(fwd, np.array([0, 1, 0]))
        r = r / np.linalg.norm(r)
        u = np.cross(r, fwd)
        pose = np.eye(4)
        pose[:3, 0] = r
        pose[:3, 1] = u
        pose[:3, 2] = -fwd
        pose[:3, 3] = eye
        return pose

    cam = pyrender.PerspectiveCamera(yfov=math.radians(45))
    cam_node = pr_scene.add(cam)

    dist = ext * 2.0
    paths = []

    # View 1: South-southeast elevated (classic harbor photo angle)
    elev = math.radians(20)
    azim = math.radians(-50)
    eye_se = center + np.array([
        dist * math.cos(elev) * math.sin(azim),
        dist * math.sin(elev),
        dist * math.cos(elev) * math.cos(azim),
    ])
    pr_scene.set_pose(cam_node, _look_at(eye_se, center))
    color_se, _ = renderer.render(pr_scene)
    p1 = os.path.join(RENDER_DIR, 'opera-render-se.png')
    Image.fromarray(color_se).save(p1)
    paths.append(p1)
    print(f"  Rendered {p1}")

    # View 2: From above at ~50deg
    eye_top = center + np.array([0, dist * 0.9, dist * 0.3])
    pr_scene.set_pose(cam_node, _look_at(eye_top, center))
    color_top, _ = renderer.render(pr_scene)
    p2 = os.path.join(RENDER_DIR, 'opera-render-top.png')
    Image.fromarray(color_top).save(p2)
    paths.append(p2)
    print(f"  Rendered {p2}")

    # View 3: From north (harbor) looking south — classic postcard angle
    azim3 = math.radians(110)
    elev3 = math.radians(10)
    eye_n = center + np.array([
        dist * math.cos(elev3) * math.sin(azim3),
        dist * math.sin(elev3),
        dist * math.cos(elev3) * math.cos(azim3),
    ])
    pr_scene.set_pose(cam_node, _look_at(eye_n, center))
    color_n, _ = renderer.render(pr_scene)
    p3 = os.path.join(RENDER_DIR, 'opera-render-north.png')
    Image.fromarray(color_n).save(p3)
    paths.append(p3)
    print(f"  Rendered {p3}")

    renderer.delete()
    return paths


def render_model_fallback(glb_path):
    """Fallback: trimesh renderer if pyrender fails."""
    scene = trimesh.load(glb_path)
    center = scene.centroid
    ext = max(scene.extents)

    dist = ext * 2.0
    elev_rad = math.radians(15)
    azim_rad = math.radians(-70)
    eye_se = center + np.array([
        dist * math.cos(elev_rad) * math.sin(azim_rad),
        dist * math.sin(elev_rad),
        dist * math.cos(elev_rad) * math.cos(azim_rad),
    ])
    eye_top = center + np.array([0, dist * 0.9, dist * 0.3])

    def _set_cam(sc, eye, target):
        fwd = np.array(target) - np.array(eye)
        fwd = fwd / np.linalg.norm(fwd)
        r = np.cross(fwd, np.array([0, 1, 0]))
        r = r / np.linalg.norm(r)
        u = np.cross(r, fwd)
        xf = np.eye(4)
        xf[:3, 0] = r
        xf[:3, 1] = u
        xf[:3, 2] = -fwd
        xf[:3, 3] = eye
        sc.camera_transform = xf
        sc.camera.fov = (45, 45)

    paths = []
    p1 = os.path.join(RENDER_DIR, 'opera-render-se.png')
    _set_cam(scene, eye_se, center)
    with open(p1, 'wb') as f:
        f.write(scene.save_image(resolution=(1280, 960)))
    paths.append(p1)
    print(f"  Rendered {p1} (fallback)")

    p2 = os.path.join(RENDER_DIR, 'opera-render-top.png')
    _set_cam(scene, eye_top, center)
    with open(p2, 'wb') as f:
        f.write(scene.save_image(resolution=(1280, 960)))
    paths.append(p2)
    print(f"  Rendered {p2} (fallback)")

    return paths


def evaluate_with_gpt(render_paths, reference_url):
    """Send rendered images + reference to GPT-5.2 for evaluation."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    content = [{"type": "text", "text": """You are evaluating a 3D model of the Sydney Opera House.

Image 1: 3D model rendered from the south-southeast
Image 2: 3D model rendered from above
Image 3: 3D model rendered from the north (harbor side, classic postcard view)
Image 4: Reference photo of the real Sydney Opera House

This is a LOW-POLY architectural model (not a photorealistic render). Score it appropriately for that medium — focus on SHAPE, PROPORTIONS, and SILHOUETTE rather than material detail or photorealism.

Score the model 0-100% on geometric realism. Focus on:
1. Shell/sail shape — are they broad curved sails (good) or thin blades (bad)?
2. Shell nesting — do shells cascade from tallest at back to shortest at front?
3. Number of shells per group (real has ~4 per hall group)
4. Podium shape (real is tapered, wide at entrance, narrow at tip)
5. Overall silhouette — would someone recognize this as the Opera House?

SCORE: XX%
Then explain what's right, what's wrong, and the single most impactful geometry change."""}]

    for p in render_paths:
        # Resize to keep under 5MB: convert to JPEG at quality=85, max 1024px
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
        print(f"  pyrender failed ({e}), using trimesh fallback...")
        paths = render_model_fallback(GLB_PATH)

    print("Sending to GPT-5.2 for evaluation...")
    result = evaluate_with_gpt(paths, REFERENCE_URL)
    print("\n" + "=" * 60)
    print("GPT-5.2 EVALUATION")
    print("=" * 60)
    print(result.encode('ascii', 'replace').decode('ascii'))
