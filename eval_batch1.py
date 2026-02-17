import asyncio, sys, os, base64, math, io, json, time
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(r'c:\Users\chesk\.cursor-tutor\CityBuilder')
from citybuilder import CityBuilder, BoundingBox

API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith('OPENAI_API_KEY='):
                API_KEY = line.split('=', 1)[1].strip()

RENDER_DIR = 'output/downtown-renders'
os.makedirs(RENDER_DIR, exist_ok=True)

OFFSET = 0.003
DOWNTOWNS = [
    {"name": "Downtown San Francisco (FiDi)", "lat": 37.7946, "lon": -122.3999},
    {"name": "Downtown Los Angeles (DTLA)", "lat": 34.0522, "lon": -118.2437},
    {"name": "Downtown Vancouver", "lat": 49.2827, "lon": -123.1207},
    {"name": "Downtown Philadelphia", "lat": 39.9526, "lon": -75.1652},
]

def build_downtown(entry):
    lat, lon = entry["lat"], entry["lon"]
    slug = entry["name"].lower().replace(" ", "-").replace("(", "").replace(")", "")
    filename = f"downtown-{slug}.glb"
    bbox = BoundingBox(north=lat+OFFSET, south=lat-OFFSET, east=lon+OFFSET, west=lon-OFFSET)
    try:
        builder = CityBuilder()
        city_id = asyncio.run(builder.process_city(bbox))
        result_path = builder.generate_glb(city_id, filename)
        return str(result_path)
    except Exception as e:
        print(f"  BUILD FAILED: {e}")
        return None

def render_glb(glb_path, name_slug):
    import pyrender
    tm_scene = trimesh.load(glb_path)
    if not tm_scene.geometry: return []
    center = tm_scene.centroid
    ext = max(tm_scene.extents)
    pr_scene = pyrender.Scene(bg_color=[0.6, 0.75, 0.9, 1.0], ambient_light=[0.3, 0.3, 0.3])
    for geom_name, geom in tm_scene.geometry.items():
        try: xform = tm_scene.graph.get(geom_name)[0] if geom_name in tm_scene.graph.nodes else np.eye(4)
        except: xform = np.eye(4)
        geom.fix_normals()
        if not hasattr(geom.visual, 'material') or geom.visual.material is None: color = [0.8, 0.8, 0.8, 1.0]
        else:
            mat = geom.visual.material
            color = list(mat.baseColorFactor) if hasattr(mat, 'baseColorFactor') and mat.baseColorFactor is not None else [0.8, 0.8, 0.8, 1.0]
        pr_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=0.1, roughnessFactor=0.5, doubleSided=True)
        pr_mesh = pyrender.Mesh.from_trimesh(geom, material=pr_mat, smooth=True)
        pr_scene.add(pr_mesh, pose=xform)
    up = np.array([0, 1, 0])
    sun = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=4.0)
    sun_pose = np.eye(4); sun_dir = np.array([0.5, -0.8, -0.3]); sun_dir /= np.linalg.norm(sun_dir)
    right = np.cross(-sun_dir, up); right /= np.linalg.norm(right); true_up = np.cross(right, -sun_dir)
    sun_pose[:3, 0] = right; sun_pose[:3, 1] = true_up; sun_pose[:3, 2] = -sun_dir; sun_pose[:3, 3] = center + np.array([0, ext, 0])
    pr_scene.add(sun, pose=sun_pose)
    fill = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.5)
    fill_pose = np.eye(4); fill_dir = np.array([-0.3, -0.5, 0.5]); fill_dir /= np.linalg.norm(fill_dir)
    fr = np.cross(-fill_dir, up); fr /= np.linalg.norm(fr); fu = np.cross(fr, -fill_dir)
    fill_pose[:3, 0] = fr; fill_pose[:3, 1] = fu; fill_pose[:3, 2] = -fill_dir
    pr_scene.add(fill, pose=fill_pose)
    renderer = pyrender.OffscreenRenderer(1280, 960)
    def _look_at(eye, target):
        fwd = np.array(target) - np.array(eye); fwd /= np.linalg.norm(fwd)
        r = np.cross(fwd, np.array([0, 1, 0])); r /= np.linalg.norm(r); u = np.cross(r, fwd)
        pose = np.eye(4); pose[:3, 0] = r; pose[:3, 1] = u; pose[:3, 2] = -fwd; pose[:3, 3] = eye
        return pose
    cam = pyrender.PerspectiveCamera(yfov=math.radians(45)); cam_node = pr_scene.add(cam)
    dist = ext * 1.4; paths = []
    elev1 = math.radians(30); azim1 = math.radians(-35)
    eye1 = center + np.array([dist*math.cos(elev1)*math.sin(azim1), dist*math.sin(elev1), dist*math.cos(elev1)*math.cos(azim1)])
    pr_scene.set_pose(cam_node, _look_at(eye1, center)); color1, _ = renderer.render(pr_scene)
    p1 = os.path.join(RENDER_DIR, f'{name_slug}-1.png'); Image.fromarray(color1).save(p1); paths.append(p1)
    elev2 = math.radians(18); azim2 = math.radians(120)
    eye2 = center + np.array([dist*math.cos(elev2)*math.sin(azim2), dist*math.sin(elev2), dist*math.cos(elev2)*math.cos(azim2)])
    pr_scene.set_pose(cam_node, _look_at(eye2, center)); color2, _ = renderer.render(pr_scene)
    p2 = os.path.join(RENDER_DIR, f'{name_slug}-2.png'); Image.fromarray(color2).save(p2); paths.append(p2)
    renderer.delete(); return paths

def evaluate_downtown(name, render_paths):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    content = [{"type": "text", "text": f"""You are evaluating a 3D city model of "{name}".
Image 1: Elevated 3/4 view    Image 2: Lower angle opposite side
Auto-generated from OSM data + Overture heights + satellite imagery. Score 0-100%.
Evaluate: building heights, density, terrain/ground, vegetation, overall impression, glitches.
Respond EXACTLY:
SCORE: XX%
STRENGTHS: <brief>
WEAKNESSES: <brief>
GLITCHES: <any visual bugs or none>
OVERALL: <one sentence>"""}]
    for p in render_paths:
        img = Image.open(p); img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
    response = client.chat.completions.create(model="gpt-5.2-chat-latest", messages=[{"role": "user", "content": content}], max_completion_tokens=500)
    return response.choices[0].message.content

results = []
for i, entry in enumerate(DOWNTOWNS):
    name = entry["name"]
    slug = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
    print(f"\n[{i+1}/{len(DOWNTOWNS)}] {name}")
    t0 = time.time()
    glb_path = build_downtown(entry)
    bt = time.time() - t0
    if not glb_path: continue
    size_kb = os.path.getsize(glb_path) / 1024
    print(f"  Built {bt:.1f}s ({size_kb:.0f} KB)")
    try:
        rp = render_glb(glb_path, slug)
        if not rp: print("  SKIP"); continue
    except Exception as e: print(f"  RENDER FAIL: {e}"); continue
    try:
        txt = evaluate_downtown(name, rp)
        score = None
        for line in txt.split('\n'):
            if line.strip().upper().startswith('SCORE:'):
                try: score = int(line.split(':')[1].strip().replace('%',''))
                except: pass
        print(f"  Score: {score}%")
        print(f"  {txt}")
        results.append({"name": name, "score": score, "response": txt, "size_kb": round(size_kb,1), "build_time": round(bt,1)})
    except Exception as e: print(f"  EVAL FAIL: {e}"); continue
    time.sleep(2)

print("\n=== BATCH 1 SUMMARY ===")
for r in sorted(results, key=lambda x: x.get('score') or 0, reverse=True):
    print(f"  {r['score']}%  {r['name']}")
