#!/usr/bin/env python3
"""Generate interactive 3D visualization of semantic directions in V34 embeddings.

Produces a self-contained HTML file at docs/semantic_3d.html using Three.js,
showing how semantic axes (size, temperature, strength, etc.) and cyclic groups
(numbers, months, days) are arranged in embedding space.

Usage:
    python generate_semantic_3d.py
"""

import json
import os
import numpy as np
import torch

# ── Configuration ──────────────────────────────────────────────────────────

OUTPUT_PATH = "docs/semantic_3d.html"
CHECKPOINT = "checkpoints/word2vec_v34/latest.pt"
VOCAB_PATH = "checkpoints/word2vec_v28/vocab.json"
EMBED_KEY = "target_embeddings.weight"

SEMANTIC_AXES = {
    "small → big": ["tiny", "small", "little", "medium", "large", "big", "huge", "enormous", "giant", "massive"],
    "cold → hot": ["freezing", "cold", "cool", "warm", "hot", "burning", "boiling"],
    "weak → strong": ["weak", "feeble", "fragile", "moderate", "firm", "strong", "powerful", "mighty"],
    "slow → fast": ["slow", "sluggish", "steady", "quick", "fast", "rapid", "swift"],
    "near → far": ["near", "close", "nearby", "distant", "far", "remote"],
    "numbers 1-10": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "months": ["january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november", "december"],
    "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
}

LINEAR_AXES = ["small → big", "cold → hot", "weak → strong", "slow → fast", "near → far"]
CYCLIC_AXES = ["numbers 1-10", "months", "days"]

# Colors per semantic group
AXIS_COLORS = {
    "small → big": "#ff6b6b",
    "cold → hot": "#4ecdc4",
    "weak → strong": "#ffd93d",
    "slow → fast": "#6bcb77",
    "near → far": "#ff8e53",
    "numbers 1-10": "#a78bfa",
    "months": "#f472b6",
    "days": "#38bdf8",
}

CONTEXT_CATEGORIES = {
    "animals": {
        "words": ["dog", "cat", "horse", "bird", "fish", "bear", "lion", "tiger",
                  "wolf", "deer", "eagle", "shark", "whale", "snake", "rabbit",
                  "mouse", "cow", "pig", "sheep", "fox", "monkey", "elephant",
                  "dolphin", "frog", "turtle"],
        "color": "#8B4513",
    },
    "countries": {
        "words": ["france", "germany", "japan", "china", "india", "brazil",
                  "canada", "russia", "spain", "italy", "australia", "mexico",
                  "england", "sweden", "norway", "egypt", "korea", "turkey",
                  "argentina", "poland"],
        "color": "#228B22",
    },
    "emotions": {
        "words": ["happy", "sad", "angry", "fear", "love", "hate", "joy", "pain",
                  "hope", "anxiety", "excitement", "surprise", "shame", "guilt",
                  "pride", "jealousy", "grief", "rage", "terror", "disgust",
                  "lonely", "calm", "peaceful", "nervous", "frustrated"],
        "color": "#DC143C",
    },
    "family": {
        "words": ["mother", "father", "son", "daughter", "brother", "sister",
                  "husband", "wife", "parent", "child", "baby", "grandmother",
                  "grandfather", "uncle", "aunt", "cousin", "family", "mom",
                  "dad", "twin"],
        "color": "#FF69B4",
    },
    "professions": {
        "words": ["doctor", "teacher", "lawyer", "engineer", "nurse", "scientist",
                  "artist", "musician", "writer", "actor", "singer", "professor",
                  "surgeon", "architect", "pilot", "soldier", "chef", "farmer",
                  "judge", "detective"],
        "color": "#20B2AA",
    },
}

# Common background words (smaller gray dots)
BACKGROUND_WORDS = [
    "the", "and", "that", "have", "with", "this", "will", "from", "they",
    "been", "said", "each", "which", "their", "time", "very", "when",
    "come", "could", "now", "than", "first", "water", "long", "make",
    "thing", "see", "way", "look", "world",
]


# ── Model Loading ──────────────────────────────────────────────────────────

def load_model():
    """Load V34 embeddings + vocab, return normalized embeddings, w2i, i2w."""
    print("Loading vocabulary...")
    with open(VOCAB_PATH) as f:
        vocab_data = json.load(f)
    w2i = vocab_data["word2id"]
    i2w = {v: k for k, v in w2i.items()}

    print(f"Loading checkpoint from {CHECKPOINT}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    emb = state[EMBED_KEY]
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb, device=device)
    else:
        emb = emb.to(device)

    # L2-normalize
    norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    emb = emb / norms

    step = ckpt.get("step", "?")
    print(f"  Embeddings: {emb.shape}, device={emb.device}, step={step}")
    return emb, w2i, i2w, step


# ── Direction Computation ──────────────────────────────────────────────────

def compute_directions(embeddings, w2i):
    """Compute direction vectors for each semantic axis.

    Returns dict of axis_name -> list of direction vectors (1 for linear, 2 for cyclic).
    Each direction is a unit vector in embedding space.
    """
    device = embeddings.device
    directions = {}

    for axis_name in LINEAR_AXES:
        words = SEMANTIC_AXES[axis_name]
        valid = [w for w in words if w in w2i]
        if len(valid) < 4:
            print(f"  WARNING: {axis_name} only has {len(valid)} valid words, skipping")
            continue
        indices = torch.tensor([w2i[w] for w in valid], device=device)
        vecs = embeddings[indices]
        # Direction = mean(last 3) - mean(first 3)
        n_end = min(3, len(valid) // 2)
        direction = vecs[-n_end:].mean(dim=0) - vecs[:n_end].mean(dim=0)
        direction = direction / direction.norm().clamp(min=1e-8)
        directions[axis_name] = [direction]
        print(f"  {axis_name}: {len(valid)} words, direction computed")

    for axis_name in CYCLIC_AXES:
        words = SEMANTIC_AXES[axis_name]
        valid = [w for w in words if w in w2i]
        if len(valid) < 4:
            print(f"  WARNING: {axis_name} only has {len(valid)} valid words, skipping")
            continue
        indices = torch.tensor([w2i[w] for w in valid], device=device)
        vecs = embeddings[indices]
        # PCA: center, then SVD
        centered = vecs - vecs.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        pc1 = Vh[0]  # first principal component
        pc2 = Vh[1]  # second principal component
        pc1 = pc1 / pc1.norm().clamp(min=1e-8)
        pc2 = pc2 / pc2.norm().clamp(min=1e-8)
        directions[axis_name] = [pc1, pc2]
        print(f"  {axis_name}: {len(valid)} words, PC1+PC2 computed (var: {S[0]**2:.2f}, {S[1]**2:.2f})")

    return directions


# ── Projection Computation ─────────────────────────────────────────────────

def compute_projections(directions, embeddings, word_indices, w2i):
    """Compute multiple 3D projections of the word embeddings.

    Returns dict of projection_name -> {
        'coords': Nx3 numpy array,
        'arrows': list of (name, direction_3d_unit_vector),
        'preservation': dict of axis_name -> cosine preservation score
    }
    """
    device = embeddings.device

    # Gather all direction vectors for SVD-based projection
    all_dirs = []
    dir_labels = []
    for axis_name, vecs in directions.items():
        for i, v in enumerate(vecs):
            all_dirs.append(v)
            suffix = f" PC{i+1}" if len(vecs) > 1 else ""
            dir_labels.append(f"{axis_name}{suffix}")

    D = torch.stack(all_dirs)  # (K, 300)
    print(f"\n  Direction matrix: {D.shape} ({len(dir_labels)} directions)")

    # Get word embeddings for the selected words
    idx_tensor = torch.tensor(word_indices, device=device)
    W = embeddings[idx_tensor]  # (N, 300)

    projections = {}

    # --- Projection 1: SVD of all directions ---
    def make_projection(basis_vectors, name, basis_labels):
        """Project words and directions onto a 3D basis."""
        B = torch.stack(basis_vectors)  # (3, 300)
        # Project words
        coords = (W @ B.T).cpu().numpy()  # (N, 3)
        # Project direction vectors for arrows
        arrows = []
        for label, d in zip(dir_labels, all_dirs):
            proj_d = (d @ B.T).cpu().numpy()
            norm = np.linalg.norm(proj_d)
            if norm > 0.05:  # Only show if reasonably preserved
                arrows.append((label, (proj_d / max(norm, 1e-8)).tolist(), float(norm)))
        # Preservation: cosine between original direction and its projection
        preservation = {}
        for label, d in zip(dir_labels, all_dirs):
            proj_d = B.T @ (B @ d)  # project then reconstruct in 300d
            cos = float(torch.dot(d, proj_d) / (d.norm() * proj_d.norm()).clamp(min=1e-8))
            preservation[label] = round(cos, 3)
        return {
            'coords': coords,
            'arrows': arrows,
            'preservation': preservation,
            'basis_labels': basis_labels,
        }

    # SVD of direction matrix to find best 3D subspace
    U_d, S_d, Vh_d = torch.linalg.svd(D, full_matrices=False)
    svd_basis = [Vh_d[0], Vh_d[1], Vh_d[2]]
    projections["All Directions (SVD)"] = make_projection(
        svd_basis, "SVD", ["SVD-1", "SVD-2", "SVD-3"]
    )
    print(f"  SVD singular values: {S_d[:5].cpu().numpy().round(3)}")

    # --- Projection 2: Size x Temperature x Speed ---
    if "small → big" in directions and "cold → hot" in directions and "slow → fast" in directions:
        basis = [directions["small → big"][0], directions["cold → hot"][0], directions["slow → fast"][0]]
        # Gram-Schmidt to orthogonalize
        basis = _gram_schmidt(basis)
        projections["Size x Temperature x Speed"] = make_projection(
            basis, "STS", ["small→big", "cold→hot", "slow→fast"]
        )

    # --- Projection 3: Numbers x Months x Days (PC1 of each) ---
    if all(a in directions for a in CYCLIC_AXES):
        basis = [directions[a][0] for a in CYCLIC_AXES]
        basis = _gram_schmidt(basis)
        projections["Numbers x Months x Days"] = make_projection(
            basis, "Cyclic", ["numbers-PC1", "months-PC1", "days-PC1"]
        )

    # --- Projection 4: Strength x Distance x Months ---
    if "weak → strong" in directions and "near → far" in directions and "months" in directions:
        basis = [directions["weak → strong"][0], directions["near → far"][0], directions["months"][0]]
        basis = _gram_schmidt(basis)
        projections["Strength x Distance x Months"] = make_projection(
            basis, "SDM", ["weak→strong", "near→far", "months-PC1"]
        )

    return projections


def _gram_schmidt(vectors):
    """Orthogonalize a list of vectors via Gram-Schmidt."""
    result = []
    for v in vectors:
        v = v.clone()
        for u in result:
            v = v - torch.dot(v, u) * u
        norm = v.norm().clamp(min=1e-8)
        result.append(v / norm)
    return result


# ── HTML Generation ────────────────────────────────────────────────────────

def build_html(projections, words, word_categories, word_colors,
               axis_word_set, axis_paths, step):
    """Build self-contained HTML with Three.js 3D visualization."""

    # Prepare projection data as JSON
    proj_data = {}
    for proj_name, pdata in projections.items():
        coords = pdata['coords']
        # Normalize coords to a reasonable range (roughly -2 to 2)
        scale = np.percentile(np.abs(coords), 95)
        if scale > 0:
            coords = coords / scale * 2.0
        proj_data[proj_name] = {
            'coords': [[round(float(c), 4) for c in row] for row in coords],
            'arrows': [(name, [round(float(x), 4) for x in d], round(p, 3))
                       for name, d, p in pdata['arrows']],
            'preservation': pdata['preservation'],
            'basis_labels': pdata['basis_labels'],
        }

    # Build word data
    word_data = []
    for i, w in enumerate(words):
        cat = word_categories.get(w, "background")
        col = word_colors.get(w, "#555555")
        is_axis = w in axis_word_set
        word_data.append({
            'w': w,
            'c': cat,
            'col': col,
            'ax': is_axis,
        })

    # Build axis paths (for connecting lines)
    # axis_paths: dict of axis_name -> list of word indices into `words`
    path_data = {}
    for axis_name, widxs in axis_paths.items():
        path_data[axis_name] = {
            'indices': widxs,
            'color': AXIS_COLORS.get(axis_name, "#ffffff"),
        }

    # Categories present
    used_cats = sorted(set(word_categories.values()))
    cat_colors = {}
    for cat in used_cats:
        # Find a word in this category and use its color
        for w, c in word_categories.items():
            if c == cat:
                cat_colors[cat] = word_colors[w]
                break

    word_json = json.dumps(word_data, separators=(',', ':'))
    proj_json = json.dumps(proj_data, separators=(',', ':'))
    path_json = json.dumps(path_data, separators=(',', ':'))

    # Build toggle HTML
    toggle_html = ""
    # Axis groups first
    for axis_name in list(SEMANTIC_AXES.keys()):
        col = AXIS_COLORS.get(axis_name, "#fff")
        safe_name = axis_name.replace("→", "to").replace(" ", "_")
        toggle_html += f'    <label><input type="checkbox" class="cat-toggle" data-cat="{axis_name}" checked><span class="leg-dot" style="background:{col}"></span>{axis_name}</label>\n'
    toggle_html += '    <hr style="border-color:#333;margin:6px 0">\n'
    # Context categories
    for cat in sorted(CONTEXT_CATEGORIES.keys()):
        col = CONTEXT_CATEGORIES[cat]["color"]
        toggle_html += f'    <label><input type="checkbox" class="cat-toggle" data-cat="{cat}" checked><span class="leg-dot" style="background:{col}"></span>{cat}</label>\n'
    toggle_html += f'    <label><input type="checkbox" class="cat-toggle" data-cat="background" checked><span class="leg-dot" style="background:#555"></span>background</label>\n'

    # Projection selector HTML
    proj_options = ""
    for i, pname in enumerate(proj_data.keys()):
        selected = " selected" if i == 0 else ""
        proj_options += f'      <option value="{pname}"{selected}>{pname}</option>\n'

    # Preservation table (generated in JS dynamically)

    step_str = f"{step:,}" if isinstance(step, int) else str(step)

    html = f'''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>V34 — Semantic Directions in 3D ({step_str} steps)</title>
<style>
body {{ margin: 0; overflow: hidden; background: #0a0a1a; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #ccc; }}
#info {{
    position: absolute; top: 10px; left: 10px;
    background: rgba(10,10,26,0.92); padding: 14px 18px; border-radius: 10px;
    max-width: 340px; z-index: 10; font-size: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
}}
#info h2 {{ margin: 0 0 8px 0; color: #FFD700; font-size: 16px; }}
#info p {{ margin: 3px 0; }}
#info .stat {{ color: #4fc3f7; }}
#info .muted {{ color: #888; font-size: 11px; }}
#preservation-table {{
    margin-top: 8px; font-size: 11px; width: 100%;
    border-collapse: collapse;
}}
#preservation-table td {{
    padding: 2px 6px; border-bottom: 1px solid rgba(255,255,255,0.05);
}}
#preservation-table .bar-cell {{ width: 80px; }}
.pbar {{
    height: 8px; border-radius: 4px; background: #4fc3f7; display: inline-block;
    transition: width 0.5s ease;
}}
#tooltip {{
    position: absolute; display: none; background: rgba(10,10,26,0.95); color: #fff;
    padding: 10px 14px; border-radius: 8px; font-size: 13px; pointer-events: none; z-index: 20;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(4px);
}}
#controls {{
    position: absolute; top: 10px; right: 10px; color: #ccc; font-size: 12px;
    background: rgba(10,10,26,0.92); padding: 12px 16px; border-radius: 10px; z-index: 10;
    max-height: 90vh; overflow-y: auto; min-width: 200px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
}}
#controls h3 {{ margin: 0 0 6px 0; color: #FFD700; font-size: 13px; }}
#controls label {{ display: block; margin: 2px 0; cursor: pointer; font-size: 11px; }}
#controls input[type=checkbox] {{ margin-right: 5px; vertical-align: middle; }}
.leg-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; display: inline-block; vertical-align: middle; }}
#search-box {{
    width: 100%; padding: 5px 8px; margin: 4px 0 8px 0; background: #1a1a2e; border: 1px solid #444;
    color: #fff; border-radius: 5px; font-size: 12px; box-sizing: border-box;
}}
#proj-select {{
    width: 100%; padding: 5px 8px; margin: 4px 0 8px 0; background: #1a1a2e; border: 1px solid #444;
    color: #fff; border-radius: 5px; font-size: 12px; box-sizing: border-box;
}}
hr {{ border: none; border-top: 1px solid #333; margin: 8px 0; }}
</style>
</head><body>

<div id="info">
    <h2>V34 — Semantic Directions in 3D</h2>
    <p>Word2vec dynamic masking, <span class="stat">{step_str}</span> steps</p>
    <p><span class="stat">{len(words)}</span> words projected from 300d to 3d</p>
    <p class="muted">Drag to rotate | Scroll to zoom | Hover for details</p>
    <hr>
    <p style="color:#FFD700;font-size:11px;margin-bottom:4px;">Direction Preservation</p>
    <table id="preservation-table"></table>
</div>

<div id="tooltip"></div>

<div id="controls">
    <h3>Projection</h3>
    <select id="proj-select">
{proj_options}    </select>
    <hr>
    <h3>Search</h3>
    <input type="text" id="search-box" placeholder="Type a word...">
    <h3>Semantic Axes</h3>
{toggle_html}    <hr>
    <h3>Display</h3>
    <label><input type="checkbox" id="show-labels" checked> Show labels</label>
    <label><input type="checkbox" id="show-arrows" checked> Show direction arrows</label>
    <label><input type="checkbox" id="show-paths" checked> Show axis paths</label>
    <label><input type="checkbox" id="auto-rotate" checked> Auto-rotate</label>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
// ── Data ──────────────────────────────────────────────────────────────────
const words = {word_json};
const projections = {proj_json};
const axisPaths = {path_json};

// ── Scene setup ───────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);
scene.fog = new THREE.FogExp2(0x0a0a1a, 0.08);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.01, 200);
camera.position.set(4, 3, 4);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
dirLight.position.set(5, 8, 5);
scene.add(dirLight);

// ── Create word spheres ───────────────────────────────────────────────────
const allMeshes = [];
const meshByWord = {{}};
const pointGroups = {{}};
const labelSprites = [];

// Get initial projection
const projNames = Object.keys(projections);
let currentProj = projNames[0];
const initCoords = projections[currentProj].coords;

words.forEach((p, i) => {{
    const isAxis = p.ax;
    const isBg = p.c === 'background';
    const size = isAxis ? 0.06 : (isBg ? 0.02 : 0.04);
    const geo = new THREE.SphereGeometry(size, isAxis ? 12 : 8, isAxis ? 8 : 6);
    const color = new THREE.Color(p.col);
    const mat = new THREE.MeshPhongMaterial({{
        color: color,
        emissive: color,
        emissiveIntensity: isAxis ? 0.4 : 0.2,
        transparent: isBg,
        opacity: isBg ? 0.5 : 1.0,
    }});
    const mesh = new THREE.Mesh(geo, mat);
    const c = initCoords[i];
    mesh.position.set(c[0], c[1], c[2]);
    mesh.userData = {{ ...p, idx: i }};
    scene.add(mesh);
    allMeshes.push(mesh);
    meshByWord[p.w] = mesh;
    if (!pointGroups[p.c]) pointGroups[p.c] = [];
    pointGroups[p.c].push(mesh);
}});

// ── Labels (canvas texture sprites) ──────────────────────────────────────
function makeLabel(text, position, color) {{
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 64;
    ctx.font = 'bold 20px -apple-system, sans-serif';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.shadowColor = 'rgba(0,0,0,0.9)';
    ctx.shadowBlur = 4;
    ctx.fillText(text, 128, 38);
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false }});
    const sprite = new THREE.Sprite(mat);
    sprite.position.copy(position);
    sprite.position.y += 0.1;
    sprite.scale.set(0.5, 0.125, 1);
    return sprite;
}}

const labelsByCat = {{}};
words.forEach((p, i) => {{
    if (!p.ax) return;  // Only label axis words
    const c = initCoords[i];
    const pos = new THREE.Vector3(c[0], c[1], c[2]);
    const sprite = makeLabel(p.w, pos, p.col);
    sprite.userData = {{ cat: p.c, idx: i }};
    scene.add(sprite);
    labelSprites.push(sprite);
    if (!labelsByCat[p.c]) labelsByCat[p.c] = [];
    labelsByCat[p.c].push(sprite);
}});

// ── Axis paths (connecting lines) ─────────────────────────────────────────
const pathLines = {{}};
Object.entries(axisPaths).forEach(([axisName, info]) => {{
    const positions = [];
    info.indices.forEach(wi => {{
        const c = initCoords[wi];
        positions.push(c[0], c[1], c[2]);
    }});
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({{
        color: new THREE.Color(info.color),
        transparent: true,
        opacity: 0.6,
        linewidth: 1,
    }});
    const line = new THREE.Line(geo, mat);
    scene.add(line);
    pathLines[axisName] = line;
}});

// ── Direction arrows ──────────────────────────────────────────────────────
const arrowGroup = new THREE.Group();
scene.add(arrowGroup);
const arrowLabels = [];

function buildArrows() {{
    // Clear existing
    while (arrowGroup.children.length > 0) {{
        arrowGroup.remove(arrowGroup.children[0]);
    }}
    arrowLabels.forEach(s => scene.remove(s));
    arrowLabels.length = 0;

    const proj = projections[currentProj];
    proj.arrows.forEach(([name, dir, preservation]) => {{
        const d = new THREE.Vector3(dir[0], dir[1], dir[2]);
        const len = 1.5 * preservation;
        if (len < 0.1) return;
        const origin = new THREE.Vector3(0, 0, 0);
        const arrow = new THREE.ArrowHelper(d.normalize(), origin, len, 0xffffff, 0.12, 0.06);
        arrow.userData = {{ name: name }};
        arrowGroup.add(arrow);

        // Label at arrow tip
        const tipPos = d.clone().multiplyScalar(len + 0.15);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 512;
        canvas.height = 64;
        ctx.font = '18px -apple-system, sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.shadowColor = 'rgba(0,0,0,0.9)';
        ctx.shadowBlur = 4;
        // Truncate long names
        const shortName = name.length > 20 ? name.slice(0, 18) + '..' : name;
        ctx.fillText(shortName, 256, 40);
        const texture = new THREE.CanvasTexture(canvas);
        const mat = new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false, opacity: 0.7 }});
        const sprite = new THREE.Sprite(mat);
        sprite.position.copy(tipPos);
        sprite.scale.set(0.7, 0.09, 1);
        scene.add(sprite);
        arrowLabels.push(sprite);
    }});
}}
buildArrows();

// ── Preservation table ────────────────────────────────────────────────────
function updatePreservation() {{
    const table = document.getElementById('preservation-table');
    const proj = projections[currentProj];
    const pres = proj.preservation;
    let html = '';
    const entries = Object.entries(pres).sort((a, b) => b[1] - a[1]);
    entries.forEach(([name, val]) => {{
        const pct = Math.round(val * 100);
        const barW = Math.max(0, Math.min(100, pct));
        const color = val > 0.7 ? '#4ecdc4' : val > 0.4 ? '#ffd93d' : '#ff6b6b';
        html += '<tr><td style="color:#aaa;white-space:nowrap">' + name + '</td>';
        html += '<td class="bar-cell"><span class="pbar" style="width:' + barW + '%;background:' + color + '"></span></td>';
        html += '<td style="color:' + color + '">' + pct + '%</td></tr>';
    }});
    table.innerHTML = html;
}}
updatePreservation();

// ── Projection switching with LERP animation ─────────────────────────────
let animating = false;
let animFrame = 0;
const ANIM_FRAMES = 60;
let startCoords = null;
let targetCoords = null;

function switchProjection(newProj) {{
    if (newProj === currentProj || animating) return;
    startCoords = projections[currentProj].coords.map(c => [...c]);
    currentProj = newProj;
    targetCoords = projections[currentProj].coords.map(c => [...c]);
    animFrame = 0;
    animating = true;
    buildArrows();
    updatePreservation();
    // Update arrow visibility
    arrowGroup.visible = document.getElementById('show-arrows').checked;
    arrowLabels.forEach(s => s.visible = arrowGroup.visible);
}}

function easeInOut(t) {{
    return t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
}}

function updateAnimation() {{
    if (!animating) return;
    animFrame++;
    const t = easeInOut(Math.min(animFrame / ANIM_FRAMES, 1.0));

    words.forEach((p, i) => {{
        const s = startCoords[i];
        const e = targetCoords[i];
        const x = s[0] + (e[0] - s[0]) * t;
        const y = s[1] + (e[1] - s[1]) * t;
        const z = s[2] + (e[2] - s[2]) * t;
        allMeshes[i].position.set(x, y, z);
    }});

    // Update labels
    labelSprites.forEach(sprite => {{
        const i = sprite.userData.idx;
        const mesh = allMeshes[i];
        sprite.position.set(mesh.position.x, mesh.position.y + 0.1, mesh.position.z);
    }});

    // Update path lines
    Object.entries(axisPaths).forEach(([axisName, info]) => {{
        const line = pathLines[axisName];
        if (!line) return;
        const positions = line.geometry.attributes.position.array;
        info.indices.forEach((wi, j) => {{
            positions[j*3] = allMeshes[wi].position.x;
            positions[j*3+1] = allMeshes[wi].position.y;
            positions[j*3+2] = allMeshes[wi].position.z;
        }});
        line.geometry.attributes.position.needsUpdate = true;
    }});

    if (animFrame >= ANIM_FRAMES) {{
        animating = false;
    }}
}}

// ── Tooltip / Raycaster ───────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
raycaster.params.Mesh = {{ threshold: 0.05 }};
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');

renderer.domElement.addEventListener('mousemove', (e) => {{
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(allMeshes);
    if (hits.length > 0) {{
        const p = hits[0].object.userData;
        const pos = hits[0].object.position;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
        tooltip.innerHTML = '<b style="color:' + p.col + '">' + p.w + '</b> &nbsp;<span style="color:#888">[' + p.c + ']</span><br>' +
            '<span style="color:#666">x=' + pos.x.toFixed(3) + '  y=' + pos.y.toFixed(3) + '  z=' + pos.z.toFixed(3) + '</span>';
    }} else {{
        tooltip.style.display = 'none';
    }}
}});

// ── Search ────────────────────────────────────────────────────────────────
let highlightedMesh = null;
let highlightOrigColor = null;
let highlightOrigEmissive = null;

document.getElementById('search-box').addEventListener('input', (e) => {{
    const q = e.target.value.toLowerCase().trim();
    if (highlightedMesh) {{
        highlightedMesh.material.color.copy(highlightOrigColor);
        highlightedMesh.material.emissive.copy(highlightOrigEmissive);
        highlightedMesh.scale.set(1, 1, 1);
        highlightedMesh = null;
    }}
    if (q && meshByWord[q]) {{
        const mesh = meshByWord[q];
        highlightOrigColor = mesh.material.color.clone();
        highlightOrigEmissive = mesh.material.emissive.clone();
        highlightedMesh = mesh;
        mesh.material.color.set(0xffffff);
        mesh.material.emissive.set(0xffffff);
        mesh.scale.set(3, 3, 3);
        controls.target.copy(mesh.position);
    }}
}});

// ── Projection selector ──────────────────────────────────────────────────
document.getElementById('proj-select').addEventListener('change', (e) => {{
    switchProjection(e.target.value);
}});

// ── Category toggles ─────────────────────────────────────────────────────
document.querySelectorAll('.cat-toggle').forEach(cb => {{
    cb.addEventListener('change', () => {{
        const cat = cb.dataset.cat;
        const visible = cb.checked;
        if (pointGroups[cat]) pointGroups[cat].forEach(m => m.visible = visible);
        const showLabels = document.getElementById('show-labels').checked;
        if (labelsByCat[cat]) labelsByCat[cat].forEach(s => s.visible = visible && showLabels);
        if (pathLines[cat]) pathLines[cat].visible = visible && document.getElementById('show-paths').checked;
    }});
}});

// ── Display toggles ──────────────────────────────────────────────────────
document.getElementById('show-labels').addEventListener('change', (e) => {{
    const show = e.target.checked;
    labelSprites.forEach(s => {{
        const catCb = document.querySelector('.cat-toggle[data-cat="' + s.userData.cat + '"]');
        s.visible = show && (!catCb || catCb.checked);
    }});
}});

document.getElementById('show-arrows').addEventListener('change', (e) => {{
    arrowGroup.visible = e.target.checked;
    arrowLabels.forEach(s => s.visible = e.target.checked);
}});

document.getElementById('show-paths').addEventListener('change', (e) => {{
    const show = e.target.checked;
    Object.entries(pathLines).forEach(([name, line]) => {{
        const catCb = document.querySelector('.cat-toggle[data-cat="' + name + '"]');
        line.visible = show && (!catCb || catCb.checked);
    }});
}});

document.getElementById('auto-rotate').addEventListener('change', (e) => {{
    controls.autoRotate = e.target.checked;
}});

// ── Resize ────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ── Animate ───────────────────────────────────────────────────────────────
function animate() {{
    requestAnimationFrame(animate);
    updateAnimation();
    controls.update();
    renderer.render(scene, camera);
}}
animate();
</script>
</body></html>'''

    return html


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating Semantic 3D Visualization")
    print("=" * 60)

    # Load model
    embeddings, w2i, i2w, step = load_model()

    # Compute direction vectors
    print("\nComputing semantic directions...")
    directions = compute_directions(embeddings, w2i)

    # Collect all words to include
    print("\nCollecting words...")

    # 1. All axis words
    axis_word_set = set()
    word_to_axis = {}  # word -> axis_name (first match)
    for axis_name, words_list in SEMANTIC_AXES.items():
        for w in words_list:
            if w in w2i:
                axis_word_set.add(w)
                if w not in word_to_axis:
                    word_to_axis[w] = axis_name

    # 2. Context category words (only those in vocab)
    context_words = {}  # word -> category
    for cat, info in CONTEXT_CATEGORIES.items():
        count = 0
        for w in info["words"]:
            if w in w2i and w not in axis_word_set and w not in context_words:
                context_words[w] = cat
                count += 1
                if count >= 25:  # cap per category
                    break
        print(f"  {cat}: {count} words in vocab")

    # 3. Background words
    bg_words = set()
    for w in BACKGROUND_WORDS:
        if w in w2i and w not in axis_word_set and w not in context_words:
            bg_words.add(w)
    print(f"  background: {len(bg_words)} words")

    # Build unified word list
    all_words = []
    word_categories = {}
    word_colors = {}

    # Add axis words first
    for axis_name, words_list in SEMANTIC_AXES.items():
        for w in words_list:
            if w in w2i and w not in [x for x in all_words]:
                all_words.append(w)
                word_categories[w] = axis_name
                word_colors[w] = AXIS_COLORS.get(axis_name, "#ffffff")

    # Add context words
    for w, cat in context_words.items():
        all_words.append(w)
        word_categories[w] = cat
        word_colors[w] = CONTEXT_CATEGORIES[cat]["color"]

    # Add background words
    for w in sorted(bg_words):
        all_words.append(w)
        word_categories[w] = "background"
        word_colors[w] = "#555555"

    print(f"\n  Total words: {len(all_words)}")

    # Build word index mapping
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    word_indices = [w2i[w] for w in all_words]  # indices into embedding matrix

    # Build axis paths (indices into all_words list for connecting lines)
    axis_paths = {}
    for axis_name, words_list in SEMANTIC_AXES.items():
        path_indices = []
        for w in words_list:
            if w in word_to_idx:
                path_indices.append(word_to_idx[w])
        if len(path_indices) >= 2:
            axis_paths[axis_name] = path_indices

    # Compute projections
    print("\nComputing projections...")
    projections = compute_projections(directions, embeddings, word_indices, w2i)

    for pname, pdata in projections.items():
        avg_pres = np.mean(list(pdata['preservation'].values()))
        print(f"  {pname}: avg preservation = {avg_pres:.3f}")

    # Generate HTML
    print("\nGenerating HTML...")
    html = build_html(projections, all_words, word_categories, word_colors,
                      axis_word_set, axis_paths, step)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)

    print(f"\nWrote {OUTPUT_PATH} ({len(html):,} bytes)")
    print(f"  {len(all_words)} words, {len(projections)} projections")
    print(f"  Axis words: {len(axis_word_set)}")
    print(f"  Context words: {len(context_words)}")
    print(f"  Background words: {len(bg_words)}")


if __name__ == "__main__":
    main()
