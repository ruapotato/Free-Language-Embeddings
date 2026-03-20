#!/usr/bin/env python3
"""Generate an interactive 3D globe visualization of V33 word embeddings."""

import json
import torch
import numpy as np
from sklearn.decomposition import PCA

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/word2vec_v33/latest.pt"
VOCAB_FILE = "checkpoints/word2vec_v28/vocab.json"
OUTPUT = "docs/probe_v33_3d.html"
TOP_N = 3000
EMBED_KEY = "target_embeddings.weight"

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading vocab...")
with open(VOCAB_FILE) as f:
    vocab_data = json.load(f)

vocab = vocab_data["word2id"]  # word -> index
counts = vocab_data.get("counts", [])  # list of counts by index
idx2word = {v: k for k, v in vocab.items()}

print("Loading checkpoint...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
if "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
elif "state_dict" in ckpt:
    state = ckpt["state_dict"]
else:
    state = ckpt

embeddings = state[EMBED_KEY].numpy()
print(f"Embeddings shape: {embeddings.shape}")

# ── Select top N words by index (lower index = higher frequency) ───────────
# Skip index 0 (<UNK> or padding)
indices = list(range(1, min(TOP_N + 1, len(idx2word))))
words = [idx2word[i] for i in indices if i in idx2word]
vecs = np.array([embeddings[vocab[w]] for w in words])
print(f"Selected {len(words)} words")

# ── PCA → 3D ───────────────────────────────────────────────────────────────
print("Running PCA...")
pca = PCA(n_components=3)
coords3d = pca.fit_transform(vecs)
print(f"Explained variance: {pca.explained_variance_ratio_}")

# Normalize to unit sphere
norms = np.linalg.norm(coords3d, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-8)
coords3d = coords3d / norms

# ── Semantic categories ─────────────────────────────────────────────────────
categories = {
    "country": {
        "words": {"china", "japan", "india", "france", "germany", "italy", "spain",
                  "russia", "brazil", "canada", "mexico", "australia", "england",
                  "ireland", "sweden", "norway", "denmark", "poland", "greece",
                  "turkey", "egypt", "iran", "iraq", "korea", "vietnam", "thailand",
                  "indonesia", "pakistan", "nigeria", "kenya", "argentina", "chile",
                  "colombia", "peru", "portugal", "austria", "switzerland", "belgium",
                  "netherlands", "finland", "ukraine", "romania", "hungary", "czech",
                  "israel", "africa", "europe", "asia", "america"},
        "color": "#228B22"
    },
    "animal": {
        "words": {"dog", "cat", "horse", "fish", "bird", "bear", "wolf", "lion",
                  "tiger", "eagle", "shark", "whale", "snake", "deer", "rabbit",
                  "mouse", "rat", "cow", "pig", "sheep", "chicken", "duck", "fox",
                  "monkey", "elephant", "dolphin", "frog", "turtle", "butterfly",
                  "bee", "ant", "spider", "crab", "owl", "hawk", "goat", "donkey",
                  "buffalo", "leopard", "zebra", "giraffe", "penguin", "parrot",
                  "salmon", "tuna", "lobster", "octopus", "bat", "dragon"},
        "color": "#8B4513"
    },
    "color_word": {
        "words": {"red", "blue", "green", "yellow", "black", "white", "orange",
                  "purple", "pink", "brown", "gray", "grey", "golden", "silver",
                  "violet", "crimson", "scarlet", "azure", "ivory", "bronze"},
        "color": "#9370DB"
    },
    "emotion": {
        "words": {"happy", "sad", "angry", "fear", "love", "hate", "joy", "pain",
                  "hope", "anxiety", "depression", "excitement", "surprise", "shame",
                  "guilt", "pride", "jealousy", "envy", "grief", "sorrow", "pleasure",
                  "delight", "rage", "fury", "terror", "horror", "disgust", "lonely",
                  "grateful", "confused", "bored", "nervous", "calm", "peaceful",
                  "worried", "stressed", "proud", "embarrassed", "frustrated"},
        "color": "#DC143C"
    },
    "science": {
        "words": {"science", "physics", "chemistry", "biology", "math", "mathematics",
                  "atom", "molecule", "cell", "gene", "dna", "protein", "electron",
                  "proton", "neutron", "gravity", "energy", "quantum", "theory",
                  "experiment", "hypothesis", "evolution", "species", "genome",
                  "bacteria", "virus", "enzyme", "radiation", "frequency", "wavelength",
                  "equation", "formula", "calculus", "algebra", "geometry", "oxygen",
                  "hydrogen", "carbon", "nitrogen", "helium"},
        "color": "#00CED1"
    },
    "body": {
        "words": {"head", "face", "eye", "eyes", "ear", "ears", "nose", "mouth",
                  "hand", "hands", "finger", "fingers", "arm", "arms", "leg", "legs",
                  "foot", "feet", "heart", "brain", "blood", "bone", "bones", "skin",
                  "hair", "teeth", "tooth", "tongue", "neck", "shoulder", "chest",
                  "stomach", "knee", "elbow", "wrist", "ankle", "hip", "thumb",
                  "lung", "liver", "kidney", "muscle"},
        "color": "#CD853F"
    },
    "food": {
        "words": {"food", "bread", "rice", "meat", "fish", "fruit", "apple", "banana",
                  "orange", "milk", "cheese", "butter", "sugar", "salt", "pepper",
                  "chicken", "beef", "pork", "egg", "eggs", "cake", "pie", "soup",
                  "coffee", "tea", "wine", "beer", "juice", "chocolate", "pizza",
                  "pasta", "salad", "sandwich", "potato", "tomato", "onion", "garlic",
                  "lemon", "strawberry", "grape", "corn", "wheat", "flour",
                  "honey", "cream", "bacon", "sausage", "mushroom"},
        "color": "#FF8C00"
    },
    "family": {
        "words": {"mother", "father", "son", "daughter", "brother", "sister",
                  "husband", "wife", "parent", "parents", "child", "children",
                  "baby", "grandmother", "grandfather", "uncle", "aunt", "cousin",
                  "nephew", "niece", "family", "mom", "dad", "grandma", "grandpa",
                  "twin", "sibling", "spouse", "ancestor", "descendant"},
        "color": "#FF69B4"
    },
    "profession": {
        "words": {"doctor", "teacher", "lawyer", "engineer", "nurse", "scientist",
                  "artist", "musician", "writer", "actor", "actress", "singer",
                  "painter", "photographer", "journalist", "professor", "surgeon",
                  "architect", "pilot", "soldier", "officer", "detective", "chef",
                  "farmer", "priest", "monk", "judge", "coach", "captain",
                  "manager", "director", "president", "minister", "ambassador",
                  "librarian", "accountant", "mechanic", "carpenter", "plumber"},
        "color": "#20B2AA"
    },
    "time": {
        "words": {"today", "tomorrow", "yesterday", "morning", "evening", "night",
                  "midnight", "noon", "dawn", "dusk", "monday", "tuesday",
                  "wednesday", "thursday", "friday", "saturday", "sunday",
                  "january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december",
                  "spring", "summer", "autumn", "winter", "year", "month", "week",
                  "hour", "minute", "second", "century", "decade", "season"},
        "color": "#DAA520"
    },
    "nature": {
        "words": {"mountain", "river", "ocean", "sea", "lake", "forest", "desert",
                  "island", "valley", "hill", "cliff", "cave", "beach", "volcano",
                  "glacier", "waterfall", "storm", "rain", "snow", "wind", "thunder",
                  "lightning", "cloud", "clouds", "sun", "moon", "star", "stars",
                  "sky", "earth", "fire", "water", "ice", "flame", "smoke",
                  "rainbow", "sunset", "sunrise", "tide", "wave", "flood"},
        "color": "#2E8B57"
    },
    "royalty": {
        "words": {"king", "queen", "prince", "princess", "emperor", "empress",
                  "throne", "crown", "kingdom", "palace", "royal", "duke",
                  "duchess", "knight", "lord", "lady", "noble", "monarch",
                  "dynasty", "reign", "sultan", "pharaoh"},
        "color": "#FFD700"
    },
    "music": {
        "words": {"music", "song", "guitar", "piano", "drum", "drums", "violin",
                  "trumpet", "orchestra", "concert", "album", "band", "choir",
                  "melody", "rhythm", "harmony", "bass", "jazz", "rock", "blues",
                  "opera", "symphony", "flute", "harp", "saxophone", "cello",
                  "singing", "lyric", "lyrics", "rap", "hip", "folk", "pop"},
        "color": "#E040FB"
    },
    "war": {
        "words": {"war", "battle", "army", "military", "weapon", "weapons", "bomb",
                  "missile", "tank", "rifle", "sword", "shield", "cannon", "bullet",
                  "troops", "invasion", "conquest", "siege", "combat", "warfare",
                  "navy", "admiral", "general", "colonel", "sergeant", "commander",
                  "assault", "retreat", "surrender", "victory", "defeat", "alliance"},
        "color": "#B71C1C"
    },
    "sport": {
        "words": {"football", "basketball", "baseball", "soccer", "tennis", "golf",
                  "hockey", "cricket", "rugby", "boxing", "wrestling", "swimming",
                  "running", "marathon", "olympics", "championship", "tournament",
                  "stadium", "athlete", "coach", "referee", "goalkeeper", "pitcher",
                  "quarterback", "striker", "midfielder", "defender"},
        "color": "#4CAF50"
    },
    "tech": {
        "words": {"computer", "software", "hardware", "internet", "website", "email",
                  "digital", "data", "database", "algorithm", "programming", "code",
                  "network", "server", "processor", "memory", "keyboard", "screen",
                  "pixel", "download", "upload", "browser", "google", "microsoft",
                  "apple", "facebook", "twitter", "robot", "artificial", "virtual",
                  "cyber", "bitcoin", "smartphone", "laptop", "tablet"},
        "color": "#00BCD4"
    },
    "religion": {
        "words": {"god", "church", "temple", "mosque", "bible", "prayer", "heaven",
                  "hell", "angel", "devil", "satan", "soul", "spirit", "faith",
                  "christian", "muslim", "jewish", "buddhist", "hindu", "islam",
                  "catholic", "protestant", "orthodox", "holy", "sacred", "divine",
                  "sin", "blessing", "miracle", "prophet", "apostle", "saint",
                  "resurrection", "salvation", "worship"},
        "color": "#CE93D8"
    },
    "clothing": {
        "words": {"shirt", "dress", "shoes", "hat", "coat", "jacket", "pants",
                  "jeans", "boots", "gloves", "scarf", "tie", "suit", "skirt",
                  "sweater", "socks", "uniform", "costume", "silk", "cotton",
                  "leather", "wool", "fabric", "cloth", "ribbon", "belt"},
        "color": "#FFAB91"
    },
}

# Build word->category mapping (first match wins)
word2cat = {}
word2color = {}
for cat, info in categories.items():
    for w in info["words"]:
        if w not in word2cat:
            word2cat[w] = cat
            word2color[w] = info["color"]

# ── Build points data ───────────────────────────────────────────────────────
points = []
for i, w in enumerate(words):
    x, y, z = coords3d[i]
    cat = word2cat.get(w, "other")
    col = word2color.get(w, "#555555")
    word_idx = vocab[w]
    freq = counts[word_idx] if word_idx < len(counts) else 0
    points.append({
        "w": w,
        "x": round(float(x), 5),
        "y": round(float(y), 5),
        "z": round(float(z), 5),
        "c": cat,
        "col": col,
        "f": freq
    })

categorized_count = sum(1 for p in points if p["c"] != "other")
print(f"Categorized: {categorized_count}, Uncategorized: {len(points) - categorized_count}")

# Get step info from checkpoint
step = ckpt.get("step", "?")
print(f"Checkpoint step: {step}")

# ── Build category toggles HTML ─────────────────────────────────────────────
used_cats = sorted(set(p["c"] for p in points if p["c"] != "other"))
toggle_html = ""
for cat in used_cats:
    col = categories[cat]["color"]
    toggle_html += f'    <label><input type="checkbox" class="cat-toggle" data-cat="{cat}" checked><span class="leg-dot" style="background:{col}"></span>{cat}</label>\n'
toggle_html += '    <label><input type="checkbox" class="cat-toggle" data-cat="other" checked><span class="leg-dot" style="background:#555"></span>other</label>\n'

points_json = json.dumps(points, separators=(',', ':'))

# ── Generate HTML ───────────────────────────────────────────────────────────
html = f'''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>V33 — 3D Word Space ({step:,} steps)</title>
<style>
body {{ margin: 0; overflow: hidden; background: #0a0a1a; font-family: -apple-system, sans-serif; }}
#info {{
    position: absolute; top: 10px; left: 10px; color: #ccc; font-size: 13px;
    background: rgba(0,0,0,0.7); padding: 12px 16px; border-radius: 8px;
    max-width: 320px; z-index: 10;
}}
#info h2 {{ margin: 0 0 8px 0; color: #FFD700; font-size: 16px; }}
#info p {{ margin: 4px 0; font-size: 12px; }}
#info .stat {{ color: #4fc3f7; }}
#tooltip {{
    position: absolute; display: none; background: rgba(0,0,0,0.85); color: #fff;
    padding: 8px 12px; border-radius: 6px; font-size: 13px; pointer-events: none; z-index: 20;
}}
#controls {{
    position: absolute; top: 10px; right: 10px; color: #ccc; font-size: 12px;
    background: rgba(0,0,0,0.7); padding: 10px 14px; border-radius: 8px; z-index: 10;
    max-height: 90vh; overflow-y: auto;
}}
#controls label {{ display: block; margin: 3px 0; cursor: pointer; }}
#controls input[type=checkbox] {{ margin-right: 6px; }}
.leg-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; display: inline-block; }}
#search-box {{
    width: 100%; padding: 4px 6px; margin: 6px 0; background: #1a1a2e; border: 1px solid #444;
    color: #fff; border-radius: 4px; font-size: 12px;
}}
</style>
</head><body>
<div id="info">
    <h2>V33 — 3D Word Space</h2>
    <p>Word2vec skip-gram, {step:,} steps</p>
    <p>Top {len(words)} words, <span class="stat">300d → 3d PCA</span>, normalized to unit sphere</p>
    <p>PCA variance explained: <span class="stat">{pca.explained_variance_ratio_.sum():.1%}</span></p>
    <p>{categorized_count} categorized across {len(used_cats)} categories</p>
    <hr style="border-color:#333">
    <p>Drag to rotate | Scroll to zoom | Hover for details</p>
</div>
<div id="tooltip"></div>
<div id="controls">
    <b>Search:</b><br>
    <input type="text" id="search-box" placeholder="Type a word...">
    <br><b>Categories:</b><br>
{toggle_html}    <hr style="border-color:#333">
    <label><input type="checkbox" id="show-labels" checked> Show labels</label>
    <label><input type="checkbox" id="show-axes"> Show axes</label>
    <label><input type="checkbox" id="show-sphere" checked> Show sphere</label>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const points = {points_json};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(1.8, 1.2, 1.8);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Unit sphere wireframe
const sphereGeo = new THREE.SphereGeometry(1, 32, 24);
const sphereMat = new THREE.MeshBasicMaterial({{
    color: 0x333355, wireframe: true, transparent: true, opacity: 0.08
}});
const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
scene.add(sphereMesh);

// Axes
const axesGroup = new THREE.Group();
const axColors = [0xff4444, 0x44ff44, 0x4444ff];
for (let i = 0; i < 3; i++) {{
    const dir = new THREE.Vector3();
    dir.setComponent(i, 1);
    axesGroup.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), 1.3, axColors[i], 0.05, 0.03));
    const neg = new THREE.Vector3();
    neg.setComponent(i, -1);
    axesGroup.add(new THREE.ArrowHelper(neg, new THREE.Vector3(), 1.3, axColors[i], 0.05, 0.03));
}}
axesGroup.visible = false;
scene.add(axesGroup);

// Ambient light for depth perception
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
dirLight.position.set(2, 3, 2);
scene.add(dirLight);

// Create points and labels
const categorized = points.filter(p => p.c !== 'other');
const uncategorized = points.filter(p => p.c === 'other');
const pointGroups = {{}};
const labelSprites = [];
const allMeshes = [];
const meshByWord = {{}};

function createPoints(pts, size) {{
    pts.forEach(p => {{
        const geo = new THREE.SphereGeometry(size, 8, 6);
        const mat = new THREE.MeshPhongMaterial({{ color: new THREE.Color(p.col), emissive: new THREE.Color(p.col), emissiveIntensity: 0.3 }});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(p.x, p.y, p.z);
        mesh.userData = p;
        scene.add(mesh);
        allMeshes.push(mesh);
        meshByWord[p.w] = mesh;
        if (!pointGroups[p.c]) pointGroups[p.c] = [];
        pointGroups[p.c].push(mesh);
    }});
}}

createPoints(categorized, 0.014);
createPoints(uncategorized, 0.004);

function makeLabel(text, position, color) {{
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 64;
    ctx.font = 'bold 22px -apple-system, sans-serif';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 4;
    ctx.fillText(text, 128, 40);
    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false }});
    const sprite = new THREE.Sprite(mat);
    sprite.position.copy(position);
    sprite.position.y += 0.035;
    sprite.scale.set(0.18, 0.045, 1);
    sprite.userData = {{ cat: '' }};
    return sprite;
}}

categorized.forEach(p => {{
    const pos = new THREE.Vector3(p.x, p.y, p.z);
    const sprite = makeLabel(p.w, pos, p.col);
    sprite.userData.cat = p.c;
    scene.add(sprite);
    labelSprites.push(sprite);
}});

// Tooltip
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');

renderer.domElement.addEventListener('mousemove', (e) => {{
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(allMeshes);
    if (hits.length > 0) {{
        const p = hits[0].object.userData;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
        tooltip.innerHTML = '<b>' + p.w + '</b> [' + p.c + ']<br>' +
            'x=' + p.x.toFixed(4) + ' y=' + p.y.toFixed(4) + ' z=' + p.z.toFixed(4);
    }} else {{
        tooltip.style.display = 'none';
    }}
}});

// Search
let highlightedMesh = null;
let highlightOrigColor = null;
document.getElementById('search-box').addEventListener('input', (e) => {{
    const q = e.target.value.toLowerCase().trim();
    // Reset previous highlight
    if (highlightedMesh && highlightOrigColor) {{
        highlightedMesh.material.color.copy(highlightOrigColor);
        highlightedMesh.material.emissive.copy(highlightOrigColor);
        highlightedMesh.scale.set(1, 1, 1);
    }}
    if (q && meshByWord[q]) {{
        const mesh = meshByWord[q];
        highlightOrigColor = mesh.material.color.clone();
        highlightedMesh = mesh;
        mesh.material.color.set(0xffffff);
        mesh.material.emissive.set(0xffffff);
        mesh.scale.set(3, 3, 3);
        // Move camera to look at it
        controls.target.copy(mesh.position);
    }}
}});

// Category toggles
document.querySelectorAll('.cat-toggle').forEach(cb => {{
    cb.addEventListener('change', () => {{
        const cat = cb.dataset.cat;
        const visible = cb.checked;
        if (pointGroups[cat]) pointGroups[cat].forEach(m => m.visible = visible);
        const showLabels = document.getElementById('show-labels').checked;
        labelSprites.filter(s => s.userData.cat === cat).forEach(s => s.visible = visible && showLabels);
    }});
}});

document.getElementById('show-labels').addEventListener('change', (e) => {{
    labelSprites.forEach(s => {{
        const catCb = document.querySelector('.cat-toggle[data-cat="' + s.userData.cat + '"]');
        s.visible = e.target.checked && (!catCb || catCb.checked);
    }});
}});

document.getElementById('show-axes').addEventListener('change', (e) => {{
    axesGroup.visible = e.target.checked;
}});

document.getElementById('show-sphere').addEventListener('change', (e) => {{
    sphereMesh.visible = e.target.checked;
}});

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}}
animate();
</script>
</body></html>'''

with open(OUTPUT, 'w') as f:
    f.write(html)

print(f"\nWrote {OUTPUT} ({len(html):,} bytes)")
print(f"Categories used: {', '.join(used_cats)}")
