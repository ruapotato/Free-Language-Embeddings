#!/usr/bin/env python3
"""
Interactive 3D concept space explorer.

Serves a web UI where you can:
- Fly around the concept space in 3D
- Define custom axes from sentence pairs (e.g. "big cat" - "cat" = size)
- Type any sentence and see where it lands
- Switch between UMAP, PCA, or custom concept directions

Usage:
    python explore_concepts.py                         # latest checkpoint
    python explore_concepts.py --checkpoint step_020000.pt
    python explore_concepts.py --port 8080
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, jsonify, Response

CKPT_DIR = Path("checkpoints/concept_v3")

# ── Sentence corpus ────────────────────────────────────────────────────

GROUPS = {
    "Animals": [
        "the cat sat on the mat", "a cat is sitting on the rug",
        "the dog ran across the yard", "a bird flew over the house",
        "the fish swam in the pond", "the horse galloped through the field",
        "a rabbit hid under the bush", "the snake slithered through the grass",
        "my dog loves to play fetch", "the parrot can talk",
    ],
    "Emotions": [
        "I am very happy today", "she feels joyful and excited",
        "he is sad and lonely", "they were angry about it",
        "I feel scared and anxious", "she was filled with pride",
        "he felt deeply ashamed", "I am so grateful for this",
        "they were bored out of their minds", "she felt a wave of jealousy",
    ],
    "Weather": [
        "it is raining outside", "the rain is pouring down",
        "the sun is shining brightly", "it is a cold winter day",
        "the snow is falling gently", "there is a thunderstorm coming",
        "the wind is blowing hard", "it is a hot summer afternoon",
        "the fog rolled in this morning", "a rainbow appeared after the storm",
    ],
    "Food": [
        "I ate breakfast this morning", "she cooked dinner for us",
        "the pizza was delicious", "he drank a cup of coffee",
        "they had lunch at noon", "the soup was too salty",
        "she baked a chocolate cake", "I ordered a sandwich and fries",
        "the steak was perfectly cooked", "he made fresh orange juice",
    ],
    "Work": [
        "I have a meeting today", "she finished her report",
        "he is working on the project", "they discussed the budget",
        "the deadline is tomorrow", "she got promoted last week",
        "he sent the email to his boss", "the team is behind schedule",
        "I need to update the spreadsheet", "they hired three new employees",
    ],
    "Sports": [
        "he kicked the ball into the goal", "she won the tennis match",
        "the team lost the championship", "he ran a marathon last weekend",
        "she swam fifty laps in the pool", "the basketball game was exciting",
        "he hit a home run", "the referee blew the whistle",
        "she scored the winning point", "they practiced every morning",
    ],
    "Travel": [
        "we flew to paris last summer", "the train arrived on time",
        "she booked a hotel near the beach", "he drove across the country",
        "the flight was delayed by two hours", "we took a boat to the island",
        "the airport was very crowded", "she packed her suitcase the night before",
        "they visited three countries in one trip", "I need to renew my passport",
    ],
    "Health": [
        "she went to the doctor yesterday", "he broke his arm playing football",
        "I have a terrible headache", "the medicine made me feel better",
        "she runs every morning to stay fit", "he needs to get more sleep",
        "the hospital was very busy", "I caught a cold last week",
        "she is allergic to peanuts", "he went for his annual checkup",
    ],
    "Technology": [
        "the computer crashed again", "she wrote a python program",
        "he bought a new smartphone", "the internet is down",
        "they launched a new website", "the robot can walk on two legs",
        "she updated the software", "the printer is out of ink",
        "he built a machine learning model", "the server needs more memory",
    ],
    "Nature": [
        "the mountain was covered in snow", "a river flows through the valley",
        "the forest is full of tall trees", "the ocean waves crashed on the shore",
        "the desert is hot and dry", "flowers bloomed in the garden",
        "the volcano erupted last night", "a waterfall cascaded down the cliff",
        "the lake was perfectly still", "the stars were bright in the sky",
    ],
    "Education": [
        "she studied for the exam all night", "the teacher explained the lesson",
        "he graduated from college last year", "they read the textbook chapter",
        "the students took a quiz", "she wrote an essay about history",
        "he learned to speak french", "the library was quiet",
        "they had a science experiment", "the professor gave a long lecture",
    ],
    "Family": [
        "my mother cooked a big meal", "his father taught him to ride a bike",
        "the baby started walking today", "she visited her grandparents",
        "the kids played in the backyard", "my brother is older than me",
        "she hugged her daughter", "the family gathered for the holiday",
        "he called his sister on the phone", "they adopted a puppy for the children",
    ],
    "Music": [
        "she played the piano beautifully", "he sang a song at the concert",
        "the band performed on stage", "I listened to music all day",
        "she learned to play the guitar", "the drums were very loud",
        "he composed a new symphony", "the crowd cheered after the song",
        "she has a beautiful singing voice", "they danced to the rhythm",
    ],
}

SPECIAL = {
    "Word Order": [
        ("the dog bit the man", "the man bit the dog"),
        ("alice likes bob", "bob likes alice"),
        ("she gave him a book", "he gave her a book"),
        ("the cat chased the mouse", "the mouse chased the cat"),
        ("the teacher praised the student", "the student praised the teacher"),
    ],
    "Paraphrase": [
        ("the king died", "the monarch passed away"),
        ("the massive cat stepped on the rug",
         "there was a rug that a massive cat stepped on"),
        ("he is very smart", "he is quite intelligent"),
        ("she began to cry", "she started crying"),
        ("the car is fast", "the automobile is quick"),
    ],
}

# Predefined concept directions
DEFAULT_DIRECTIONS = [
    ("Size", "a big cat", "a cat"),
    ("Negation", "I am not happy", "I am happy"),
    ("Tense", "I ran", "I run"),
    ("Emotion", "I am happy", "I am sad"),
    ("Formality", "we need to discuss this matter", "we gotta talk about this"),
    ("Plurality", "the cats", "the cat"),
    ("Certainty", "it will definitely rain", "it might rain"),
    ("Complexity", "the intricate mechanism failed", "the thing broke"),
]


# ── Model ──────────────────────────────────────────────────────────────

model = None
tokenizer = None
step = 0


def load_model_global(ckpt_path):
    global model, tokenizer, step
    from concept_model import ConceptAutoencoder, ConceptConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("step", 0)
    print(f"Loaded model at step {step:,}")


def encode(texts):
    enc = tokenizer(texts, max_length=128, padding=True,
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        concepts = model.encode(enc["input_ids"], enc["attention_mask"])
    return concepts.view(concepts.shape[0], -1).numpy()


# ── Projection methods ─────────────────────────────────────────────────

def project_umap_3d(vecs):
    import umap
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15,
                        min_dist=0.25, metric="cosine")
    return reducer.fit_transform(vecs)


def project_pca_3d(vecs):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    return pca.fit_transform(vecs)


def project_custom(vecs, axes):
    """Project vecs onto custom concept direction axes.
    axes: list of (name, positive_text, negative_text)
    Returns projected coords (n_points, len(axes))
    """
    coords = np.zeros((len(vecs), len(axes)))
    for i, (name, pos, neg) in enumerate(axes):
        v_pos = encode([pos])[0]
        v_neg = encode([neg])[0]
        direction = v_pos - v_neg
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        coords[:, i] = vecs @ direction
    return coords


# ── Flask app ──────────────────────────────────────────────────────────

app = Flask(__name__)

# Cache
cached_vecs = None
cached_texts = None
cached_groups = None


def get_corpus_data():
    global cached_vecs, cached_texts, cached_groups
    if cached_vecs is not None:
        return cached_vecs, cached_texts, cached_groups

    texts = []
    groups = []
    for gname, sents in GROUPS.items():
        for s in sents:
            texts.append(s)
            groups.append(gname)

    # Add special pairs
    seen = set(texts)
    for pname, pairs in SPECIAL.items():
        for a, b in pairs:
            if a not in seen:
                texts.append(a)
                groups.append(f"_{pname}")
                seen.add(a)
            if b not in seen:
                texts.append(b)
                groups.append(f"_{pname}")
                seen.add(b)

    print(f"Encoding {len(texts)} corpus sentences...")
    cached_vecs = encode(texts)
    cached_texts = texts
    cached_groups = groups
    return cached_vecs, cached_texts, cached_groups


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/corpus")
def api_corpus():
    """Return the full corpus with 3D coordinates."""
    method = request.args.get("method", "umap")
    ax1 = request.args.get("ax1", "")
    ax2 = request.args.get("ax2", "")
    ax3 = request.args.get("ax3", "")

    vecs, texts, groups = get_corpus_data()

    if method == "pca":
        coords = project_pca_3d(vecs)
    elif method == "custom" and ax1 and ax2 and ax3:
        axes = []
        for ax_str in [ax1, ax2, ax3]:
            parts = ax_str.split("|||")
            if len(parts) == 3:
                axes.append((parts[0], parts[1], parts[2]))
        if len(axes) == 3:
            coords = project_custom(vecs, axes)
        else:
            coords = project_umap_3d(vecs)
    else:
        coords = project_umap_3d(vecs)

    # Build special pair links
    links = []
    text_idx = {t: i for i, t in enumerate(texts)}
    for pname, pairs in SPECIAL.items():
        for a, b in pairs:
            if a in text_idx and b in text_idx:
                ia, ib = text_idx[a], text_idx[b]
                va, vb = vecs[ia], vecs[ib]
                cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
                links.append({
                    "a": ia, "b": ib, "type": pname, "cos": round(cos, 3),
                })

    points = []
    for i in range(len(texts)):
        points.append({
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]),
            "text": texts[i],
            "group": groups[i],
        })

    return jsonify({"points": points, "links": links, "step": step,
                     "method": method})


@app.route("/api/encode", methods=["POST"])
def api_encode():
    """Encode a custom sentence and return its position."""
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    vec = encode([text])[0]
    vecs, texts, groups = get_corpus_data()

    # Find nearest neighbors
    sims = vecs @ vec / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(vec) + 1e-8)
    top_idx = np.argsort(-sims)[:5]
    neighbors = [{"text": texts[i], "group": groups[i],
                  "sim": round(float(sims[i]), 3)} for i in top_idx]

    return jsonify({"vec": vec.tolist(), "neighbors": neighbors})


@app.route("/api/similarity", methods=["POST"])
def api_similarity():
    """Compute cosine similarity between two sentences."""
    data = request.get_json()
    a = data.get("a", "").strip()
    b = data.get("b", "").strip()
    if not a or not b:
        return jsonify({"error": "need two sentences"}), 400

    vecs = encode([a, b])
    cos = float(np.dot(vecs[0], vecs[1]) /
                (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-8))
    return jsonify({"a": a, "b": b, "similarity": round(cos, 4)})


@app.route("/api/directions")
def api_directions():
    return jsonify(DEFAULT_DIRECTIONS)


# ── HTML/JS ────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Concept Space Explorer</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, sans-serif; }
#container { display: flex; height: 100vh; }
#sidebar { width: 340px; padding: 16px; overflow-y: auto; border-right: 1px solid #30363d;
           display: flex; flex-direction: column; gap: 12px; }
#plot { flex: 1; }
h2 { font-size: 14px; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }
label { font-size: 12px; color: #8b949e; }
input, select, textarea { width: 100%; background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
       border-radius: 4px; padding: 6px 8px; font-size: 13px; font-family: inherit; }
input:focus, select:focus, textarea:focus { outline: none; border-color: #58a6ff; }
button { background: #238636; color: white; border: none; border-radius: 4px; padding: 6px 12px;
         cursor: pointer; font-size: 13px; }
button:hover { background: #2ea043; }
button.secondary { background: #30363d; }
button.secondary:hover { background: #484f58; }
.section { display: flex; flex-direction: column; gap: 6px; }
.row { display: flex; gap: 6px; align-items: center; }
.neighbors { font-size: 11px; color: #8b949e; }
.neighbors div { padding: 2px 0; border-bottom: 1px solid #21262d; }
.neighbors .sim { color: #58a6ff; font-weight: bold; }
.dir-row { display: flex; gap: 4px; align-items: center; font-size: 11px; }
.dir-row input { font-size: 11px; padding: 3px 5px; }
.dir-row span { white-space: nowrap; }
#status { font-size: 11px; color: #8b949e; padding: 4px; }
.pair-info { font-size: 11px; margin-top: 4px; padding: 6px; background: #161b22;
             border-radius: 4px; border: 1px solid #30363d; }
.pair-info .wo { color: #ff6b6b; }
.pair-info .para { color: #69db7c; }
.compare-section textarea { height: 50px; resize: vertical; }
</style>
</head>
<body>
<div id="container">
<div id="sidebar">
    <div style="font-size:16px;font-weight:bold;color:#58a6ff;">
        Concept Space Explorer
        <span id="step-label" style="font-size:11px;color:#8b949e;"></span>
    </div>

    <div class="section">
        <h2>Projection</h2>
        <select id="method" onchange="reload()">
            <option value="umap">UMAP 3D</option>
            <option value="pca">PCA 3D</option>
            <option value="custom">Custom Axes</option>
        </select>
        <div id="custom-axes" style="display:none;">
            <div id="axis-editors"></div>
            <button onclick="applyCustomAxes()" style="margin-top:4px;">Apply Axes</button>
            <div style="margin-top:4px;">
                <label>Presets:</label>
                <select id="dir-presets" onchange="loadPreset()">
                    <option value="">-- pick 3 --</option>
                </select>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Search / Encode</h2>
        <input id="search-input" type="text" placeholder="Type a sentence and press Enter..."
               onkeydown="if(event.key==='Enter')doSearch()">
        <button onclick="doSearch()">Encode &amp; Locate</button>
        <div id="search-results" class="neighbors"></div>
    </div>

    <div class="section compare-section">
        <h2>Compare Two Sentences</h2>
        <textarea id="compare-a" placeholder="Sentence A..."></textarea>
        <textarea id="compare-b" placeholder="Sentence B..."></textarea>
        <button onclick="doCompare()">Compare</button>
        <div id="compare-result" style="font-size:12px;margin-top:4px;"></div>
    </div>

    <div class="section">
        <h2>Special Pairs</h2>
        <div id="pair-info" class="pair-info"></div>
    </div>

    <div id="status">Loading...</div>
</div>
<div id="plot"></div>
</div>

<script>
const GROUP_COLORS = {
    Animals: '#2ecc71', Emotions: '#e74c3c', Weather: '#3498db',
    Food: '#f39c12', Work: '#9b59b6', Sports: '#1abc9c',
    Travel: '#e67e22', Health: '#e91e63', Technology: '#00bcd4',
    Nature: '#8bc34a', Education: '#ff9800', Family: '#ff5722',
    Music: '#673ab7',
    '_Word Order': '#ff1744', '_Paraphrase': '#00e676',
};

let corpusData = null;
let customPoints = [];
let directions = [];

async function reload() {
    const method = document.getElementById('method').value;
    document.getElementById('custom-axes').style.display = method === 'custom' ? 'block' : 'none';
    if (method === 'custom') return; // user clicks Apply
    document.getElementById('status').textContent = 'Loading ' + method + '...';

    const resp = await fetch('/api/corpus?method=' + method);
    corpusData = await resp.json();
    document.getElementById('step-label').textContent = '(step ' + corpusData.step.toLocaleString() + ')';
    customPoints = [];
    plotAll();
    showPairInfo();
    document.getElementById('status').textContent = corpusData.points.length + ' points loaded (' + method + ')';
}

async function applyCustomAxes() {
    const editors = document.querySelectorAll('.axis-editor');
    const params = new URLSearchParams({method: 'custom'});
    ['ax1','ax2','ax3'].forEach((key, i) => {
        const ed = editors[i];
        if (!ed) return;
        const name = ed.querySelector('.ax-name').value;
        const pos = ed.querySelector('.ax-pos').value;
        const neg = ed.querySelector('.ax-neg').value;
        params.set(key, name + '|||' + pos + '|||' + neg);
    });
    document.getElementById('status').textContent = 'Computing custom projection...';
    const resp = await fetch('/api/corpus?' + params);
    corpusData = await resp.json();
    customPoints = [];
    plotAll();
    showPairInfo();
    document.getElementById('status').textContent = 'Custom axes applied';
}

function buildAxisEditors() {
    const container = document.getElementById('axis-editors');
    container.innerHTML = '';
    const defaults = [
        ['Size', 'a big cat', 'a cat'],
        ['Emotion', 'I am happy', 'I am sad'],
        ['Tense', 'I ran', 'I run'],
    ];
    for (let i = 0; i < 3; i++) {
        const d = defaults[i];
        const div = document.createElement('div');
        div.className = 'dir-row axis-editor';
        div.style.marginBottom = '6px';
        div.innerHTML = `
            <span>Axis ${i+1}:</span>
            <input class="ax-name" value="${d[0]}" style="width:55px" placeholder="Name">
            <input class="ax-pos" value="${d[1]}" style="width:90px" placeholder="Positive">
            <span>-</span>
            <input class="ax-neg" value="${d[2]}" style="width:90px" placeholder="Negative">
        `;
        container.appendChild(div);
    }
}

async function loadDirections() {
    const resp = await fetch('/api/directions');
    directions = await resp.json();
    const sel = document.getElementById('dir-presets');
    // Build combos of 3
    for (let i = 0; i < directions.length; i++) {
        for (let j = i+1; j < directions.length; j++) {
            for (let k = j+1; k < directions.length; k++) {
                const opt = document.createElement('option');
                opt.value = [i,j,k].join(',');
                opt.textContent = directions[i][0] + ' / ' + directions[j][0] + ' / ' + directions[k][0];
                sel.appendChild(opt);
            }
        }
    }
}

function loadPreset() {
    const val = document.getElementById('dir-presets').value;
    if (!val) return;
    const [i,j,k] = val.split(',').map(Number);
    const eds = document.querySelectorAll('.axis-editor');
    [directions[i], directions[j], directions[k]].forEach((d, idx) => {
        eds[idx].querySelector('.ax-name').value = d[0];
        eds[idx].querySelector('.ax-pos').value = d[1];
        eds[idx].querySelector('.ax-neg').value = d[2];
    });
}

function plotAll() {
    if (!corpusData) return;
    const traces = [];
    const grouped = {};

    corpusData.points.forEach((p, i) => {
        if (!grouped[p.group]) grouped[p.group] = {x:[], y:[], z:[], text:[], idx:[]};
        const g = grouped[p.group];
        g.x.push(p.x); g.y.push(p.y); g.z.push(p.z);
        g.text.push(p.text); g.idx.push(i);
    });

    for (const [gname, g] of Object.entries(grouped)) {
        traces.push({
            x: g.x, y: g.y, z: g.z,
            text: g.text,
            type: 'scatter3d', mode: 'markers',
            name: gname.replace(/^_/, ''),
            legendgroup: gname,
            marker: {
                size: gname.startsWith('_') ? 5 : 3.5,
                color: GROUP_COLORS[gname] || '#888',
                opacity: 0.85,
                symbol: gname.startsWith('_') ? 'diamond' : 'circle',
            },
            hoverinfo: 'text',
            hovertext: g.text.map((t, j) => `<b>${gname.replace(/^_/,'')}</b><br>${t}`),
        });
    }

    // Links — grouped with their point traces so legend toggles both
    for (const link of corpusData.links) {
        const a = corpusData.points[link.a];
        const b = corpusData.points[link.b];
        const color = link.type === 'Word Order' ? '#ff1744' : '#00e676';
        const lgGroup = '_' + link.type;
        traces.push({
            x: [a.x, b.x], y: [a.y, b.y], z: [a.z, b.z],
            type: 'scatter3d', mode: 'lines',
            line: { color: color, width: 2.5 },
            legendgroup: lgGroup,
            showlegend: false, hoverinfo: 'none',
        });
    }

    // Custom search points
    for (const cp of customPoints) {
        traces.push({
            x: [cp.x], y: [cp.y], z: [cp.z],
            text: [cp.text],
            type: 'scatter3d', mode: 'markers+text',
            name: 'Search: ' + cp.text.substring(0, 20),
            marker: { size: 8, color: '#ffffff', symbol: 'x', opacity: 1,
                      line: { color: '#ff0', width: 1 } },
            textposition: 'top center',
            textfont: { size: 10, color: '#ffffff' },
            hovertext: ['<b>SEARCH</b><br>' + cp.text],
            hoverinfo: 'text',
        });
    }

    const axLabels = corpusData.method === 'custom' ? {} : {
        xaxis: {title: ''}, yaxis: {title: ''}, zaxis: {title: ''}
    };

    Plotly.newPlot('plot', traces, {
        scene: {
            bgcolor: '#0d1117',
            xaxis: { gridcolor: '#1a1e24', zerolinecolor: '#30363d', color: '#484f58',
                     title: { text: '' } },
            yaxis: { gridcolor: '#1a1e24', zerolinecolor: '#30363d', color: '#484f58',
                     title: { text: '' } },
            zaxis: { gridcolor: '#1a1e24', zerolinecolor: '#30363d', color: '#484f58',
                     title: { text: '' } },
        },
        paper_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        legend: { bgcolor: '#161b22', bordercolor: '#30363d', font: { size: 10 } },
        margin: { l: 0, r: 0, t: 0, b: 0 },
    }, { responsive: true });
}

async function doSearch() {
    const text = document.getElementById('search-input').value.trim();
    if (!text) return;
    document.getElementById('status').textContent = 'Encoding "' + text.substring(0, 40) + '"...';

    const resp = await fetch('/api/encode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text}),
    });
    const data = await resp.json();

    // Show neighbors
    const div = document.getElementById('search-results');
    div.innerHTML = '<b>Nearest neighbors:</b>';
    for (const n of data.neighbors) {
        const d = document.createElement('div');
        d.innerHTML = `<span class="sim">${n.sim}</span> [${n.group}] ${n.text}`;
        div.appendChild(d);
    }

    // Find approximate position: weighted average of top neighbors in current projection
    if (corpusData) {
        const texts = corpusData.points.map(p => p.text);
        let sx=0, sy=0, sz=0, sw=0;
        for (const n of data.neighbors) {
            const idx = texts.indexOf(n.text);
            if (idx >= 0) {
                const w = Math.max(0, n.sim);
                sx += corpusData.points[idx].x * w;
                sy += corpusData.points[idx].y * w;
                sz += corpusData.points[idx].z * w;
                sw += w;
            }
        }
        if (sw > 0) {
            customPoints.push({x: sx/sw, y: sy/sw, z: sz/sw, text: text});
            plotAll();
        }
    }
    document.getElementById('status').textContent = 'Encoded. ' + data.neighbors.length + ' neighbors found.';
}

async function doCompare() {
    const a = document.getElementById('compare-a').value.trim();
    const b = document.getElementById('compare-b').value.trim();
    if (!a || !b) return;

    const resp = await fetch('/api/similarity', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({a, b}),
    });
    const data = await resp.json();
    const el = document.getElementById('compare-result');
    const sim = data.similarity;
    const color = sim > 0.7 ? '#69db7c' : sim > 0.3 ? '#ffd43b' : '#ff6b6b';
    el.innerHTML = `Cosine similarity: <span style="color:${color};font-weight:bold;font-size:16px">${sim}</span>`;
}

function showPairInfo() {
    if (!corpusData) return;
    const div = document.getElementById('pair-info');
    let html = '';
    for (const link of corpusData.links) {
        const a = corpusData.points[link.a];
        const b = corpusData.points[link.b];
        const cls = link.type === 'Word Order' ? 'wo' : 'para';
        html += `<div class="${cls}"><b>${link.cos}</b> ${link.type}: "${a.text.substring(0,30)}" / "${b.text.substring(0,30)}"</div>`;
    }
    div.innerHTML = html;
}

// Init
buildAxisEditors();
loadDirections();
reload();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Interactive concept space explorer")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = CKPT_DIR / args.checkpoint
    else:
        ckpt_path = CKPT_DIR / "latest.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    load_model_global(ckpt_path)

    # Pre-encode corpus
    get_corpus_data()

    print(f"\nStarting explorer at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
