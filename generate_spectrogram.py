#!/usr/bin/env python3
"""Generate embedding spectrogram visualization (heatmap-based).

Produces a self-contained HTML file at docs/spectrogram.html with:
1. Semantic Sweep Heatmaps (dimension activations sorted by correlation)
2. Cosine Similarity Matrices
3. PCA Loop Plots
4. Dimension Activation Fingerprints
"""

import json
import os
import numpy as np
import torch

# ── Configuration ──────────────────────────────────────────────────────────

OUTPUT_PATH = "docs/spectrogram.html"

MODELS = {
    "V28": {
        "checkpoint": "checkpoints/word2vec_v28/latest.pt",
        "color": "#ff6b6b",
        "desc": "Standard Skip-Gram",
    },
    "V33": {
        "checkpoint": "checkpoints/word2vec_v33/latest.pt",
        "color": "#4ecdc4",
        "desc": "Mixed SG+CBOW",
    },
    "V34": {
        "checkpoint": "checkpoints/word2vec_v34/latest.pt",
        "color": "#FFD700",
        "desc": "Dynamic Masking",
    },
    "Google": {
        "data": "data/google_w2v_top100k.npz",
        "color": "#bb86fc",
        "desc": "Google Word2Vec",
    },
}

VOCAB_PATH = "checkpoints/word2vec_v28/vocab.json"

# Semantic axes — word lists in meaningful order
SEMANTIC_AXES = {
    "small → big": ["tiny", "small", "little", "medium", "large", "big", "huge", "enormous", "giant", "massive"],
    "cold → hot": ["freezing", "cold", "cool", "warm", "hot", "burning", "boiling"],
    "numbers 1-10": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "months": ["january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november", "december"],
    "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "weak → strong": ["weak", "feeble", "fragile", "moderate", "firm", "strong", "powerful", "mighty"],
    "slow → fast": ["slow", "sluggish", "steady", "quick", "fast", "rapid", "swift"],
    "near → far": ["near", "close", "nearby", "distant", "far", "remote"],
}

# Google has capitalized month/day names
GOOGLE_CASE_MAP = {}
for m in ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]:
    GOOGLE_CASE_MAP[m] = m.capitalize()
for d in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
    GOOGLE_CASE_MAP[d] = d.capitalize()

# Categories for dimension fingerprints
CATEGORIES = {
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "colors": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"],
    "animals": ["dog", "cat", "horse", "bird", "fish", "bear", "lion", "tiger", "wolf", "deer"],
    "countries": ["france", "germany", "japan", "china", "india", "brazil", "canada", "russia", "spain", "italy"],
    "emotions": ["happy", "sad", "angry", "afraid", "surprised", "disgusted", "proud", "ashamed", "jealous", "guilty"],
    "body": ["head", "hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg", "heart"],
    "professions": ["doctor", "teacher", "lawyer", "engineer", "artist", "soldier", "farmer", "pilot", "nurse", "chef"],
}

GOOGLE_CATEGORY_MAP = {}
for cat, words in CATEGORIES.items():
    for w in words:
        if w in GOOGLE_CASE_MAP:
            GOOGLE_CATEGORY_MAP[w] = GOOGLE_CASE_MAP[w]
# Also map country names for Google
for w in ["france", "germany", "japan", "china", "india", "brazil", "canada", "russia", "spain", "italy"]:
    GOOGLE_CATEGORY_MAP[w] = w.capitalize()


# ── Load Models ────────────────────────────────────────────────────────────

def load_models():
    """Load all model embeddings and build word→vector lookup."""
    print("Loading vocabulary...")
    with open(VOCAB_PATH) as f:
        vocab_data = json.load(f)
    custom_w2i = vocab_data["word2id"]

    print("Loading Google word2vec...")
    gdata = np.load("data/google_w2v_top100k.npz", allow_pickle=True)
    google_words = list(gdata["words"])
    google_vecs = gdata["vecs"]
    google_w2i = {w: i for i, w in enumerate(google_words)}

    models = {}
    for name, cfg in MODELS.items():
        if name == "Google":
            models[name] = {"w2i": google_w2i, "vecs": google_vecs}
            print(f"  {name}: {len(google_w2i):,} words, {google_vecs.shape[1]}d")
        else:
            print(f"  Loading {name} from {cfg['checkpoint']}...")
            cp = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
            vecs = cp["model_state_dict"]["target_embeddings.weight"].numpy()
            models[name] = {"w2i": custom_w2i, "vecs": vecs}
            print(f"  {name}: {len(custom_w2i):,} words, {vecs.shape[1]}d")

    return models


def get_vec(models, model_name, word):
    """Get embedding vector for a word, handling Google casing."""
    m = models[model_name]
    w = word
    if model_name == "Google":
        # Try as-is first, then capitalized
        if w not in m["w2i"]:
            w = GOOGLE_CASE_MAP.get(word, word)
        if w not in m["w2i"]:
            w = word.capitalize()
        if w not in m["w2i"]:
            return None
    else:
        if w not in m["w2i"]:
            return None
    return m["vecs"][m["w2i"][w]]


def filter_words(models, words):
    """Return only words present in ALL models."""
    return [w for w in words if all(get_vec(models, mn, w) is not None for mn in MODELS)]


# ── Compute Data ───────────────────────────────────────────────────────────

def compute_semantic_sweep(models):
    """Compute heatmap data for semantic sweep visualization."""
    print("\nComputing semantic sweeps...")
    result = {}
    for axis_name, words in SEMANTIC_AXES.items():
        valid_words = filter_words(models, words)
        if len(valid_words) < 3:
            print(f"  Skipping '{axis_name}': only {len(valid_words)} words in common")
            continue
        print(f"  {axis_name}: {len(valid_words)} words")

        axis_data = {"words": valid_words, "models": {}}
        for mn in MODELS:
            # Get embedding matrix for these words
            vecs = np.array([get_vec(models, mn, w) for w in valid_words])  # [n_words, 300]

            # Compute correlation of each dimension with word index
            indices = np.arange(len(valid_words), dtype=np.float64)
            correlations = np.array([
                np.corrcoef(indices, vecs[:, d])[0, 1] if np.std(vecs[:, d]) > 1e-10 else 0.0
                for d in range(vecs.shape[1])
            ])
            # Sort dimensions by correlation (most positive at top)
            sort_order = np.argsort(-correlations)

            # Build sorted heatmap data
            sorted_vecs = vecs[:, sort_order].T  # [300, n_words]

            # Normalize for display: clip to [-3std, +3std] then scale to [-1, 1]
            std = np.std(sorted_vecs)
            if std > 1e-10:
                clip_val = 3 * std
                normed = np.clip(sorted_vecs, -clip_val, clip_val) / clip_val
            else:
                normed = sorted_vecs

            axis_data["models"][mn] = {
                "heatmap": normed.tolist(),
                "correlations": correlations[sort_order].tolist(),
            }

        result[axis_name] = axis_data
    return result


def compute_cosine_matrices(models):
    """Compute cosine similarity matrices for ordered sequences."""
    print("\nComputing cosine similarity matrices...")
    sequences = {
        "numbers 1-10": SEMANTIC_AXES["numbers 1-10"],
        "months": SEMANTIC_AXES["months"],
        "days": SEMANTIC_AXES["days"],
    }
    result = {}
    for seq_name, words in sequences.items():
        valid_words = filter_words(models, words)
        if len(valid_words) < 3:
            continue
        print(f"  {seq_name}: {len(valid_words)} words")

        seq_data = {"words": valid_words, "models": {}}
        for mn in MODELS:
            vecs = np.array([get_vec(models, mn, w) for w in valid_words])
            # Normalize
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            vecs_norm = vecs / norms
            # Cosine similarity matrix
            sim = (vecs_norm @ vecs_norm.T).tolist()
            seq_data["models"][mn] = sim

        result[seq_name] = seq_data
    return result


def compute_pca_loops(models):
    """Compute PCA projections for ordered sequences."""
    print("\nComputing PCA loops...")
    sequences = {
        "numbers 1-10": SEMANTIC_AXES["numbers 1-10"],
        "months": SEMANTIC_AXES["months"],
        "days": SEMANTIC_AXES["days"],
    }
    result = {}
    for seq_name, words in sequences.items():
        valid_words = filter_words(models, words)
        if len(valid_words) < 3:
            continue
        print(f"  {seq_name}: {len(valid_words)} words")

        seq_data = {"words": valid_words, "models": {}}
        for mn in MODELS:
            vecs = np.array([get_vec(models, mn, w) for w in valid_words])
            # Center
            centered = vecs - vecs.mean(axis=0)
            # SVD for top 2 components
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[:2].T  # [n_words, 2]
            # Normalize to [-1, 1] range
            max_abs = np.abs(proj).max()
            if max_abs > 1e-10:
                proj = proj / max_abs
            seq_data["models"][mn] = proj.tolist()

        result[seq_name] = seq_data
    return result


def compute_pca_waves(models):
    """Compute top PCA components as wave curves for ordered sequences.

    Like the Welch Labs grokking video: as you sweep along an ordered axis,
    each PCA component traces out a clean sine-like wave.
    """
    print("\nComputing PCA wave curves...")
    N_COMPONENTS = 6  # top 6 PCA components

    # Use all semantic axes, not just periodic ones
    result = {}
    for axis_name, words in SEMANTIC_AXES.items():
        valid_words = filter_words(models, words)
        if len(valid_words) < 4:
            continue
        print(f"  {axis_name}: {len(valid_words)} words")

        axis_data = {"words": valid_words, "models": {}}
        for mn in MODELS:
            vecs = np.array([get_vec(models, mn, w) for w in valid_words])
            centered = vecs - vecs.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            n_comp = min(N_COMPONENTS, len(valid_words) - 1)
            proj = centered @ Vt[:n_comp].T  # [n_words, n_comp]

            # Normalize each component to [-1, 1]
            for c in range(n_comp):
                max_abs = np.abs(proj[:, c]).max()
                if max_abs > 1e-10:
                    proj[:, c] /= max_abs

            # Also compute variance explained per component
            total_var = (S ** 2).sum()
            var_explained = [(S[i] ** 2 / total_var * 100) for i in range(n_comp)]

            axis_data["models"][mn] = {
                "curves": proj.tolist(),  # [n_words, n_comp]
                "var_explained": var_explained,
            }

        result[axis_name] = axis_data
    return result


def compute_pca_surfaces(models):
    """Compute 2D PCA activation surfaces for periodic sequences.

    For each periodic sequence (months, days, numbers), compute the full
    embedding matrix projected onto top PCA components, then interpolate
    to create a smooth surface. The x-axis is position in sequence, y-axis
    is PCA component index, color is activation value.
    """
    print("\nComputing PCA surfaces...")
    sequences = {
        "numbers 1-10": SEMANTIC_AXES["numbers 1-10"],
        "months": SEMANTIC_AXES["months"],
        "days": SEMANTIC_AXES["days"],
    }

    N_COMPONENTS = 20  # more components for surface
    N_INTERP = 100  # interpolation points

    result = {}
    for seq_name, words in sequences.items():
        valid_words = filter_words(models, words)
        if len(valid_words) < 4:
            continue
        print(f"  {seq_name}: {len(valid_words)} words")

        seq_data = {"words": valid_words, "models": {}}
        for mn in MODELS:
            vecs = np.array([get_vec(models, mn, w) for w in valid_words])
            centered = vecs - vecs.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            n_comp = min(N_COMPONENTS, len(valid_words) - 1)
            proj = centered @ Vt[:n_comp].T  # [n_words, n_comp]

            # Interpolate along word axis for smooth surface
            x_orig = np.linspace(0, 1, len(valid_words))
            x_interp = np.linspace(0, 1, N_INTERP)
            surface = np.zeros((n_comp, N_INTERP))
            for c in range(n_comp):
                surface[c] = np.interp(x_interp, x_orig, proj[:, c])

            # Normalize to [-1, 1]
            max_abs = np.abs(surface).max()
            if max_abs > 1e-10:
                surface /= max_abs

            seq_data["models"][mn] = surface.tolist()

        result[seq_name] = seq_data
    return result


def compute_fingerprints(models):
    """Compute mean absolute activation per dimension per category."""
    print("\nComputing dimension fingerprints...")
    result = {}
    cat_names = list(CATEGORIES.keys())

    for mn in MODELS:
        print(f"  {mn}...")
        heatmap = []
        valid_cats = []
        for cat in cat_names:
            words = CATEGORIES[cat]
            valid_words = [w for w in words if get_vec(models, mn, w) is not None]
            if len(valid_words) < 3:
                continue
            vecs = np.array([get_vec(models, mn, w) for w in valid_words])
            mean_abs = np.mean(np.abs(vecs), axis=0)  # [300]
            heatmap.append(mean_abs.tolist())
            if mn == list(MODELS.keys())[0]:
                valid_cats.append(cat)

        # Normalize each row to [0, 1] for display
        heatmap = np.array(heatmap)
        for i in range(heatmap.shape[0]):
            row_max = heatmap[i].max()
            if row_max > 1e-10:
                heatmap[i] /= row_max

        result[mn] = heatmap.tolist()

    # Use categories valid for first model (they should be same for all)
    valid_cats = []
    for cat in cat_names:
        words = CATEGORIES[cat]
        if all(all(get_vec(models, mn, w) is not None for w in words[:3]) for mn in MODELS):
            valid_cats.append(cat)
        elif any(
            sum(1 for w in words if get_vec(models, mn, w) is not None) >= 3
            for mn in MODELS
        ):
            valid_cats.append(cat)

    return {"categories": cat_names, "models": result}


# ── Generate HTML ──────────────────────────────────────────────────────────

def generate_html(sweep_data, cosine_data, pca_data, pca_wave_data, pca_surface_data, fingerprint_data):
    """Generate self-contained HTML with embedded data and canvas renderers."""
    # Serialize data
    data_json = json.dumps({
        "sweeps": sweep_data,
        "cosine": cosine_data,
        "pca": pca_data,
        "pcaWaves": pca_wave_data,
        "pcaSurfaces": pca_surface_data,
        "fingerprints": fingerprint_data,
        "modelNames": list(MODELS.keys()),
        "modelColors": {k: v["color"] for k, v in MODELS.items()},
        "modelDescs": {k: v["desc"] for k, v in MODELS.items()},
    }, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Embedding Spectrogram</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a1a;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    padding: 20px;
}}
h1 {{
    text-align: center;
    font-size: 28px;
    margin: 20px 0 5px;
    color: #fff;
    letter-spacing: 2px;
}}
h2 {{
    font-size: 20px;
    margin: 40px 0 10px;
    color: #888;
    border-bottom: 1px solid #222;
    padding-bottom: 8px;
}}
h3 {{
    font-size: 14px;
    color: #666;
    margin: 15px 0 5px;
    text-align: center;
}}
.subtitle {{
    text-align: center;
    color: #555;
    font-size: 13px;
    margin-bottom: 30px;
}}
.section {{ margin-bottom: 50px; }}
.row {{
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 10px 0;
}}
.panel {{
    text-align: center;
}}
.panel-label {{
    font-size: 12px;
    margin-bottom: 4px;
    font-weight: bold;
}}
canvas {{
    image-rendering: pixelated;
    border: 1px solid #1a1a2e;
}}
select {{
    background: #1a1a2e;
    color: #e0e0e0;
    border: 1px solid #333;
    padding: 6px 12px;
    font-family: inherit;
    font-size: 13px;
    border-radius: 4px;
    margin: 8px;
    cursor: pointer;
}}
select:hover {{ border-color: #555; }}
.controls {{
    text-align: center;
    margin: 10px 0;
}}
.legend {{
    display: flex;
    gap: 20px;
    justify-content: center;
    margin: 10px 0;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
.colorbar {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin: 5px 0;
    font-size: 11px;
    color: #666;
}}
.colorbar canvas {{
    border: none;
    image-rendering: auto;
}}
.axis-label {{
    font-size: 10px;
    color: #555;
    text-align: center;
    margin-top: 2px;
}}
</style>
</head>
<body>

<h1>EMBEDDING SPECTROGRAM</h1>
<p class="subtitle">Heatmap analysis of 300-dimensional word embeddings across 4 models</p>

<div class="legend">
""" + "".join(
    f'<div class="legend-item"><span class="legend-dot" style="background:{v["color"]}"></span>{k}: {v["desc"]}</div>'
    for k, v in MODELS.items()
) + """
</div>

<!-- Section 1: Semantic Sweeps -->
<div class="section" id="sweep-section">
    <h2>1. Semantic Sweep Heatmaps</h2>
    <p class="axis-label">Dimensions sorted by correlation with word order. Top = most positively correlated.</p>
    <div class="controls">
        <select id="sweep-select" onchange="renderSweep()"></select>
    </div>
    <div class="row" id="sweep-row"></div>
    <div class="colorbar">
        <span>-1</span>
        <canvas id="sweep-colorbar" width="200" height="12"></canvas>
        <span>+1</span>
    </div>
</div>

<!-- Section 2: Cosine Similarity -->
<div class="section" id="cosine-section">
    <h2>2. Cosine Similarity Matrices</h2>
    <p class="axis-label">Similarity between all word pairs. Smooth gradients indicate learned order.</p>
    <div class="controls">
        <select id="cosine-select" onchange="renderCosine()"></select>
    </div>
    <div class="row" id="cosine-row"></div>
    <div class="colorbar">
        <span>-1</span>
        <canvas id="cosine-colorbar" width="200" height="12"></canvas>
        <span>+1</span>
    </div>
</div>

<!-- Section 3: PCA Loops -->
<div class="section" id="pca-section">
    <h2>3. PCA Loop Plots</h2>
    <p class="axis-label">Top 2 PCA components. Loops indicate learned circular/periodic structure.</p>
    <div class="controls">
        <select id="pca-select" onchange="renderPCA()"></select>
    </div>
    <div class="row" id="pca-row"></div>
</div>

<!-- Section 4: PCA Sine Waves -->
<div class="section" id="wave-section">
    <h2>4. PCA Component Waves</h2>
    <p class="axis-label">Top 6 PCA components plotted as you sweep along a semantic axis. Clean sine waves = learned periodic structure.</p>
    <div class="controls">
        <select id="wave-select" onchange="renderWaves()"></select>
    </div>
    <div class="row" id="wave-row"></div>
</div>

<!-- Section 5: PCA Activation Surfaces -->
<div class="section" id="surface-section">
    <h2>5. PCA Activation Surface</h2>
    <p class="axis-label">All PCA components as a heatmap surface. X = position in sequence (interpolated), Y = PCA component. Wave patterns visible as colored bands.</p>
    <div class="controls">
        <select id="surface-select" onchange="renderSurfaces()"></select>
    </div>
    <div class="row" id="surface-row"></div>
    <div class="colorbar">
        <span>-1</span>
        <canvas id="surface-colorbar" width="200" height="12"></canvas>
        <span>+1</span>
    </div>
</div>

<!-- Section 6: Fingerprints -->
<div class="section" id="fp-section">
    <h2>4. Dimension Activation Fingerprints</h2>
    <p class="axis-label">Mean |activation| per dimension, per word category. Normalized per row.</p>
    <div class="row" id="fp-row"></div>
    <div class="colorbar">
        <span>0</span>
        <canvas id="fp-colorbar" width="200" height="12"></canvas>
        <span>max</span>
    </div>
</div>

<script>
const DATA = """ + data_json + """;

// ── Color helpers ──
function divergingColor(v) {
    // v in [-1, 1] → blue-black-red
    const r = v > 0 ? Math.round(v * 220 + 35) : 35;
    const g = 20;
    const b = v < 0 ? Math.round(-v * 220 + 35) : 35;
    return [r, g, b];
}

function sequentialColor(v) {
    // v in [0, 1] → black to bright yellow-white
    const r = Math.round(v * 255);
    const g = Math.round(v * 200);
    const b = Math.round(v * 60);
    return [r, g, b];
}

function cosineColor(v) {
    // v in [-1, 1] → blue-black-red but shifted for similarity
    // Most values will be positive, so use a perceptual scale
    if (v > 0) {
        const t = v;
        return [Math.round(255 * t), Math.round(100 * t), Math.round(30 * t)];
    } else {
        const t = -v;
        return [Math.round(30 * t), Math.round(100 * t), Math.round(255 * t)];
    }
}

// ── Draw colorbar ──
function drawColorbar(canvasId, type) {
    const c = document.getElementById(canvasId);
    if (!c) return;
    const ctx = c.getContext('2d');
    for (let x = 0; x < c.width; x++) {
        const t = (x / (c.width - 1)) * 2 - 1; // -1 to 1
        let rgb;
        if (type === 'diverging') rgb = divergingColor(t);
        else if (type === 'cosine') rgb = cosineColor(t);
        else rgb = sequentialColor((t + 1) / 2);
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(x, 0, 1, c.height);
    }
}

// ── Heatmap renderer ──
function renderHeatmap(canvas, data, colorFn, wordLabelsX, wordLabelsY) {
    const rows = data.length;
    const cols = data[0].length;

    // Layout
    const cellW = wordLabelsX ? Math.max(4, Math.min(40, Math.floor(360 / cols))) : 4;
    const cellH = wordLabelsX ? Math.max(1, Math.min(4, Math.floor(300 / rows))) : Math.max(1, Math.min(4, Math.floor(300 / rows)));
    const marginLeft = wordLabelsY ? 70 : 10;
    const marginBottom = wordLabelsX ? 60 : 10;
    const marginTop = 5;
    const marginRight = 5;

    const w = marginLeft + cols * cellW + marginRight;
    const h = marginTop + rows * cellH + marginBottom;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Draw cells
    const imgData = ctx.createImageData(cols * cellW, rows * cellH);
    for (let r = 0; r < rows; r++) {
        for (let c2 = 0; c2 < cols; c2++) {
            const rgb = colorFn(data[r][c2]);
            for (let dy = 0; dy < cellH; dy++) {
                for (let dx = 0; dx < cellW; dx++) {
                    const px = (r * cellH + dy) * cols * cellW * 4 + (c2 * cellW + dx) * 4;
                    imgData.data[px] = rgb[0];
                    imgData.data[px + 1] = rgb[1];
                    imgData.data[px + 2] = rgb[2];
                    imgData.data[px + 3] = 255;
                }
            }
        }
    }
    ctx.putImageData(imgData, marginLeft, marginTop);

    // X labels
    if (wordLabelsX) {
        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i < wordLabelsX.length; i++) {
            const x = marginLeft + i * cellW + cellW / 2;
            const y = marginTop + rows * cellH + 8;
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(Math.PI / 3);
            ctx.fillText(wordLabelsX[i], 0, 0);
            ctx.restore();
        }
    }

    // Y labels
    if (wordLabelsY) {
        ctx.fillStyle = '#666';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i < wordLabelsY.length; i++) {
            const y = marginTop + i * cellH + cellH / 2;
            ctx.fillText(wordLabelsY[i], marginLeft - 4, y);
        }
    }
}

// ── Section 1: Semantic Sweeps ──
function initSweepDropdown() {
    const sel = document.getElementById('sweep-select');
    for (const name of Object.keys(DATA.sweeps)) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    }
}

function renderSweep() {
    const sel = document.getElementById('sweep-select');
    const axisName = sel.value;
    const axis = DATA.sweeps[axisName];
    if (!axis) return;

    const row = document.getElementById('sweep-row');
    row.innerHTML = '';

    for (const mn of DATA.modelNames) {
        const md = axis.models[mn];
        if (!md) continue;

        const panel = document.createElement('div');
        panel.className = 'panel';

        const label = document.createElement('div');
        label.className = 'panel-label';
        label.style.color = DATA.modelColors[mn];
        label.textContent = mn;
        panel.appendChild(label);

        const canvas = document.createElement('canvas');
        panel.appendChild(canvas);
        row.appendChild(panel);

        renderHeatmap(canvas, md.heatmap, divergingColor, axis.words, null);
    }
}

// ── Section 2: Cosine Similarity ──
function initCosineDropdown() {
    const sel = document.getElementById('cosine-select');
    for (const name of Object.keys(DATA.cosine)) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    }
}

function renderCosine() {
    const sel = document.getElementById('cosine-select');
    const seqName = sel.value;
    const seq = DATA.cosine[seqName];
    if (!seq) return;

    const row = document.getElementById('cosine-row');
    row.innerHTML = '';

    for (const mn of DATA.modelNames) {
        const simMatrix = seq.models[mn];
        if (!simMatrix) continue;

        const panel = document.createElement('div');
        panel.className = 'panel';

        const label = document.createElement('div');
        label.className = 'panel-label';
        label.style.color = DATA.modelColors[mn];
        label.textContent = mn;
        panel.appendChild(label);

        const canvas = document.createElement('canvas');
        panel.appendChild(canvas);
        row.appendChild(panel);

        // For cosine sim, use larger cells since matrix is small
        const n = simMatrix.length;
        const cellSize = Math.max(8, Math.min(30, Math.floor(250 / n)));
        const marginLeft = 70;
        const marginBottom = 60;
        const marginTop = 5;
        const w = marginLeft + n * cellSize + 5;
        const h = marginTop + n * cellSize + marginBottom;
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, w, h);

        // Draw matrix
        for (let r = 0; r < n; r++) {
            for (let c = 0; c < n; c++) {
                const v = simMatrix[r][c];
                const rgb = cosineColor(v);
                ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
                ctx.fillRect(marginLeft + c * cellSize, marginTop + r * cellSize, cellSize - 1, cellSize - 1);
            }
        }

        // Labels
        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        for (let i = 0; i < n; i++) {
            // Bottom
            ctx.save();
            ctx.translate(marginLeft + i * cellSize + cellSize / 2, marginTop + n * cellSize + 8);
            ctx.rotate(Math.PI / 3);
            ctx.textAlign = 'right';
            ctx.fillText(seq.words[i], 0, 0);
            ctx.restore();
            // Left
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(seq.words[i], marginLeft - 4, marginTop + i * cellSize + cellSize / 2);
        }
    }
}

// ── Section 3: PCA Loops ──
function initPCADropdown() {
    const sel = document.getElementById('pca-select');
    for (const name of Object.keys(DATA.pca)) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    }
}

function renderPCA() {
    const sel = document.getElementById('pca-select');
    const seqName = sel.value;
    const seq = DATA.pca[seqName];
    if (!seq) return;

    const row = document.getElementById('pca-row');
    row.innerHTML = '';

    const size = 250;
    const pad = 35;

    for (const mn of DATA.modelNames) {
        const pts = seq.models[mn];
        if (!pts) continue;

        const panel = document.createElement('div');
        panel.className = 'panel';

        const label = document.createElement('div');
        label.className = 'panel-label';
        label.style.color = DATA.modelColors[mn];
        label.textContent = mn;
        panel.appendChild(label);

        const canvas = document.createElement('canvas');
        canvas.width = size + pad * 2;
        canvas.height = size + pad * 2;
        canvas.style.width = (size + pad * 2) + 'px';
        canvas.style.height = (size + pad * 2) + 'px';
        panel.appendChild(canvas);
        row.appendChild(panel);

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw axes
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad, pad + size / 2);
        ctx.lineTo(pad + size, pad + size / 2);
        ctx.moveTo(pad + size / 2, pad);
        ctx.lineTo(pad + size / 2, pad + size);
        ctx.stroke();

        // Map points to canvas
        const toX = v => pad + (v + 1) / 2 * size;
        const toY = v => pad + (1 - (v + 1) / 2) * size;

        const color = DATA.modelColors[mn];

        // Draw connecting lines
        ctx.strokeStyle = color + '80';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < pts.length; i++) {
            const x = toX(pts[i][0]);
            const y = toY(pts[i][1]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        // Close loop
        ctx.lineTo(toX(pts[0][0]), toY(pts[0][1]));
        ctx.stroke();

        // Draw points and labels
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        for (let i = 0; i < pts.length; i++) {
            const x = toX(pts[i][0]);
            const y = toY(pts[i][1]);

            // Point
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();

            // Label
            ctx.fillStyle = '#ccc';
            ctx.fillText(seq.words[i], x, y - 8);
        }
    }
}

// ── Section 4: PCA Sine Waves ──
function initWaveDropdown() {
    const sel = document.getElementById('wave-select');
    for (const name of Object.keys(DATA.pcaWaves)) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    }
}

function renderWaves() {
    const sel = document.getElementById('wave-select');
    const axisName = sel.value;
    const axis = DATA.pcaWaves[axisName];
    if (!axis) return;

    const row = document.getElementById('wave-row');
    row.innerHTML = '';

    const W = 350, H = 280, pad = 40, padBottom = 50;

    for (const mn of DATA.modelNames) {
        const md = axis.models[mn];
        if (!md) continue;

        const panel = document.createElement('div');
        panel.className = 'panel';

        const label = document.createElement('div');
        label.className = 'panel-label';
        label.style.color = DATA.modelColors[mn];
        label.textContent = mn + ' (' + DATA.modelDescs[mn] + ')';
        panel.appendChild(label);

        const canvas = document.createElement('canvas');
        canvas.width = W;
        canvas.height = H;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        panel.appendChild(canvas);
        row.appendChild(panel);

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, W, H);

        // Draw grid
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        // Horizontal zero line
        const zeroY = pad + (H - pad - padBottom) / 2;
        ctx.moveTo(pad, zeroY);
        ctx.lineTo(W - 10, zeroY);
        // Vertical lines at word positions
        const nWords = md.curves.length;
        for (let i = 0; i < nWords; i++) {
            const x = pad + i * (W - pad - 10) / (nWords - 1);
            ctx.moveTo(x, pad);
            ctx.lineTo(x, H - padBottom);
        }
        ctx.stroke();

        // Component colors (rainbow-ish)
        const compColors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#9b59b6', '#e67e22'];
        const nComp = md.curves[0].length;

        // Draw each component as a smooth curve
        for (let c = 0; c < nComp; c++) {
            ctx.strokeStyle = compColors[c % compColors.length];
            ctx.lineWidth = 2.5;
            ctx.beginPath();

            for (let i = 0; i < nWords; i++) {
                const x = pad + i * (W - pad - 10) / (nWords - 1);
                const v = md.curves[i][c];
                const y = pad + (1 - v) / 2 * (H - pad - padBottom);
                if (i === 0) ctx.moveTo(x, y);
                else {
                    // Smooth curve using bezier
                    const prevX = pad + (i - 1) * (W - pad - 10) / (nWords - 1);
                    const prevV = md.curves[i - 1][c];
                    const prevY = pad + (1 - prevV) / 2 * (H - pad - padBottom);
                    const cpx = (prevX + x) / 2;
                    ctx.bezierCurveTo(cpx, prevY, cpx, y, x, y);
                }
            }
            ctx.stroke();

            // Dot at each word
            for (let i = 0; i < nWords; i++) {
                const x = pad + i * (W - pad - 10) / (nWords - 1);
                const v = md.curves[i][c];
                const y = pad + (1 - v) / 2 * (H - pad - padBottom);
                ctx.fillStyle = compColors[c % compColors.length];
                ctx.beginPath();
                ctx.arc(x, y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Word labels
        ctx.fillStyle = '#888';
        ctx.font = '9px monospace';
        for (let i = 0; i < nWords; i++) {
            const x = pad + i * (W - pad - 10) / (nWords - 1);
            ctx.save();
            ctx.translate(x, H - padBottom + 8);
            ctx.rotate(Math.PI / 4);
            ctx.textAlign = 'left';
            ctx.fillText(axis.words[i], 0, 0);
            ctx.restore();
        }

        // Legend (variance explained)
        ctx.font = '9px monospace';
        for (let c = 0; c < nComp; c++) {
            ctx.fillStyle = compColors[c % compColors.length];
            const varPct = md.var_explained[c].toFixed(1);
            ctx.fillText('PC' + (c + 1) + ' (' + varPct + '%)', W - 95, pad + c * 13);
        }
    }
}

// ── Section 5: PCA Surfaces ──
function initSurfaceDropdown() {
    const sel = document.getElementById('surface-select');
    for (const name of Object.keys(DATA.pcaSurfaces)) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    }
}

function renderSurfaces() {
    const sel = document.getElementById('surface-select');
    const seqName = sel.value;
    const seq = DATA.pcaSurfaces[seqName];
    if (!seq) return;

    const row = document.getElementById('surface-row');
    row.innerHTML = '';

    for (const mn of DATA.modelNames) {
        const surface = seq.models[mn];
        if (!surface) continue;

        const panel = document.createElement('div');
        panel.className = 'panel';

        const label = document.createElement('div');
        label.className = 'panel-label';
        label.style.color = DATA.modelColors[mn];
        label.textContent = mn;
        panel.appendChild(label);

        const canvas = document.createElement('canvas');
        panel.appendChild(canvas);
        row.appendChild(panel);

        // surface is [n_comp, 100] — render as heatmap
        const nComp = surface.length;
        const nInterp = surface[0].length;
        const cellW = 3;
        const cellH = Math.max(6, Math.min(15, Math.floor(200 / nComp)));
        const marginLeft = 40;
        const marginBottom = 30;
        const marginTop = 5;
        const w = marginLeft + nInterp * cellW + 5;
        const h = marginTop + nComp * cellH + marginBottom;
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, w, h);

        // Draw heatmap
        const imgData = ctx.createImageData(nInterp * cellW, nComp * cellH);
        for (let r = 0; r < nComp; r++) {
            for (let c = 0; c < nInterp; c++) {
                const v = surface[r][c];
                const rgb = divergingColor(v);
                for (let dy = 0; dy < cellH; dy++) {
                    for (let dx = 0; dx < cellW; dx++) {
                        const px = ((r * cellH + dy) * nInterp * cellW + (c * cellW + dx)) * 4;
                        imgData.data[px] = rgb[0];
                        imgData.data[px + 1] = rgb[1];
                        imgData.data[px + 2] = rgb[2];
                        imgData.data[px + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imgData, marginLeft, marginTop);

        // Y labels (PC1, PC2, ...)
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let r = 0; r < nComp; r++) {
            ctx.fillText('PC' + (r + 1), marginLeft - 4, marginTop + r * cellH + cellH / 2);
        }

        // X labels (word names at original positions)
        const words = seq.words;
        ctx.fillStyle = '#888';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        for (let i = 0; i < words.length; i++) {
            const xFrac = i / (words.length - 1);
            const x = marginLeft + xFrac * nInterp * cellW;
            ctx.save();
            ctx.translate(x, marginTop + nComp * cellH + 8);
            ctx.rotate(Math.PI / 4);
            ctx.textAlign = 'left';
            ctx.fillText(words[i], 0, 0);
            ctx.restore();
        }
    }
}

// ── Section 6: Fingerprints ──
function renderFingerprints() {
    const row = document.getElementById('fp-row');
    row.innerHTML = '';
    const cats = DATA.fingerprints.categories;

    for (const mn of DATA.modelNames) {
        const hm = DATA.fingerprints.models[mn];
        if (!hm || hm.length === 0) continue;

        const panel = document.createElement('div');
        panel.className = 'panel';

        const label = document.createElement('div');
        label.className = 'panel-label';
        label.style.color = DATA.modelColors[mn];
        label.textContent = mn;
        panel.appendChild(label);

        const canvas = document.createElement('canvas');
        panel.appendChild(canvas);
        row.appendChild(panel);

        // Custom render: 300 dims wide, N categories tall
        const nDims = hm[0].length;
        const nCats = hm.length;
        const cellW = 2;
        const cellH = 20;
        const marginLeft = 80;
        const marginBottom = 10;
        const marginTop = 5;
        const w = marginLeft + nDims * cellW + 5;
        const h = marginTop + nCats * cellH + marginBottom;
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, w, h);

        // Draw heatmap
        const imgData = ctx.createImageData(nDims * cellW, nCats * cellH);
        for (let r = 0; r < nCats; r++) {
            for (let c = 0; c < nDims; c++) {
                const rgb = sequentialColor(hm[r][c]);
                for (let dy = 0; dy < cellH; dy++) {
                    for (let dx = 0; dx < cellW; dx++) {
                        const px = ((r * cellH + dy) * nDims * cellW + (c * cellW + dx)) * 4;
                        imgData.data[px] = rgb[0];
                        imgData.data[px + 1] = rgb[1];
                        imgData.data[px + 2] = rgb[2];
                        imgData.data[px + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imgData, marginLeft, marginTop);

        // Y labels
        ctx.fillStyle = '#888';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i < nCats && i < cats.length; i++) {
            ctx.fillText(cats[i], marginLeft - 6, marginTop + i * cellH + cellH / 2);
        }
    }
}

// ── Init ──
initSweepDropdown();
initCosineDropdown();
initPCADropdown();
initWaveDropdown();
initSurfaceDropdown();
renderSweep();
renderCosine();
renderPCA();
renderWaves();
renderSurfaces();
renderFingerprints();
drawColorbar('sweep-colorbar', 'diverging');
drawColorbar('cosine-colorbar', 'cosine');
drawColorbar('surface-colorbar', 'diverging');
drawColorbar('fp-colorbar', 'sequential');
</script>
</body>
</html>"""
    return html


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    models = load_models()

    sweep_data = compute_semantic_sweep(models)
    cosine_data = compute_cosine_matrices(models)
    pca_data = compute_pca_loops(models)
    pca_wave_data = compute_pca_waves(models)
    pca_surface_data = compute_pca_surfaces(models)
    fingerprint_data = compute_fingerprints(models)

    print("\nGenerating HTML...")
    html = generate_html(sweep_data, cosine_data, pca_data, pca_wave_data, pca_surface_data, fingerprint_data)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"Written to {OUTPUT_PATH} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
