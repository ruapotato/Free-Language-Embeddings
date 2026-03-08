#!/usr/bin/env python3
"""
Web dashboard for flm concept autoencoder training.

Reads training logs directly, serves JSON data via Flask,
renders charts with Chart.js. Auto-refreshes every 30 seconds.
Mobile-friendly responsive layout. Compare current vs archived runs.

Usage:
    python web_dashboard.py              # http://localhost:8501
    python web_dashboard.py --port 9000  # custom port
"""

import os
import re
import glob
import argparse
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def parse_step_data(log_path):
    """Parse per-step training data from log file."""
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path) as f:
        for line in f:
            if "step" in line and "loss" in line and "recon=" in line:
                try:
                    parts = line.split("|")
                    step = int(parts[0].split("step")[1].strip())
                    total_loss = float(parts[1].split("loss")[1].split("(")[0].strip())
                    detail = parts[1].split("(")[1].split(")")[0]
                    sim_part = parts[2]

                    def grab(pattern, text, default=0.0):
                        m = re.search(pattern, text)
                        return float(m.group(1)) if m else default

                    rows.append({
                        "step": step,
                        "total_loss": total_loss,
                        "recon": grab(r"recon=([\d.]+)", detail),
                        "nce": grab(r"nce=([\d.]+)", detail),
                        "wo": grab(r"wo=([\d.]+)", detail),
                        "decorr": grab(r"decorr=([\d.]+)", detail),
                        "iso": grab(r"iso=([\d.]+)", detail),
                        "cls": grab(r"cls=([\d.]+)", detail),
                        "scon": grab(r"scon=([\d.]+)", detail),
                        "xrecon": grab(r"xrecon=([\d.]+)", detail),
                        "sts": grab(r"sts=([\d.]+)", detail),
                        # V7 fields
                        "m_para": grab(r"m_para=([\d.]+)", detail),
                        "m_neg": grab(r"m_neg=([\d.]+)", detail),
                        "m_wo": grab(r"m_wo=([\d.]+)", detail),
                        "sp": grab(r"sp=([\d.]+)", detail),
                        "p_sim": grab(r"p_sim=([\d.]+)", sim_part),
                        "n_sim": grab(r"n_sim=([\d.]+)", sim_part),
                        "wo_sim": grab(r"wo_sim=([\d.]+)", sim_part),
                        "sp_sim": grab(r"sp_sim=([\d.]+)", sim_part),
                        "cls_acc": grab(r"cls_acc=([\d.]+)", sim_part),
                    })
                except (ValueError, IndexError, AttributeError):
                    continue
    return rows


def parse_eval_data(log_path):
    """Parse EVAL and GEO diagnostic data from log file."""
    if not os.path.exists(log_path):
        return []
    rows = []
    last_step = 0
    current_eval = None
    with open(log_path) as f:
        for line in f:
            if "step" in line and "loss" in line and "recon=" in line:
                try:
                    last_step = int(line.split("|")[0].split("step")[1].strip())
                except (ValueError, IndexError):
                    pass
            if "EVAL:" in line:
                current_eval = {"step": last_step}
                for m in re.finditer(r"(\w+_sim)=([-\d.]+)", line):
                    current_eval[m.group(1)] = float(m.group(2))
                for m in re.finditer(r"(rank\d+)=(\d+)", line):
                    current_eval[m.group(1)] = int(m.group(2))
                iso_m = re.search(r"slot_iso=([-\d.]+)", line)
                if iso_m:
                    current_eval["slot_iso"] = float(iso_m.group(1))
                assign_m = re.search(r"slot_assign=(\d+)/32", line)
                if assign_m:
                    current_eval["slot_assign"] = int(assign_m.group(1))
            if "GEO:" in line:
                current_eval = current_eval or {"step": last_step}
                cg_m = re.search(r"clustering_gap=([-+\d.]+)", line)
                if cg_m:
                    current_eval["clustering_gap"] = float(cg_m.group(1))
                dc_m = re.search(r"dir_consistency=([\d.]+)", line)
                if dc_m:
                    current_eval["dir_consistency"] = float(dc_m.group(1))
                wi_m = re.search(r"within=([\d.]+)", line)
                if wi_m:
                    current_eval["within_sim"] = float(wi_m.group(1))
                bt_m = re.search(r"between=([\d.]+)", line)
                if bt_m:
                    current_eval["between_sim"] = float(bt_m.group(1))
            if current_eval and ("--- RECON" in line or
                                 ("step" in line and "loss" in line and "recon=" in line
                                  and current_eval["step"] != last_step)):
                rows.append(current_eval)
                current_eval = None
    if current_eval:
        rows.append(current_eval)
    return rows


def downsample(step_data, max_points=3000):
    """Downsample step data, keeping recent points at full resolution."""
    if len(step_data) <= max_points:
        return step_data
    keep_recent = min(1500, max_points // 2)
    n = len(step_data) - keep_recent
    stride = max(1, n // (max_points - keep_recent))
    return step_data[:n:stride] + step_data[n:]


def detect_run():
    """Find latest run with data."""
    for v in ["v7", "v6", "v5", "v4", "v3", "v2", "v1"]:
        if os.path.exists(os.path.join(LOG_DIR, f"concept_{v}.log")):
            return v
    return "v7"


def list_available_runs():
    """List all available log files for comparison."""
    runs = {}
    # Main version logs
    for v in ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]:
        path = os.path.join(LOG_DIR, f"concept_{v}.log")
        if os.path.exists(path):
            runs[v] = path
    # Archived attempt logs
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "concept_v*_attempt*.log"))):
        name = os.path.basename(path).replace("concept_", "").replace(".log", "")
        runs[name] = path
    return runs


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/runs")
def api_runs():
    return jsonify(list_available_runs())


@app.route("/api/data")
def api_data():
    run = request.args.get("run")
    available = list_available_runs()

    if not run:
        run = detect_run()

    log_path = available.get(run)
    if not log_path:
        log_path = os.path.join(LOG_DIR, f"concept_{run}.log")

    step_data = downsample(parse_step_data(log_path))
    eval_data = parse_eval_data(log_path)

    # Optional comparison run
    compare = request.args.get("compare")
    compare_steps = []
    compare_evals = []
    if compare and compare in available:
        compare_steps = downsample(parse_step_data(available[compare]))
        compare_evals = parse_eval_data(available[compare])

    return jsonify({
        "run": run,
        "steps": step_data,
        "evals": eval_data,
        "compare_run": compare or None,
        "compare_steps": compare_steps,
        "compare_evals": compare_evals,
    })


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>flm Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    padding: 8px;
    -webkit-text-size-adjust: 100%;
}
.header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    padding: 8px 12px; margin-bottom: 8px;
    background: #16213e; border-radius: 8px;
    gap: 6px;
}
.header h1 { font-size: 1.1em; color: #4fc3f7; white-space: nowrap; }
.header .status { font-size: 0.75em; color: #888; }
.header .status .live { color: #66bb6a; }

.controls {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    padding: 6px 12px; margin-bottom: 8px;
    background: #16213e; border-radius: 8px;
    font-size: 0.8em;
}
.controls label { color: #90caf9; }
.controls select {
    background: #0f3460; color: #e0e0e0; border: 1px solid #444;
    border-radius: 4px; padding: 4px 8px; font-size: 0.9em;
}
.controls .compare-toggle {
    display: flex; align-items: center; gap: 4px;
}
.controls .compare-toggle input[type="checkbox"] {
    accent-color: #4fc3f7;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
}
.card {
    background: #16213e; border-radius: 8px; padding: 10px;
    position: relative; min-width: 0;
}
.card h3 {
    font-size: 0.75em; color: #90caf9; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.card canvas { width: 100% !important; height: 220px !important; }
.stats-card {
    background: #16213e; border-radius: 8px; padding: 12px;
    font-family: 'Courier New', monospace; font-size: 0.7em;
    line-height: 1.5; white-space: pre; overflow-x: auto;
}
.stats-card .label { color: #78909c; }
.stats-card .value { color: #4fc3f7; }
.stats-card .good { color: #66bb6a; }
.stats-card .warn { color: #ffa726; }
.stats-card .bad { color: #ef5350; }

/* Tablet: 2 columns */
@media (max-width: 1100px) {
    .grid { grid-template-columns: 1fr 1fr; }
    .card canvas { height: 200px !important; }
}

/* Phone: 1 column */
@media (max-width: 600px) {
    body { padding: 4px; }
    .header { padding: 6px 8px; }
    .header h1 { font-size: 0.95em; }
    .header .status { font-size: 0.65em; }
    .controls { padding: 6px 8px; font-size: 0.75em; }
    .grid { grid-template-columns: 1fr; gap: 6px; }
    .card { padding: 8px; }
    .card h3 { font-size: 0.7em; }
    .card canvas { height: 180px !important; }
    .stats-card { font-size: 0.6em; padding: 8px; }
}
</style>
</head>
<body>
<div class="header">
    <h1>flm &mdash; Training Dashboard</h1>
    <div class="status">
        <span class="live">&#9679;</span> 30s
        &nbsp;| <span id="run-id">--</span>
        &nbsp;| Step <span id="current-step">--</span>
        &nbsp;| <span id="last-update">--</span>
    </div>
</div>
<div class="controls">
    <label>Compare:</label>
    <div class="compare-toggle">
        <input type="checkbox" id="compare-enabled">
        <select id="compare-select" disabled>
            <option value="">Loading...</option>
        </select>
    </div>
</div>
<div class="grid">
    <div class="card"><h3>Reconstruction Loss</h3><canvas id="chart-recon"></canvas></div>
    <div class="card"><h3>Geometry Losses</h3><canvas id="chart-geo-losses"></canvas></div>
    <div class="card"><h3>Batch Similarities</h3><canvas id="chart-batch-sim"></canvas></div>
    <div class="card"><h3>Diagnostic Similarities</h3><canvas id="chart-diag-sim"></canvas></div>
    <div class="card"><h3>Effective Rank & Slot Isolation</h3><canvas id="chart-rank"></canvas></div>
    <div class="card"><h3>Classifier + Contrastive</h3><canvas id="chart-v6-cls"></canvas></div>
    <div class="card"><h3>V7: Margin Losses</h3><canvas id="chart-v7-margins"></canvas></div>
    <div class="card"><h3>Clustering & Direction</h3><canvas id="chart-v6-geo"></canvas></div>
    <div class="card"><h3>Slot Assignment</h3><canvas id="chart-v6-slots"></canvas></div>
    <div class="stats-card" id="stats-panel">Loading...</div>
</div>

<script>
const C = {
    recon: '#ef5350', nce: '#66bb6a', wo: '#ffa726', decorr: '#ab47bc',
    iso: '#ff7043', cls: '#26a69a', scon: '#5c6bc0', xrecon: '#ffca28',
    pSim: '#66bb6a', nSim: '#ef5350', woSim: '#ffa726',
    para: '#66bb6a', unrel: '#42a5f5', woEval: '#ef5350',
    clsGap: '#ef5350', dirCon: '#42a5f5', slots: '#66bb6a',
    rank90: '#42a5f5', rank95: '#66bb6a', slotIso: '#ff7043',
    clsAcc: '#66bb6a',
    // Comparison colors — same hues, dimmed/dashed
    cmpRecon: '#ef535088', cmpPSim: '#66bb6a88', cmpNSim: '#ef535088',
    cmpWoSim: '#ffa72688', cmpScon: '#5c6bc088', cmpCls: '#26a69a88',
    cmpXrecon: '#ffca2888', cmpClsAcc: '#66bb6a88',
    cmpClsGap: '#ef535088', cmpDirCon: '#42a5f588', cmpSlots: '#66bb6a88',
};

// Responsive legend/tick sizing
function isMobile() { return window.innerWidth < 600; }
function legendSize() { return isMobile() ? 8 : 10; }
function tickSize() { return isMobile() ? 9 : 11; }

function chartOpts(yLabel, yMin, yMax) {
    return {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { labels: { color: '#aaa', font: { size: legendSize() },
                   boxWidth: isMobile() ? 10 : 20 } } },
        scales: {
            x: { ticks: { color: '#666', font: { size: tickSize() }, maxTicksLimit: isMobile() ? 5 : 10,
                  callback: v => v >= 1000 ? (v/1000).toFixed(0)+'K' : v },
                 grid: { color: '#333' } },
            y: { min: yMin, max: yMax, ticks: { color: '#666', font: { size: tickSize() } },
                 grid: { color: '#333' },
                 title: { display: !!yLabel && !isMobile(), text: yLabel, color: '#888' } },
        },
    };
}

function dualAxisOpts(yLabel, y2Label, yMin, yMax, y2Min, y2Max) {
    const base = chartOpts(yLabel, yMin, yMax);
    base.scales.y2 = {
        position: 'right', min: y2Min, max: y2Max,
        ticks: { color: '#66bb6a', font: { size: tickSize() } },
        grid: { drawOnChartArea: false },
        title: { display: !!y2Label && !isMobile(), text: y2Label, color: '#66bb6a' },
    };
    return base;
}

function ema(data, window) {
    window = window || 30;
    if (data.length < 2) return data;
    const alpha = 2 / (window + 1);
    const result = [data[0]];
    for (let i = 1; i < data.length; i++)
        result.push(alpha * data[i] + (1 - alpha) * result[i-1]);
    return result;
}

function ds(label, data, color, opts) {
    opts = opts || {};
    return {
        label, data, borderColor: color, backgroundColor: color + '33',
        borderWidth: opts.borderWidth || 1.5, pointRadius: opts.pointRadius || 0,
        tension: 0, fill: opts.fill || false,
        yAxisID: opts.yAxisID || 'y',
        borderDash: opts.borderDash || [],
        ...opts,
    };
}

// Comparison dataset helper — dashed, dimmer
function cmpDs(label, data, color, opts) {
    opts = opts || {};
    return ds(label, data, color, { borderDash: [6, 3], borderWidth: 1.2, ...opts });
}

let charts = {};
function mkChart(id, config) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), config);
}

// Load available runs for comparison dropdown
async function loadRuns() {
    try {
        const resp = await fetch('/api/runs');
        const runs = await resp.json();
        const sel = document.getElementById('compare-select');
        sel.innerHTML = '';
        for (const [name, path] of Object.entries(runs)) {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            sel.appendChild(opt);
        }
        // Default to attempt1 if available
        if (runs['v6_attempt1']) sel.value = 'v6_attempt1';
    } catch(e) { console.error(e); }
}

function updateDashboard(response) {
    const { run, steps, evals, compare_run, compare_steps, compare_evals } = response;
    if (!steps.length) return;

    const s = steps.map(d => d.step);
    const sw = Math.min(30, Math.max(5, Math.floor(steps.length / 10)));
    const latest = steps[steps.length - 1];
    const hasCmp = compare_steps && compare_steps.length > 0;
    const cs = hasCmp ? compare_steps.map(d => d.step) : [];
    const csw = hasCmp ? Math.min(30, Math.max(5, Math.floor(compare_steps.length / 10))) : 30;
    const cmpLabel = compare_run ? ` (${compare_run})` : '';

    document.getElementById('run-id').textContent = run.toUpperCase();
    document.getElementById('current-step').textContent = latest.step.toLocaleString();
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

    // Helper: merge labels from two step arrays for overlaid chart
    function mergedLabels(s1, s2) {
        const all = new Set([...s1, ...s2]);
        return Array.from(all).sort((a,b) => a - b);
    }

    // Clip comparison data to current run's max step so charts don't zoom out
    const maxStep = latest.step;
    const clippedCmpSteps = hasCmp ? compare_steps.filter(d => d.step <= maxStep) : [];
    const clippedCmpEvals = hasCmp ? compare_evals.filter(d => d.step <= maxStep) : [];
    const ccs = clippedCmpSteps.map(d => d.step);
    const ccw = clippedCmpSteps.length > 10 ? Math.min(30, Math.max(5, Math.floor(clippedCmpSteps.length / 10))) : csw;

    // 1. Reconstruction Loss (comparison clipped to current max step)
    const reconDs = [ds('Recon', ema(steps.map(d => d.recon), sw), C.recon)];
    if (hasCmp && clippedCmpSteps.length) reconDs.push(cmpDs('Recon' + cmpLabel,
        ema(clippedCmpSteps.map(d => d.recon), ccw), C.cmpRecon));
    mkChart('chart-recon', {
        type: 'line',
        data: { labels: hasCmp && clippedCmpSteps.length ? mergedLabels(s, ccs) : s, datasets: reconDs.map(d => {
            if (hasCmp && clippedCmpSteps.length) {
                const ml = mergedLabels(s, ccs);
                const isCmp = d.label.includes(cmpLabel);
                const srcSteps = isCmp ? ccs : s;
                const raw = isCmp
                    ? ema(clippedCmpSteps.map(x => x.recon), ccw)
                    : ema(steps.map(x => x.recon), sw);
                const map = {}; srcSteps.forEach((st, i) => map[st] = raw[i]);
                d.data = ml.map(st => map[st] != null ? map[st] : null);
                d.spanGaps = true;
            }
            return d;
        }) },
        options: chartOpts('Loss'),
    });

    // 2. Geometry Losses
    const geoDs = [];
    const addGeo = (key, label, color) => {
        const vals = steps.map(d => d[key]);
        if (vals.some(v => v > 0))
            geoDs.push(ds(label, ema(vals, sw), color));
    };
    addGeo('nce', 'NCE', C.nce);
    addGeo('wo', 'Word Order', C.wo);
    addGeo('decorr', 'Decorrelation', C.decorr);
    addGeo('iso', 'Slot Isolation', C.iso);
    addGeo('cls', 'Classifier', C.cls);
    addGeo('scon', 'Slot Contrastive', C.scon);
    // Add comparison scon overlay (clipped to current max step)
    if (hasCmp && clippedCmpSteps.length) {
        const cmpScon = clippedCmpSteps.map(d => d.scon);
        if (cmpScon.some(v => v > 0))
            geoDs.push(cmpDs('Scon' + cmpLabel, ema(cmpScon, ccw), C.cmpScon));
    }
    mkChart('chart-geo-losses', {
        type: 'line',
        data: { labels: s, datasets: geoDs },
        options: chartOpts('Loss'),
    });

    // 3. Batch Similarities
    const bsDs = [
        ds('Pos sim', ema(steps.map(d => d.p_sim), sw), C.pSim),
        ds('Neg sim', ema(steps.map(d => d.n_sim), sw), C.nSim),
    ];
    if (steps.some(d => d.wo_sim > 0))
        bsDs.push(ds('WO sim', ema(steps.map(d => d.wo_sim), sw), C.woSim));
    if (steps.some(d => d.sp_sim > 0))
        bsDs.push(ds('SP sim', ema(steps.map(d => d.sp_sim), sw), '#ab47bc'));
    if (hasCmp && clippedCmpSteps.length) {
        bsDs.push(cmpDs('Pos' + cmpLabel, ema(clippedCmpSteps.map(d => d.p_sim), ccw), C.cmpPSim));
        bsDs.push(cmpDs('Neg' + cmpLabel, ema(clippedCmpSteps.map(d => d.n_sim), ccw), C.cmpNSim));
    }
    mkChart('chart-batch-sim', {
        type: 'line',
        data: { labels: hasCmp && clippedCmpSteps.length ? mergedLabels(s, ccs) : s, datasets: bsDs.map(d => {
            if (hasCmp && clippedCmpSteps.length) {
                const ml = mergedLabels(s, ccs);
                const isCmp = d.label.includes(cmpLabel);
                const src = isCmp ? clippedCmpSteps : steps;
                const srcSteps = isCmp ? ccs : s;
                const key = d.label.startsWith('Pos') ? 'p_sim'
                          : d.label.startsWith('Neg') ? 'n_sim' : 'wo_sim';
                const raw = ema(src.map(x => x[key]), isCmp ? ccw : sw);
                const map = {}; srcSteps.forEach((st, i) => map[st] = raw[i]);
                d.data = ml.map(st => map[st] != null ? map[st] : null);
                d.spanGaps = true;
            }
            return d;
        }) },
        options: chartOpts('Cosine Sim', -0.1, 1.05),
    });

    // 4. Diagnostic Similarities
    if (evals.length) {
        const es = evals.map(d => d.step);
        const diagDs = [
            ds('Paraphrase', evals.map(d => d.para_sim || 0), C.para, {pointRadius: 3}),
            ds('Unrelated', evals.map(d => d.unrelated_sim || 0), C.unrel, {pointRadius: 3}),
            ds('Word Order', evals.map(d => d.wo_sim || d.word_order_sim || 0), C.woEval, {pointRadius: 3}),
        ];
        if (hasCmp && clippedCmpEvals.length) {
            diagDs.push(cmpDs('Para' + cmpLabel, clippedCmpEvals.map(d => d.para_sim || 0), C.cmpPSim, {pointRadius: 2}));
            diagDs.push(cmpDs('Unrel' + cmpLabel, clippedCmpEvals.map(d => d.unrelated_sim || 0), C.cmpNSim, {pointRadius: 2}));
        }
        const allEvalSteps = hasCmp && clippedCmpEvals.length
            ? mergedLabels(es, clippedCmpEvals.map(d => d.step)) : es;
        mkChart('chart-diag-sim', {
            type: 'line',
            data: { labels: allEvalSteps, datasets: diagDs.map(d => {
                if (hasCmp && clippedCmpEvals.length) {
                    const isCmp = d.label.includes(cmpLabel);
                    const src = isCmp ? clippedCmpEvals : evals;
                    const srcSteps = src.map(x => x.step);
                    const key = d.label.startsWith('Para') ? 'para_sim'
                              : d.label.startsWith('Unrel') ? 'unrelated_sim'
                              : (d.label.startsWith('Word') ? 'wo_sim' : null);
                    if (key) {
                        const map = {}; srcSteps.forEach((st, i) => map[st] = src[i][key] || 0);
                        d.data = allEvalSteps.map(st => map[st] != null ? map[st] : null);
                        d.spanGaps = true;
                    }
                }
                return d;
            }) },
            options: chartOpts('Cosine Sim', -0.5, 1.05),
        });
    }

    // 5. Effective Rank & Slot Isolation
    if (evals.length && evals.some(d => d.rank90)) {
        const es = evals.map(d => d.step);
        const rkDs = [
            ds('Rank 90%', evals.map(d => d.rank90 || 0), C.rank90, {pointRadius: 3}),
            ds('Rank 95%', evals.map(d => d.rank95 || 0), C.rank95, {pointRadius: 3}),
        ];
        if (evals.some(d => d.slot_iso != null))
            rkDs.push(ds('Slot Iso', evals.map(d => d.slot_iso || 0), C.slotIso, {pointRadius: 3, yAxisID: 'y2'}));
        mkChart('chart-rank', {
            type: 'line',
            data: { labels: es, datasets: rkDs },
            options: dualAxisOpts('Rank', 'Slot Isolation'),
        });
    }

    // 6. V6 Classifier + Contrastive + Cross-Recon
    const v6Ds = [];
    const addV6 = (key, label, color, axis) => {
        const vals = steps.map(d => d[key]);
        if (vals.some(v => v > 0))
            v6Ds.push(ds(label, ema(vals, sw), color, {yAxisID: axis || 'y'}));
    };
    addV6('cls', 'Classifier', C.cls, 'y');
    addV6('scon', 'Slot Con', C.scon, 'y');
    addV6('xrecon', 'Cross-Recon', C.xrecon, 'y');
    addV6('cls_acc', 'Cls Acc', C.clsAcc, 'y2');
    if (hasCmp && clippedCmpSteps.length) {
        const cmpClsAcc = clippedCmpSteps.map(d => d.cls_acc);
        if (cmpClsAcc.some(v => v > 0))
            v6Ds.push(cmpDs('Acc' + cmpLabel, ema(cmpClsAcc, ccw), C.cmpClsAcc, {yAxisID: 'y2'}));
        const cmpXr = clippedCmpSteps.map(d => d.xrecon);
        if (cmpXr.some(v => v > 0))
            v6Ds.push(cmpDs('XRecon' + cmpLabel, ema(cmpXr, ccw), C.cmpXrecon));
    }
    mkChart('chart-v6-cls', {
        type: 'line',
        data: { labels: s, datasets: v6Ds },
        options: dualAxisOpts('Loss', 'Accuracy', undefined, undefined, 0, 1.05),
    });

    // 6b. V7: Margin Losses + Slot Paraphrase
    const hasV7 = steps.some(d => d.m_para > 0 || d.sp > 0);
    if (hasV7) {
        const v7Ds = [];
        const addV7 = (key, label, color, axis) => {
            const vals = steps.map(d => d[key]);
            if (vals.some(v => v > 0))
                v7Ds.push(ds(label, ema(vals, sw), color, {yAxisID: axis || 'y'}));
        };
        addV7('m_para', 'Margin Para', '#66bb6a', 'y');
        addV7('m_neg', 'Margin Neg', '#ef5350', 'y');
        addV7('m_wo', 'Margin WO', '#ffa726', 'y');
        addV7('sp', 'Slot Para', '#ab47bc', 'y');
        // sp_sim on right axis
        const spSimVals = steps.map(d => d.sp_sim);
        if (spSimVals.some(v => v > 0))
            v7Ds.push(ds('SP Sim', ema(spSimVals, sw), '#42a5f5', {yAxisID: 'y2'}));
        mkChart('chart-v7-margins', {
            type: 'line',
            data: { labels: s, datasets: v7Ds },
            options: dualAxisOpts('Loss', 'Similarity', undefined, undefined, 0, 1.05),
        });
    } else {
        // Show placeholder for non-V7 runs
        const ctx = document.getElementById('chart-v7-margins');
        if (ctx) {
            mkChart('chart-v7-margins', {
                type: 'line',
                data: { labels: [0], datasets: [ds('No V7 data', [0], '#666')] },
                options: chartOpts('Loss'),
            });
        }
    }

    // 7. Geometry: Clustering Gap & Direction Consistency
    const hasGeo = evals.length && evals.some(d => d.clustering_gap != null);
    const hasCmpGeo = hasCmp && clippedCmpEvals.length && clippedCmpEvals.some(d => d.clustering_gap != null);
    if (hasGeo || hasCmpGeo) {
        const geoEvals = evals.filter(d => d.clustering_gap != null);
        const ges = geoEvals.map(d => d.step);
        const gDs = [];
        if (hasGeo) {
            gDs.push(ds('Clustering Gap', geoEvals.map(d => d.clustering_gap), C.clsGap, {pointRadius: 3}));
            gDs.push(ds('Dir Consistency', geoEvals.map(d => d.dir_consistency || 0), C.dirCon, {pointRadius: 3}));
        }
        if (hasCmpGeo) {
            const cge = clippedCmpEvals.filter(d => d.clustering_gap != null);
            gDs.push(cmpDs('ClsGap' + cmpLabel, cge.map(d => d.clustering_gap), C.cmpClsGap, {pointRadius: 2}));
            gDs.push(cmpDs('DirCon' + cmpLabel, cge.map(d => d.dir_consistency || 0), C.cmpDirCon, {pointRadius: 2}));
        }
        const allGeoSteps = hasCmpGeo
            ? mergedLabels(ges, clippedCmpEvals.filter(d => d.clustering_gap != null).map(d => d.step))
            : ges;
        mkChart('chart-v6-geo', {
            type: 'line',
            data: { labels: allGeoSteps, datasets: gDs.map(d => {
                const isCmp = d.label.includes(cmpLabel);
                const src = isCmp
                    ? clippedCmpEvals.filter(x => x.clustering_gap != null)
                    : geoEvals;
                const srcSteps = src.map(x => x.step);
                const key = d.label.startsWith('Cls') ? 'clustering_gap' : 'dir_consistency';
                const map = {}; srcSteps.forEach((st, i) => map[st] = src[i][key] || 0);
                d.data = allGeoSteps.map(st => map[st] != null ? map[st] : null);
                d.spanGaps = true;
                return d;
            }) },
            options: chartOpts('Score'),
        });
    }

    // 8. Slot Assignment
    const hasSlot = evals.length && evals.some(d => d.slot_assign != null);
    const hasCmpSlot = hasCmp && clippedCmpEvals.length && clippedCmpEvals.some(d => d.slot_assign != null);
    if (hasSlot || hasCmpSlot) {
        const slotDs = [];
        if (hasSlot) {
            const se = evals.filter(d => d.slot_assign != null);
            slotDs.push(ds('Correct /32', se.map(d => d.slot_assign), C.slots, {pointRadius: 4, borderWidth: 2, fill: true}));
        }
        if (hasCmpSlot) {
            const cse = clippedCmpEvals.filter(d => d.slot_assign != null);
            slotDs.push(cmpDs('Slots' + cmpLabel, cse.map(d => d.slot_assign), C.cmpSlots, {pointRadius: 2}));
        }
        const allSlotSteps = (() => {
            const a = hasSlot ? evals.filter(d => d.slot_assign != null).map(d => d.step) : [];
            const b = hasCmpSlot ? clippedCmpEvals.filter(d => d.slot_assign != null).map(d => d.step) : [];
            return mergedLabels(a, b);
        })();
        mkChart('chart-v6-slots', {
            type: 'line',
            data: { labels: allSlotSteps, datasets: slotDs.map(d => {
                const isCmp = d.label.includes(cmpLabel);
                const src = isCmp
                    ? clippedCmpEvals.filter(x => x.slot_assign != null)
                    : evals.filter(x => x.slot_assign != null);
                const map = {}; src.forEach(x => map[x.step] = x.slot_assign);
                d.data = allSlotSteps.map(st => map[st] != null ? map[st] : null);
                d.spanGaps = true;
                return d;
            }) },
            options: chartOpts('Slots', 0, 33),
        });
    }

    // 9. Stats panel
    const le = evals.length ? evals[evals.length - 1] : {};
    const fmt = (v, d) => v != null ? v.toFixed(d || 3) : '--';
    const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '--';
    const rating = (v, good, bad) => {
        if (v == null) return 'label';
        return v >= good ? 'good' : v >= bad ? 'warn' : 'bad';
    };

    let html = `<span class="value">flm (${run.toUpperCase()})</span>`;
    if (hasCmp) html += `  <span class="label">vs ${compare_run}</span>`;
    html += `\n${'='.repeat(40)}\n\n`;
    html += `<span class="label"> Step:</span>      <span class="value">${latest.step.toLocaleString()}</span>\n`;
    html += `<span class="label"> Recon:</span>     <span class="value">${fmt(latest.recon)}</span>\n`;
    html += `<span class="label"> Total:</span>     <span class="value">${fmt(latest.total_loss)}</span>\n`;

    if (le.clustering_gap != null) {
        html += `\n<span class="label"> Geometry:</span>\n`;
        html += `  Gap:   <span class="${rating(le.clustering_gap, 0.1, 0.03)}">${fmt(le.clustering_gap, 4)}</span> (>0.1)\n`;
        html += `  Dir:   <span class="${rating(le.dir_consistency, 0.5, 0.2)}">${fmt(le.dir_consistency, 4)}</span> (>0.5)\n`;
    }
    if (le.slot_assign != null)
        html += `  Slots: <span class="${rating(le.slot_assign, 28, 16)}">${le.slot_assign}/32</span>\n`;

    if (latest.cls > 0) {
        html += `\n<span class="label"> V6 Losses:</span>\n`;
        html += `  Cls:   <span class="value">${fmt(latest.cls)}</span>  `;
        html += `Acc: <span class="${rating(latest.cls_acc, 0.95, 0.5)}">${pct(latest.cls_acc)}</span>\n`;
        html += `  SCon:  <span class="value">${fmt(latest.scon)}</span>  `;
        html += `XRec: <span class="value">${fmt(latest.xrecon)}</span>\n`;
    }

    // Comparison stats
    if (hasCmp && compare_steps.length) {
        const cl = compare_steps[compare_steps.length - 1];
        const ce = compare_evals.length ? compare_evals[compare_evals.length - 1] : {};
        html += `\n<span class="label"> ${compare_run} (${cl.step.toLocaleString()} steps):</span>\n`;
        html += `  Recon: <span class="value">${fmt(cl.recon)}</span>  `;
        html += `SCon: <span class="value">${fmt(cl.scon)}</span>\n`;
        if (ce.clustering_gap != null)
            html += `  Gap:   <span class="value">${fmt(ce.clustering_gap, 4)}</span>  `;
        if (ce.dir_consistency != null)
            html += `Dir: <span class="value">${fmt(ce.dir_consistency, 4)}</span>\n`;
        if (ce.slot_assign != null)
            html += `  Slots: <span class="value">${ce.slot_assign}/32</span>\n`;
    }

    document.getElementById('stats-panel').innerHTML = html;
}

async function refresh() {
    try {
        const cmpEnabled = document.getElementById('compare-enabled').checked;
        const cmpRun = document.getElementById('compare-select').value;
        let url = '/api/data';
        if (cmpEnabled && cmpRun) url += '?compare=' + encodeURIComponent(cmpRun);
        const resp = await fetch(url);
        const data = await resp.json();
        updateDashboard(data);
    } catch (e) {
        console.error('Refresh failed:', e);
    }
}

// Compare toggle
document.getElementById('compare-enabled').addEventListener('change', function() {
    document.getElementById('compare-select').disabled = !this.checked;
    refresh();
});
document.getElementById('compare-select').addEventListener('change', refresh);

// Init
loadRuns();
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="flm web training dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"flm Training Dashboard: http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
