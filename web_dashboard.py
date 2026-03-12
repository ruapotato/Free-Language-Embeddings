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
import json
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
            # V14/V15/V16/V17/V18/V19 format: step N [HYDRA+GEO] or [V17+GEO] or [V19] | en=X para=X parse=X | ...
            if "step" in line and ("[HYDRA" in line or "[V17" in line or "[V16" in line or "[V18" in line or "[V19]" in line or "[V20]" in line or "[V21]" in line):
                try:
                    def grab(pattern, text, default=0.0):
                        m = re.search(pattern, text)
                        return float(m.group(1)) if m else default

                    step_str = line.split("step")[1].split("[")[0].strip()
                    step = int(re.search(r"(\d+)", step_str).group(1))
                    row = {
                        "step": step,
                        "recon": grab(r"en=([\d.]+)", line),
                        "em_ema": grab(r"em_ema=([\d.]+)", line),
                        "lr": grab(r"lr ([\d.eE+-]+)", line),
                        "progress": grab(r"([\d.]+)%", line),
                        "fr_loss": grab(r"fr=([\d.]+)", line),
                        "es_loss": grab(r"es=([\d.]+)", line),
                        "de_loss": grab(r"de=([\d.]+)", line),
                        "pt_loss": grab(r"pt=([\d.]+)", line),
                        "zh_loss": grab(r"zh=([\d.]+)", line),
                        "ja_loss": grab(r"ja=([\d.]+)", line),
                        "para_loss": grab(r"para=([\d.]+)", line),
                        "parse_loss": grab(r"parse=([\d.]+)", line),
                    }
                    # V19: contrastive loss
                    ctr_val = grab(r"ctr=([\d.]+)", line, default=None)
                    if ctr_val is not None:
                        row["contrastive_loss"] = ctr_val
                    # V20: NLI + WordNet losses
                    nli_val = grab(r"nli=([\d.]+)", line, default=None)
                    if nli_val is not None:
                        row["nli_loss"] = nli_val
                    wn_noun_val = grab(r"wn_noun=([\d.]+)", line, default=None)
                    if wn_noun_val is not None:
                        row["wn_noun_loss"] = wn_noun_val
                    wn_axis_val = grab(r"wn_axis=([\d.]+)", line, default=None)
                    if wn_axis_val is not None:
                        row["wn_axis_loss"] = wn_axis_val
                    wn_tropo_val = grab(r"wn_tropo=([\d.]+)", line, default=None)
                    if wn_tropo_val is not None:
                        row["wn_tropo_loss"] = wn_tropo_val
                    # V15: geo scale and geo losses
                    geo_val = grab(r"geo=([\d.]+)", line, default=None)
                    if geo_val is not None:
                        row["geo_gate"] = geo_val
                    row["total_loss"] = row["recon"] + row.get("fr_loss", 0)
                    rows.append(row)
                except (ValueError, IndexError, AttributeError):
                    pass
                continue

            if "step" in line and "loss" in line and "recon=" in line:
                try:
                    parts = line.split("|")
                    step_str = parts[0].split("step")[1].strip()
                    step = int(re.search(r"(\d+)", step_str).group(1))
                    total_loss = float(parts[1].split("loss")[1].split("(")[0].strip())
                    detail = parts[1].split("(")[1].split(")")[0]

                    def grab(pattern, text, default=0.0):
                        m = re.search(pattern, text)
                        return float(m.group(1)) if m else default

                    full_line = line

                    # Detect V10 format: "em_ema=" present, no geometry losses
                    is_v10 = "em_ema=" in line

                    if is_v10:
                        # V10/V11/V12/V13: step N | loss X (recon=X fr=X) | em_ema=X | lr X | X%
                        lr_val = grab(r"lr ([\d.eE+-]+)", full_line)
                        progress = grab(r"([\d.]+)%", full_line)
                        row = {
                            "step": step,
                            "total_loss": total_loss,
                            "recon": grab(r"recon=([\d.]+)", detail),
                            "em_ema": grab(r"em_ema=([\d.]+)", full_line),
                            "lr": lr_val,
                            "progress": progress,
                        }
                        # V13 FR loss
                        fr_val = grab(r"fr=([\d.]+)", detail)
                        if fr_val > 0:
                            row["fr_loss"] = fr_val
                        rows.append(row)
                    else:
                        # V5-V9 format
                        sim_part = parts[2] if len(parts) > 2 else ""

                        # V8 phase tag (P1, P2s5, P3) — in last pipe section
                        phase_str = ""
                        if len(parts) >= 4:
                            tail = parts[-1]
                            pm = re.search(r"(P[123](?:s\d+)?)", tail)
                            if pm:
                                phase_str = pm.group(1)

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
                            # V7/V8 fields
                            "m_para": grab(r"m_para=([\d.]+)", detail),
                            "m_neg": grab(r"m_neg=([\d.]+)", detail),
                            "m_wo": grab(r"m_wo=([\d.]+)", detail),
                            "sp": grab(r"sp=([\d.]+)", detail),
                            "repul": grab(r"(?<![h])repul=([\d.]+)", detail),
                            "hrepul": grab(r"hrepul=([\d.]+)", detail),
                            "p_sim": grab(r"p_sim=([\d.]+)", sim_part),
                            "n_sim": grab(r"n_sim=([\d.]+)", sim_part),
                            "wo_sim": grab(r"wo_sim=([\d.]+)", sim_part),
                            "sp_sim": grab(r"sp_sim=([\d.]+)", sim_part),
                            "r_sim": grab(r"r_sim=([\d.]+)", sim_part),
                            "hr_max": grab(r"hr_max=([\d.]+)", sim_part),
                            "cls_acc": grab(r"cls_acc=([\d.]+)", sim_part),
                            "geo_gate": grab(r"geo=([\d.]+)", parts[3] if len(parts) > 3 else ""),
                            "phase": phase_str,
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
            if ("step" in line and "loss" in line and "recon=" in line) or \
               ("step" in line and ("[HYDRA" in line or "[V17" in line or "[V16" in line or "[V18" in line or "[V19]" in line or "[V20]" in line or "[V21]" in line)):
                try:
                    step_str = line.split("|")[0].split("step")[1].strip()
                    # Strip [HYDRA] tag if present
                    step_str = step_str.split("[")[0].strip()
                    last_step = int(re.search(r"(\d+)", step_str).group(1))
                except (ValueError, IndexError, AttributeError):
                    pass
            # V10 eval format: "EN EVAL: token_acc=X exact_match=X em_ema=X" (or just "EVAL: ...")
            # Only match EN EVAL or bare EVAL, not FR/ES/DE/PT/ZH/JA EVAL
            if "EVAL:" in line and "token_acc=" in line and not any(f"{lang} EVAL:" in line for lang in ["FR", "ES", "DE", "PT", "ZH", "JA", "PARSE"]):
                current_eval = {"step": last_step}
                for key in ["token_acc", "exact_match", "em_ema"]:
                    m = re.search(rf"{key}=([\d.]+)", line)
                    if m:
                        current_eval[key] = float(m.group(1))
            # V10 per-bucket line: "short: acc=X em=X | medium: ..."
            elif current_eval and "short:" in line and "medium:" in line:
                for bucket in ["short", "medium", "long"]:
                    am = re.search(rf"{bucket}: acc=([\d.]+) em=([\d.]+)", line)
                    if am:
                        current_eval[f"acc_{bucket}"] = float(am.group(1))
                        current_eval[f"em_{bucket}"] = float(am.group(2))
            # V5-V9 eval format: "EVAL: para_sim=X wo_sim=X ..."
            elif "EVAL:" in line and "para_sim" in line:
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
            # V12+ geometry line: "GEOMETRY: ..." or "GEOMETRY (TEST): ..."
            if "GEOMETRY" in line and "analogy=" in line:
                current_eval = current_eval or {"step": last_step}
                for key, field in [("analogy", "analogy_avg"), ("cluster_gap", "clustering_gap"),
                                   ("dir_con", "dir_consistency"), ("wo_sim", "word_order_sim")]:
                    m = re.search(rf"{key}=([-+\d.]+)", line)
                    if m:
                        current_eval[field] = float(m.group(1))
                for key in ["rank90", "rank95"]:
                    m = re.search(rf"{key}=(\d+)", line)
                    if m:
                        current_eval[key] = int(m.group(1))
            # V13/V14 FR eval: "FR EVAL: token_acc=X"
            if "FR EVAL:" in line and "token_acc=" in line:
                current_eval = current_eval or {"step": last_step}
                m = re.search(r"token_acc=([\d.]+)", line)
                if m:
                    current_eval["fr_token_acc"] = float(m.group(1))
            # V14+ ES eval: "ES EVAL: token_acc=X"
            if "ES EVAL:" in line and "token_acc=" in line:
                current_eval = current_eval or {"step": last_step}
                m = re.search(r"token_acc=([\d.]+)", line)
                if m:
                    current_eval["es_token_acc"] = float(m.group(1))
            # V19 DE eval
            if "DE EVAL:" in line and "token_acc=" in line:
                current_eval = current_eval or {"step": last_step}
                m = re.search(r"token_acc=([\d.]+)", line)
                if m:
                    current_eval["de_token_acc"] = float(m.group(1))
            # V19 PT eval
            if "PT EVAL:" in line and "token_acc=" in line:
                current_eval = current_eval or {"step": last_step}
                m = re.search(r"token_acc=([\d.]+)", line)
                if m:
                    current_eval["pt_token_acc"] = float(m.group(1))
            # V19 ZH eval
            if "ZH EVAL:" in line and "token_acc=" in line:
                current_eval = current_eval or {"step": last_step}
                m = re.search(r"token_acc=([\d.]+)", line)
                if m:
                    current_eval["zh_token_acc"] = float(m.group(1))
            # V19 JA eval
            if "JA EVAL:" in line and "token_acc=" in line:
                current_eval = current_eval or {"step": last_step}
                m = re.search(r"token_acc=([\d.]+)", line)
                if m:
                    current_eval["ja_token_acc"] = float(m.group(1))
            # V14 Parse eval: "PARSE EVAL: token_acc=X"
            if "PARSE EVAL:" in line and "token_acc=" in line:
                current_eval = current_eval or {"step": last_step}
                m = re.search(r"token_acc=([\d.]+)", line)
                if m:
                    current_eval["parse_token_acc"] = float(m.group(1))
            # V20: NLI PROBE
            if "NLI PROBE:" in line:
                current_eval = current_eval or {"step": last_step}
                for key in ["entail", "neutral", "contra"]:
                    m = re.search(rf"{key}=([\d.]+)", line)
                    if m:
                        current_eval[f"nli_{key}"] = float(m.group(1))
                if "ordered=YES" in line:
                    current_eval["nli_ordered"] = 1
                elif "ordered=NO" in line:
                    current_eval["nli_ordered"] = 0
            if "SLOT_STATS:" in line:
                current_eval = current_eval or {"step": last_step}
                slot_isos = {}
                for m in re.finditer(r"(\d+):([-+\d.]+)([YN])", line):
                    slot_isos[int(m.group(1))] = {
                        "iso": float(m.group(2)),
                        "assigned": m.group(3) == "Y",
                    }
                if slot_isos:
                    current_eval["slot_isos"] = slot_isos
            # Dynamic weights line: "Dynamic weights: short=X | medium=X | long=X"
            if current_eval and "Dynamic weights:" in line:
                for bucket in ["short", "medium", "long"]:
                    wm = re.search(rf"{bucket}=([\d.]+)", line)
                    if wm:
                        current_eval[f"dw_{bucket}"] = float(wm.group(1))
            # Flush eval when we hit the next step line
            if current_eval and (
                (("step" in line and "loss" in line and "recon=" in line) or
                 ("step" in line and ("[HYDRA" in line or "[V17" in line or "[V16" in line or "[V18" in line or "[V19]" in line or "[V20]" in line or "[V21]" in line)))
                and current_eval["step"] != last_step):
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
    for v in ["v21", "v20", "v19", "v18", "v17", "v16", "v15", "v14", "v13", "v12", "v11", "v10", "v9", "v8", "v7", "v6", "v5", "v4", "v3", "v2", "v1"]:
        if os.path.exists(os.path.join(LOG_DIR, f"concept_{v}.log")):
            return v
    return "v16"


def list_available_runs():
    """List all available log files for comparison."""
    runs = {}
    # Main version logs
    for v in ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21"]:
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
<div id="stage-banner" style="margin:0 8px;padding:8px 12px;background:#1a1a2e;border-radius:6px;font-family:monospace;font-size:13px;color:#ccc;display:none">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <span id="stage-phase" style="font-weight:bold;font-size:15px"></span>
        <span id="stage-detail" style="color:#aaa"></span>
        <div style="flex:1;min-width:120px">
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#888">
                <span id="stage-bar-label">Progress</span>
                <span id="stage-geo-pct"></span>
            </div>
            <div style="background:#333;border-radius:3px;height:8px;overflow:hidden;margin-top:2px">
                <div id="stage-geo-bar" style="height:100%;border-radius:3px;transition:width 0.5s"></div>
            </div>
        </div>
        <div style="font-size:11px;color:#888" id="stage-recon-ema"></div>
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
<details style="margin:8px 12px;color:#ccc;font-size:13px;line-height:1.7">
<summary style="cursor:pointer;color:#aaa;font-weight:bold;padding:4px 0;font-size:14px">&#9encyclop; Reference Guide (click to expand)</summary>

<div style="padding:12px 0">
<h4 style="color:#42a5f5;margin:12px 0 6px">How It Works (V16 — Lean Hydra + Programmatic Geometry)</h4>
<p style="color:#bbb;margin:0 0 8px">
The model compresses English text into <b>32 concept slots</b> (each a 16-dimensional vector, <b>512 dims total</b>).
Three <b>non-autoregressive parallel decoders</b> (each 6 layers deep) share the same concept bottleneck:
(1) <b>EN</b> reconstruction, (2) <b>Paraphrase</b>, (3) <b>Semantic Parse</b>.
FR/ES heads dropped &mdash; not project goals, and 3 heads provide sufficient pressure for language-independent encoding.
</p>
<p style="color:#bbb;margin:0 0 8px">
<b>V16 key idea:</b> Programmatic geometry data generation with <b>strict train/test vocabulary separation</b>.
Templates with slot filling produce 18K+ word-order combos, diverse analogies, direction pairs, and cluster sentences.
Geometry eval uses <b>unseen test vocabulary</b> &mdash; measures genuine generalization, not memorization.
Geometry losses active from step 0 (no gate), warmup over 5K steps.
</p>
<p style="color:#bbb;margin:0 0 8px">
Trains on <b>~978K pairs</b>: paraphrase (MRPC/QQP/NLI) + semantic parse.
Each step: EN recon + one sampled secondary head (50/50 para/parse) + 7 geometry losses (every 5 steps).
</p>

<h4 style="color:#ffa726;margin:12px 0 6px">Training Setup</h4>
<table style="border-collapse:collapse;width:100%;font-size:12px;margin-bottom:8px">
<tr style="border-bottom:1px solid #333">
  <td style="padding:4px 8px;color:#4fc3f7;font-weight:bold">Architecture</td>
  <td style="padding:4px 8px">~110M params &mdash; EN encoder (6L&times;384d) + bottleneck (32&times;16=512d) + 3 decoders (6L&times;384d each)</td>
</tr>
<tr style="border-bottom:1px solid #333">
  <td style="padding:4px 8px;color:#4fc3f7;font-weight:bold">Schedule</td>
  <td style="padding:4px 8px">600K steps, cosine LR 2e-4 &rarr; 1e-5, warmup 2K steps, batch size 32</td>
</tr>
<tr style="border-bottom:1px solid #333">
  <td style="padding:4px 8px;color:#4fc3f7;font-weight:bold">Data</td>
  <td style="padding:4px 8px">~978K pairs: paraphrase (MRPC/QQP/NLI entailment) + semantic parse (custom grammar)</td>
</tr>
<tr style="border-bottom:1px solid #333">
  <td style="padding:4px 8px;color:#4fc3f7;font-weight:bold">Sampling</td>
  <td style="padding:4px 8px">50% Paraphrase, 50% Parse</td>
</tr>
<tr style="border-bottom:1px solid #333">
  <td style="padding:4px 8px;color:#4fc3f7;font-weight:bold">Geometry</td>
  <td style="padding:4px 8px">No gate &mdash; warmup 0&rarr;1 over 5K steps from step 0. Every 5 steps.</td>
</tr>
<tr>
  <td style="padding:4px 8px;color:#4fc3f7;font-weight:bold">Losses</td>
  <td style="padding:4px 8px">EN recon + head CE + geo&times;(WO=2.0 + HRepul=1.0 + BRepul=0.3 + Analogy=2.0 + DirCon=1.5 + Cluster=1.5)</td>
</tr>
</table>

<div style="padding:8px 0">
<h4 style="color:#ef5350;margin:8px 0 4px">Metrics</h4>
<div style="columns:2;column-gap:24px">
<b style="color:#ef5350">EN Recon Loss</b> &mdash; EN reconstruction cross-entropy. Lower = better.<br>
<b style="color:#ab47bc">FR/ES Translation</b> &mdash; Cross-entropy for translating EN&rarr;target language through the bottleneck.<br>
<b style="color:#7e57c2">DE/PT/ZH/JA Translation</b> &mdash; Same as above for additional languages (V19). More languages = more geometric pressure on the bottleneck.<br>
<b style="color:#ffa726">Para Loss</b> &mdash; Paraphrase decoder CE. Tests meaning-preserving rewording.<br>
<b style="color:#66bb6a">Parse Loss</b> &mdash; Semantic parse decoder CE. Tests structural understanding.<br>
<b style="color:#ef5350">Contrastive (InfoNCE)</b> &mdash; Pushes translation pairs close together in bottleneck space while pushing unrelated batch items apart. Uses temperature=0.07. This is NOT a reconstruction loss&mdash;it directly shapes the geometry of the bottleneck vectors. Falls fast early as the model learns to group same-meaning sentences, then plateaus.<br>
<b style="color:#42a5f5">NLI Graded</b> &mdash; 3-tier contrastive: entailment pairs &rarr; sim 0.85, neutral &rarr; 0.50, contradiction &rarr; 0.15. Smooth L1 loss. Builds graded similarity structure in the bottleneck.<br>
<b style="color:#66bb6a">WN Noun Hierarchy</b> &mdash; Sentence pairs differing by one noun; target cosine sim tracks WordNet path_similarity (0.3 + 0.6*dist). Builds noun taxonomy in bottleneck geometry.<br>
<b style="color:#ffa726">WN Axis Consistency</b> &mdash; Adjective antonym pairs (big/small, hot/cold) &mdash; diff vectors within each axis should be consistent (1 - mean pairwise cosine). Builds directional axes.<br>
<b style="color:#ab47bc">WN Troponym Chains</b> &mdash; Verb specificity chains (move&rarr;run&rarr;sprint) &mdash; ordering margin + direction consistency. Builds verb hierarchy.<br>
<b>EN Token Acc</b> &mdash; Fraction of EN tokens the decoder gets right.<br>
<b>Exact Match</b> &mdash; Full EN sentence reconstructed perfectly.<br>
<b>EM EMA</b> &mdash; Exponential moving average of exact match (decay=0.99).<br>
<b>Geo Scale</b> &mdash; Geometry loss scale factor (0&ndash;1). Ramps from 0 to 1 over first 5K steps.<br>
</div>

<h4 style="color:#42a5f5;margin:12px 0 4px">Geometry Probes (every 500 steps, TEST vocab)</h4>
<div style="columns:2;column-gap:24px">
<b>Analogy</b> &mdash; a&minus;b+c&cong;d cosine score (test vocab). Want &gt;0.8.<br>
<b>Clustering Gap</b> &mdash; Within-group &minus; between-group sim (test vocab). Want &gt;0.05.<br>
<b>Dir Consistency</b> &mdash; Same attribute = same direction? (test vocab). Want &gt;0.3.<br>
<b>Word Order Sim</b> &mdash; Swapped-order pair similarity (test vocab). Want &lt;0.85.<br>
<b>Effective Rank</b> &mdash; SVD dims for 90%/95% variance. Higher = richer representations.<br>
</div>

<h4 style="color:#66bb6a;margin:12px 0 4px">7 Geometry Losses (from step 0)</h4>
<div style="columns:2;column-gap:24px">
<b>Word Order</b> &mdash; Push swapped-word pairs below sim 0.5. Weight 2.0.<br>
<b>Hard Repulsion</b> &mdash; Push top-8 most similar unrelated pairs below sim 0.1. Weight 1.0.<br>
<b>Batch Repulsion</b> &mdash; Push random batch pairs below sim 0.3. Weight 0.3.<br>
<b>Analogy</b> &mdash; Reward a&minus;b+c &cong; d structure. Target sim &gt;0.9. Weight 2.0.<br>
<b>Dir Consistency</b> &mdash; Same-attribute directions should align. Target sim &gt;0.8. Weight 1.5.<br>
<b>Cluster Sep</b> &mdash; Same-group close (&gt;0.5), different-group far (&lt;0.2). Weight 1.5.<br>
</div>

<h4 style="color:#66bb6a;margin:12px 0 4px">Diagnostic Output</h4>
<b>[OK]</b> = perfect reconstruction. <b>[DIFF] (X%)</b> = X% token overlap.<br>
EN diagnostics: prose, code, math, logic. Parse: structured output.
</div>
</div>
</details>
<div class="grid">
    <div class="card"><h3 id="h-recon">Reconstruction Loss</h3><canvas id="chart-recon"></canvas></div>
    <div class="card"><h3 id="h-contrastive">Contrastive Loss (InfoNCE)</h3><canvas id="chart-contrastive"></canvas></div>
    <div class="card"><h3 id="h-geo-losses">EM EMA / Geometry Losses</h3><canvas id="chart-geo-losses"></canvas></div>
    <div class="card"><h3 id="h-batch-sim">Token Accuracy / Batch Similarities</h3><canvas id="chart-batch-sim"></canvas></div>
    <div class="card"><h3 id="h-diag-sim">Exact Match / Eval Similarities</h3><canvas id="chart-diag-sim"></canvas></div>
    <div class="card"><h3 id="h-lr">Learning Rate Schedule</h3><canvas id="chart-rank"></canvas></div>
    <div class="card"><h3 id="h-dyn-weights">Dynamic Sampling Weights</h3><canvas id="chart-v6-cls"></canvas></div>
    <div class="card"><h3 id="h-margins">Margin Losses + Repulsion</h3><canvas id="chart-v7-margins"></canvas></div>
    <div class="card"><h3 id="h-geo">Clustering Gap &amp; Direction</h3><canvas id="chart-v6-geo"></canvas></div>
    <div class="card"><h3 id="h-slots">Analogy &amp; Word Order</h3><canvas id="chart-v6-slots"></canvas></div>
    <div class="card"><h3 id="h-slot-iso">Effective Rank</h3><canvas id="chart-slot-iso"></canvas></div>
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
    analogy: '#66bb6a', woSim2: '#ffa726', rank90c: '#42a5f5', rank95c: '#66bb6a',
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
                  callback: function(val, idx) {
                      const v = this.getLabelForValue(val);
                      return v >= 1000 ? (v/1000).toFixed(v >= 10000 ? 0 : 1)+'K' : v;
                  } },
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
let _phaseTransitions = [];  // set during updateDashboard

// Custom plugin: draws vertical phase lines on every chart
const phaseLinePlugin = {
    id: 'phaseLines',
    afterDraw(chart) {
        if (!_phaseTransitions.length) return;
        const labels = chart.data.labels;
        if (!labels || !labels.length) return;
        const xScale = chart.scales.x;
        const yScale = chart.scales.y;
        if (!xScale || !yScale) return;

        const ctx = chart.ctx;
        for (const t of _phaseTransitions) {
            // Find pixel position: interpolate between nearest labels
            let pixelX = null;
            // Check if step is within chart range
            if (t.step < labels[0] || t.step > labels[labels.length - 1]) continue;
            // Find bracketing labels
            for (let i = 0; i < labels.length - 1; i++) {
                if (labels[i] <= t.step && labels[i+1] >= t.step) {
                    const frac = (t.step - labels[i]) / (labels[i+1] - labels[i] || 1);
                    const px1 = xScale.getPixelForValue(i);
                    const px2 = xScale.getPixelForValue(i + 1);
                    pixelX = px1 + frac * (px2 - px1);
                    break;
                }
            }
            if (pixelX === null) continue;

            // Draw dashed vertical line
            ctx.save();
            ctx.beginPath();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = t.color;
            ctx.lineWidth = 2;
            ctx.moveTo(pixelX, yScale.top);
            ctx.lineTo(pixelX, yScale.bottom);
            ctx.stroke();

            // Draw label
            ctx.setLineDash([]);
            ctx.font = '10px monospace';
            ctx.fillStyle = t.color + 'cc';
            ctx.textAlign = 'left';
            ctx.fillText(t.label, pixelX + 4, yScale.top + 12);
            ctx.restore();
        }
    }
};
Chart.register(phaseLinePlugin);

function mkChart(id, config) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), config);
}

// Detect phase transitions from step data
function getPhaseTransitions(steps) {
    if (!steps || !steps.length) return [];
    const transitions = [];
    const phaseColors = { P1: '#42a5f5', P2: '#ffa726', P3: '#66bb6a' };
    const phaseLabels = { P1: 'Phase 1: Recon', P2: 'Phase 2: Slots', P3: 'Phase 3: Joint' };

    let geoOpened = false;
    for (let i = 1; i < steps.length; i++) {
        // Detect geo gate first opening (0 → >0) = Phase 0 → Phase 1 transition
        if (!geoOpened && steps[i].geo_gate > 0 && (steps[i-1].geo_gate === 0 || steps[i-1].geo_gate == null)) {
            geoOpened = true;
            transitions.push({
                step: steps[i].step,
                color: '#ab47bc',
                label: 'P1: Geo Gate Opens',
            });
        }
        // Detect major phase change (P1→P2, P2→P3)
        const cur = (steps[i].phase || '').substring(0, 2);
        const prev = (steps[i-1].phase || '').substring(0, 2);
        if (cur && prev && cur !== prev) {
            transitions.push({
                step: steps[i].step,
                color: phaseColors[cur] || '#fff',
                label: phaseLabels[cur] || cur,
            });
        }
    }
    return transitions;
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
        // Default compare to previous version (e.g. v14 when viewing v15)
        try {
            const dataResp = await fetch('/api/data');
            const dataJson = await dataResp.json();
            const curMatch = dataJson.run?.match(/v?(\d+)/);
            if (curMatch) {
                const prev = 'v' + (parseInt(curMatch[1]) - 1);
                if (runs[prev]) sel.value = prev;
            }
        } catch(e2) {}
        if (!sel.value && runs['v6_attempt1']) sel.value = 'v6_attempt1';
    } catch(e) { console.error(e); }
}

function updateDashboard(response) {
    const { run, steps, evals, compare_run, compare_steps, compare_evals } = response;
    if (!steps.length) return;

    const s = steps.map(d => d.step);
    const sw = Math.min(30, Math.max(5, Math.floor(steps.length / 10)));
    _phaseTransitions = getPhaseTransitions(steps);  // for auto-injection in mkChart
    const latest = steps[steps.length - 1];
    const isV10 = latest.em_ema != null;
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
    const reconDs = [ds('EN Recon', ema(steps.map(d => d.recon), sw), C.recon)];
    // V13+: FR translation loss
    const frLossVals = steps.map(d => d.fr_loss || 0);
    if (frLossVals.some(v => v > 0))
        reconDs.push(ds('FR Trans', ema(frLossVals, sw), '#ab47bc'));
    // V14: ES, Para, Parse losses
    const esLossVals = steps.map(d => d.es_loss || 0);
    if (esLossVals.some(v => v > 0))
        reconDs.push(ds('ES Trans', ema(esLossVals, sw), '#ff7043'));
    // V19: DE, PT, ZH, JA translation losses
    const deLossVals = steps.map(d => d.de_loss || 0);
    if (deLossVals.some(v => v > 0))
        reconDs.push(ds('DE Trans', ema(deLossVals, sw), '#7e57c2'));
    const ptLossVals = steps.map(d => d.pt_loss || 0);
    if (ptLossVals.some(v => v > 0))
        reconDs.push(ds('PT Trans', ema(ptLossVals, sw), '#ec407a'));
    const zhLossVals = steps.map(d => d.zh_loss || 0);
    if (zhLossVals.some(v => v > 0))
        reconDs.push(ds('ZH Trans', ema(zhLossVals, sw), '#42a5f5'));
    const jaLossVals = steps.map(d => d.ja_loss || 0);
    if (jaLossVals.some(v => v > 0))
        reconDs.push(ds('JA Trans', ema(jaLossVals, sw), '#66bb6a'));
    const paraLossVals = steps.map(d => d.para_loss || 0);
    if (paraLossVals.some(v => v > 0))
        reconDs.push(ds('Paraphrase', ema(paraLossVals, sw), '#26a69a'));
    const parseLossVals = steps.map(d => d.parse_loss || 0);
    if (parseLossVals.some(v => v > 0))
        reconDs.push(ds('Parse', ema(parseLossVals, sw), '#ffa726'));
    if (hasCmp && clippedCmpSteps.length) reconDs.push(cmpDs('Recon' + cmpLabel,
        ema(clippedCmpSteps.map(d => d.recon), ccw), C.cmpRecon));
    // Geo gate on right axis (0-1 scale)
    const geoGateVals = steps.map(d => d.geo_gate);
    if (geoGateVals.some(v => v > 0))
        reconDs.push(ds('Geo Gate', ema(geoGateVals, sw), '#ff9800', {yAxisID: 'y2', borderDash: [5, 3]}));
    mkChart('chart-recon', {
        type: 'line',
        data: { labels: hasCmp && clippedCmpSteps.length ? mergedLabels(s, ccs) : s, datasets: reconDs.map(d => {
            if (hasCmp && clippedCmpSteps.length && !d.label.includes('Geo Gate')) {
                const ml = mergedLabels(s, ccs);
                const isCmp = hasCmp && cmpLabel && d.label.includes(cmpLabel);
                if (isCmp) {
                    const raw = ema(clippedCmpSteps.map(x => x.recon), ccw);
                    const map = {}; ccs.forEach((st, i) => map[st] = raw[i]);
                    d.data = ml.map(st => map[st] != null ? map[st] : null);
                } else {
                    // Re-align existing data to merged labels
                    const map = {}; s.forEach((st, i) => map[st] = d.data[i]);
                    d.data = ml.map(st => map[st] != null ? map[st] : null);
                }
                d.spanGaps = true;
            }
            return d;
        }) },
        options: dualAxisOpts('Loss', 'Geo Gate', undefined, undefined, 0, 1.05),
    });

    // 1b. Contrastive / NLI / WordNet Losses (own chart, different scale)
    const ctrLossVals = steps.map(d => d.contrastive_loss || 0);
    const nliLossVals = steps.map(d => d.nli_loss || 0);
    const wnNounVals = steps.map(d => d.wn_noun_loss || 0);
    const wnAxisVals = steps.map(d => d.wn_axis_loss || 0);
    const wnTropoVals = steps.map(d => d.wn_tropo_loss || 0);
    const hasCtrChart = ctrLossVals.some(v => v > 0);
    const hasNliChart = nliLossVals.some(v => v > 0);
    const hasWnChart = wnNounVals.some(v => v > 0) || wnAxisVals.some(v => v > 0) || wnTropoVals.some(v => v > 0);
    if (hasCtrChart || hasNliChart || hasWnChart) {
        const ctrDs = [];
        if (hasCtrChart) ctrDs.push(ds('InfoNCE', ema(ctrLossVals, sw), '#ef5350'));
        if (hasNliChart) ctrDs.push(ds('NLI Graded', ema(nliLossVals, sw), '#42a5f5'));
        if (wnNounVals.some(v => v > 0)) ctrDs.push(ds('WN Noun', ema(wnNounVals, sw), '#66bb6a'));
        if (wnAxisVals.some(v => v > 0)) ctrDs.push(ds('WN Axis', ema(wnAxisVals, sw), '#ffa726'));
        if (wnTropoVals.some(v => v > 0)) ctrDs.push(ds('WN Tropo', ema(wnTropoVals, sw), '#ab47bc'));
        const chartTitle = hasNliChart || hasWnChart ? 'Semantic Structure Losses' : 'Contrastive Loss';
        document.getElementById('h-contrastive').textContent = chartTitle;
        mkChart('chart-contrastive', {
            type: 'line',
            data: { labels: s, datasets: ctrDs },
            options: chartOpts(chartTitle),
        });
        document.getElementById('h-contrastive').parentElement.style.display = '';
    } else {
        const ctrCard = document.getElementById('h-contrastive');
        if (ctrCard) ctrCard.parentElement.style.display = 'none';
    }

    // V10/V11: Update chart headings
    if (isV10) {
        document.getElementById('h-recon').textContent = 'Reconstruction Loss';
        document.getElementById('h-geo-losses').textContent = 'EM EMA (Training)';
        document.getElementById('h-batch-sim').textContent = 'Token & Bucket Accuracy';
        document.getElementById('h-diag-sim').textContent = 'Per-Bucket Exact Match';
        document.getElementById('h-lr').textContent = 'Learning Rate Schedule';
        document.getElementById('h-dyn-weights').textContent = 'Dynamic Sampling Weights';
    }

    // 2. V10/V11: Exact Match EMA
    if (isV10) {
        const emDs = [
            ds('EM EMA', steps.map(d => d.em_ema || 0), '#66bb6a'),
        ];
        mkChart('chart-geo-losses', {
            type: 'line',
            data: { labels: s, datasets: emDs },
            options: chartOpts('Exact Match', 0, 1.05),
        });
    }

    // 2b. V10: Per-bucket accuracy (replaces batch similarities for V10)
    if (isV10 && evals.length && evals.some(d => d.token_acc != null)) {
        const v10Evals = evals.filter(d => d.token_acc != null);
        const v10Es = v10Evals.map(d => d.step);
        const accDs = [
            ds('Token Acc', v10Evals.map(d => d.token_acc || 0), '#4fc3f7', {pointRadius: 3}),
            ds('Exact Match', v10Evals.map(d => d.exact_match || 0), '#66bb6a', {pointRadius: 3}),
        ];
        if (v10Evals.some(d => d.acc_short != null)) {
            accDs.push(ds('Short Acc', v10Evals.map(d => d.acc_short || 0), '#ab47bc', {pointRadius: 2, borderDash: [3, 2]}));
            accDs.push(ds('Med Acc', v10Evals.map(d => d.acc_medium || 0), '#ffa726', {pointRadius: 2, borderDash: [3, 2]}));
            accDs.push(ds('Long Acc', v10Evals.map(d => d.acc_long || 0), '#ef5350', {pointRadius: 2, borderDash: [3, 2]}));
        }
        mkChart('chart-batch-sim', {
            type: 'line',
            data: { labels: v10Es, datasets: accDs },
            options: chartOpts('Accuracy', 0, 1.05),
        });

        // Per-bucket exact match (replaces diagnostic similarities)
        if (v10Evals.some(d => d.em_short != null)) {
            const emDs2 = [
                ds('Short EM', v10Evals.map(d => d.em_short || 0), '#ab47bc', {pointRadius: 3}),
                ds('Medium EM', v10Evals.map(d => d.em_medium || 0), '#ffa726', {pointRadius: 3}),
                ds('Long EM', v10Evals.map(d => d.em_long || 0), '#ef5350', {pointRadius: 3}),
            ];
            mkChart('chart-diag-sim', {
                type: 'line',
                data: { labels: v10Es, datasets: emDs2 },
                options: chartOpts('Exact Match', 0, 1.05),
            });
        }
    }

    // 2. Geometry Losses (V5-V9 only)
    if (!isV10) {
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
    }

    // 3. Batch Similarities (V5-V9 only)
    if (!isV10) {
    const bsDs = [
        ds('Pos sim', ema(steps.map(d => d.p_sim), sw), C.pSim),
        ds('Neg sim', ema(steps.map(d => d.n_sim), sw), C.nSim),
    ];
    if (steps.some(d => d.wo_sim > 0))
        bsDs.push(ds('WO sim', ema(steps.map(d => d.wo_sim), sw), C.woSim));
    if (steps.some(d => d.sp_sim > 0))
        bsDs.push(ds('SP sim', ema(steps.map(d => d.sp_sim), sw), '#ab47bc'));
    if (steps.some(d => d.r_sim > 0))
        bsDs.push(ds('Repul sim', ema(steps.map(d => d.r_sim), sw), '#78909c'));
    if (hasCmp && clippedCmpSteps.length) {
        bsDs.push(cmpDs('Pos' + cmpLabel, ema(clippedCmpSteps.map(d => d.p_sim), ccw), C.cmpPSim));
        bsDs.push(cmpDs('Neg' + cmpLabel, ema(clippedCmpSteps.map(d => d.n_sim), ccw), C.cmpNSim));
    }
    mkChart('chart-batch-sim', {
        type: 'line',
        data: { labels: hasCmp && clippedCmpSteps.length ? mergedLabels(s, ccs) : s, datasets: bsDs.map(d => {
            if (hasCmp && clippedCmpSteps.length) {
                const ml = mergedLabels(s, ccs);
                const isCmp = hasCmp && cmpLabel && d.label.includes(cmpLabel);
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
    }

    // 4. Diagnostic Similarities (V5-V9 only)
    if (!isV10 && evals.length) {
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
                    const isCmp = hasCmp && cmpLabel && d.label.includes(cmpLabel);
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

    // 5. LR Schedule (V10/V11) or Effective Rank (V5-V9)
    if (isV10 && steps.some(d => d.lr > 0)) {
        mkChart('chart-rank', {
            type: 'line',
            data: { labels: s, datasets: [
                ds('Learning Rate', steps.map(d => d.lr || 0), '#4fc3f7'),
            ] },
            options: chartOpts('LR'),
        });
    } else if (evals.length && evals.some(d => d.rank90)) {
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

    // 6. Dynamic Sampling Weights (V10/V11) or V6 Classifier (V5-V9)
    if (isV10 && evals.length && evals.some(d => d.dw_short != null)) {
        const dwEvals = evals.filter(d => d.dw_short != null);
        const dwEs = dwEvals.map(d => d.step);
        mkChart('chart-v6-cls', {
            type: 'line',
            data: { labels: dwEs, datasets: [
                ds('Short', dwEvals.map(d => d.dw_short), '#ab47bc', {pointRadius: 3}),
                ds('Medium', dwEvals.map(d => d.dw_medium), '#ffa726', {pointRadius: 3}),
                ds('Long', dwEvals.map(d => d.dw_long), '#ef5350', {pointRadius: 3}),
                ds('Uniform', dwEvals.map(() => 0.333), '#666', {borderDash: [5, 3], pointRadius: 0}),
            ] },
            options: chartOpts('Weight', 0, 1.05),
        });
    } else if (!isV10) {
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
    }

    // 6b. V7: Margin Losses + Slot Paraphrase (V5-V9 only)
    if (isV10) {
        // V10/V11/V12: hide only the margins chart (repurpose geo/slots/iso for geometry)
        const marginsEl = document.getElementById('chart-v7-margins');
        if (marginsEl) marginsEl.closest('.card').style.display = 'none';
        // Hide per-bucket exact match if no bucket data
        if (!evals.some(d => d.em_short != null)) {
            const diagEl = document.getElementById('chart-diag-sim');
            if (diagEl) diagEl.closest('.card').style.display = 'none';
        }
        // Hide dynamic sampling weights if no data
        if (!evals.some(d => d.dw_short != null)) {
            const dwEl = document.getElementById('chart-v6-cls');
            if (dwEl) dwEl.closest('.card').style.display = 'none';
        }
        // Update headings for geometry charts
        document.getElementById('h-geo').textContent = 'Clustering Gap & Direction Consistency';
        document.getElementById('h-slots').textContent = 'Analogy Score & Word Order';
        document.getElementById('h-slot-iso').textContent = 'Effective Rank';
    }
    const hasV7 = !isV10 && steps.some(d => d.m_para > 0 || d.sp > 0);
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
        addV7('repul', 'Repulsion', '#78909c', 'y');
        addV7('hrepul', 'Hard Repul', '#455a64', 'y');
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
                const isCmp = hasCmp && cmpLabel && d.label.includes(cmpLabel);
                const src = isCmp
                    ? clippedCmpEvals.filter(x => x.clustering_gap != null)
                    : geoEvals;
                const srcSteps = src.map(x => x.step);
                const key = d.label.startsWith('Cls') && isCmp ? 'clustering_gap'
                          : d.label.startsWith('Dir') || d.label.startsWith('DirCon') ? 'dir_consistency'
                          : 'clustering_gap';
                const map = {}; srcSteps.forEach((st, i) => map[st] = src[i][key] || 0);
                d.data = allGeoSteps.map(st => map[st] != null ? map[st] : null);
                d.spanGaps = true;
                return d;
            }) },
            options: chartOpts('Score'),
        });
    } else if (isV10) {
        // V10/V11/V12 with no geometry data yet — show placeholder
        mkChart('chart-v6-geo', {
            type: 'line',
            data: { labels: [0], datasets: [ds('Waiting for geometry data...', [0], '#666')] },
            options: chartOpts('Score'),
        });
    }

    // 8. Analogy Score & Word Order (V12) or Slot Assignment (V5-V9)
    const hasAnalogy = evals.length && evals.some(d => d.analogy_avg != null);
    const hasCmpAnalogy = hasCmp && clippedCmpEvals.length && clippedCmpEvals.some(d => d.analogy_avg != null);
    const hasSlot = evals.length && evals.some(d => d.slot_assign != null);
    const hasCmpSlot = hasCmp && clippedCmpEvals.length && clippedCmpEvals.some(d => d.slot_assign != null);

    if (hasAnalogy || hasCmpAnalogy) {
        // V12 geometry: Analogy + Word Order
        const aEvals = evals.filter(d => d.analogy_avg != null);
        const aEs = aEvals.map(d => d.step);
        const aDs = [
            ds('Analogy Avg', aEvals.map(d => d.analogy_avg), '#66bb6a', {pointRadius: 3}),
        ];
        if (aEvals.some(d => d.word_order_sim != null))
            aDs.push(ds('Word Order Sim', aEvals.map(d => d.word_order_sim || 0), '#ffa726', {pointRadius: 3}));
        // Reference lines
        aDs.push(ds('Target (analogy ≥0.8)', aEvals.map(() => 0.8), '#66bb6a44', {borderDash: [5, 3], pointRadius: 0}));
        aDs.push(ds('Target (WO <0.85)', aEvals.map(() => 0.85), '#ffa72644', {borderDash: [5, 3], pointRadius: 0}));
        if (hasCmpAnalogy) {
            const ce = clippedCmpEvals.filter(d => d.analogy_avg != null);
            aDs.push(cmpDs('Analogy' + cmpLabel, ce.map(d => d.analogy_avg), '#66bb6a88', {pointRadius: 2}));
            if (ce.some(d => d.word_order_sim != null))
                aDs.push(cmpDs('WO Sim' + cmpLabel, ce.map(d => d.word_order_sim || 0), '#ffa72688', {pointRadius: 2}));
        }
        mkChart('chart-v6-slots', {
            type: 'line',
            data: { labels: aEs, datasets: aDs },
            options: chartOpts('Score', 0, 1.05),
        });
    } else if (hasSlot || hasCmpSlot) {
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
                const isCmp = hasCmp && cmpLabel && d.label.includes(cmpLabel);
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
    } else if (isV10) {
        // V10/V11/V12 with no geometry data yet — show placeholder
        mkChart('chart-v6-slots', {
            type: 'line',
            data: { labels: [0], datasets: [ds('Waiting for geometry data...', [0], '#666')] },
            options: chartOpts('Score', 0, 1.05),
        });
    }

    // 9. Effective Rank (V12) or Per-Slot Isolation (V8)
    const hasRank = evals.length && evals.some(d => d.rank90 != null);
    const hasCmpRank = hasCmp && clippedCmpEvals.length && clippedCmpEvals.some(d => d.rank90 != null);
    const lastEval = evals.length ? evals[evals.length - 1] : {};

    if (hasRank || hasCmpRank) {
        const rEvals = evals.filter(d => d.rank90 != null);
        const rEs = rEvals.map(d => d.step);
        const rDs = [
            ds('Rank 90%', rEvals.map(d => d.rank90), '#42a5f5', {pointRadius: 3}),
            ds('Rank 95%', rEvals.map(d => d.rank95 || 0), '#66bb6a', {pointRadius: 3}),
        ];
        if (hasCmpRank) {
            const ce = clippedCmpEvals.filter(d => d.rank90 != null);
            rDs.push(cmpDs('Rank90' + cmpLabel, ce.map(d => d.rank90), '#42a5f588', {pointRadius: 2}));
            rDs.push(cmpDs('Rank95' + cmpLabel, ce.map(d => d.rank95 || 0), '#66bb6a88', {pointRadius: 2}));
        }
        mkChart('chart-slot-iso', {
            type: 'line',
            data: { labels: rEs, datasets: rDs },
            options: chartOpts('Rank (dimensions)'),
        });
    } else if (lastEval.slot_isos) {
        const slotIds = Object.keys(lastEval.slot_isos).map(Number).sort((a,b) => a - b);
        const isoVals = slotIds.map(s => lastEval.slot_isos[s].iso);
        const barColors = slotIds.map(s =>
            lastEval.slot_isos[s].assigned ? '#66bb6a' :
            lastEval.slot_isos[s].iso > 0.1 ? '#ffa726' : '#ef5350');
        mkChart('chart-slot-iso', {
            type: 'bar',
            data: {
                labels: slotIds.map(s => s.toString()),
                datasets: [{
                    label: 'Isolation',
                    data: isoVals,
                    backgroundColor: barColors,
                    borderWidth: 0,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: items => 'Slot ' + items[0].label,
                            label: item => {
                                const s = lastEval.slot_isos[parseInt(item.label)];
                                return `iso=${item.raw.toFixed(3)} ${s.assigned ? '(assigned)' : '(not assigned)'}`;
                            },
                        },
                    },
                },
                scales: {
                    x: { ticks: { color: '#666', font: { size: 8 } }, grid: { display: false } },
                    y: { ticks: { color: '#666', font: { size: tickSize() } }, grid: { color: '#333' },
                         title: { display: !isMobile(), text: 'Isolation', color: '#888' } },
                },
            },
        });
    } else if (isV10) {
        mkChart('chart-slot-iso', {
            type: 'line',
            data: { labels: [0], datasets: [ds('Waiting for geometry data...', [0], '#666')] },
            options: chartOpts('Rank'),
        });
    } else {
        mkChart('chart-slot-iso', {
            type: 'line',
            data: { labels: [0], datasets: [ds('No slot data', [0], '#666')] },
            options: chartOpts('Isolation'),
        });
    }

    // 10. Stats panel
    const le = evals.length ? evals[evals.length - 1] : {};
    const fmt = (v, d) => v != null ? v.toFixed(d || 3) : '--';
    const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '--';
    const rating = (v, good, bad) => {
        if (v == null) return 'label';
        return v >= good ? 'good' : v >= bad ? 'warn' : 'bad';
    };

    const hasFr = latest.fr_loss > 0 || (le.fr_token_acc != null);
    const hasEs = latest.es_loss > 0 || (le.es_token_acc != null);
    const hasDe = latest.de_loss > 0 || (le.de_token_acc != null);
    const hasPt = latest.pt_loss > 0 || (le.pt_token_acc != null);
    const hasZh = latest.zh_loss > 0 || (le.zh_token_acc != null);
    const hasJa = latest.ja_loss > 0 || (le.ja_token_acc != null);
    const hasCtr = latest.contrastive_loss > 0;
    const hasPara = latest.para_loss > 0;
    const hasParse = latest.parse_loss > 0;
    const isHydra = hasEs || hasPara || hasParse;

    let html = `<span class="value">flm (${run.toUpperCase()})</span>`;
    if (hasCmp) html += `  <span class="label">vs ${compare_run}</span>`;
    html += `\n${'='.repeat(40)}\n`;
    html += `<span class="label"> Step:</span>  <span class="value">${latest.step.toLocaleString()}</span>`;
    html += `  <span class="label">LR:</span> <span class="value">${(latest.lr || 0).toExponential(2)}</span>`;
    html += `  <span class="label">Progress:</span> <span class="value">${(latest.progress || 0).toFixed(1)}%</span>\n`;

    // --- V14 Hydra: all 5 heads ---
    if (isHydra) {
        html += `\n<span class="label"> ── Decoder Heads ──────────────────────</span>\n`;
        // EN Recon
        html += `<span class="label"> EN Recon:</span>   <span class="value">${fmt(latest.recon, 4)}</span>`;
        if (le.token_acc != null) html += `  tok <span class="${rating(le.token_acc, 0.95, 0.8)}">${pct(le.token_acc)}</span>`;
        if (le.exact_match != null) html += `  em <span class="${rating(le.exact_match, 0.9, 0.5)}">${pct(le.exact_match)}</span>`;
        html += `\n`;
        // FR Trans
        if (hasFr) {
            html += `<span class="label"> FR Trans:</span>   <span class="value">${fmt(latest.fr_loss, 4)}</span>`;
            if (le.fr_token_acc != null) html += `  tok <span class="${rating(le.fr_token_acc, 0.5, 0.2)}">${pct(le.fr_token_acc)}</span>`;
            html += `\n`;
        }
        // ES Trans
        if (hasEs) {
            html += `<span class="label"> ES Trans:</span>   <span class="value">${fmt(latest.es_loss, 4)}</span>`;
            if (le.es_token_acc != null) html += `  tok <span class="${rating(le.es_token_acc, 0.5, 0.2)}">${pct(le.es_token_acc)}</span>`;
            html += `\n`;
        }
        // DE Trans
        if (hasDe) {
            html += `<span class="label"> DE Trans:</span>   <span class="value">${fmt(latest.de_loss, 4)}</span>`;
            if (le.de_token_acc != null) html += `  tok <span class="${rating(le.de_token_acc, 0.5, 0.2)}">${pct(le.de_token_acc)}</span>`;
            html += `\n`;
        }
        // PT Trans
        if (hasPt) {
            html += `<span class="label"> PT Trans:</span>   <span class="value">${fmt(latest.pt_loss, 4)}</span>`;
            if (le.pt_token_acc != null) html += `  tok <span class="${rating(le.pt_token_acc, 0.5, 0.2)}">${pct(le.pt_token_acc)}</span>`;
            html += `\n`;
        }
        // ZH Trans
        if (hasZh) {
            html += `<span class="label"> ZH Trans:</span>   <span class="value">${fmt(latest.zh_loss, 4)}</span>`;
            if (le.zh_token_acc != null) html += `  tok <span class="${rating(le.zh_token_acc, 0.5, 0.2)}">${pct(le.zh_token_acc)}</span>`;
            html += `\n`;
        }
        // JA Trans
        if (hasJa) {
            html += `<span class="label"> JA Trans:</span>   <span class="value">${fmt(latest.ja_loss, 4)}</span>`;
            if (le.ja_token_acc != null) html += `  tok <span class="${rating(le.ja_token_acc, 0.5, 0.2)}">${pct(le.ja_token_acc)}</span>`;
            html += `\n`;
        }
        // Paraphrase
        if (hasPara) {
            html += `<span class="label"> Paraphrase:</span> <span class="value">${fmt(latest.para_loss, 4)}</span>\n`;
        }
        // Parse
        if (hasParse) {
            html += `<span class="label"> Parse:</span>      <span class="value">${fmt(latest.parse_loss, 4)}</span>`;
            if (le.parse_token_acc != null) html += `  tok <span class="${rating(le.parse_token_acc, 0.5, 0.2)}">${pct(le.parse_token_acc)}</span>`;
            html += `\n`;
        }
        html += `<span class="label"> EM EMA:</span>  <span class="${rating(latest.em_ema, 0.9, 0.5)}">${pct(latest.em_ema)}</span>\n`;
        // V19: Contrastive loss
        if (hasCtr) {
            html += `<span class="label"> Contrastive:</span> <span class="value">${fmt(latest.contrastive_loss, 4)}</span>\n`;
        }
        // V20: NLI + WordNet losses
        const hasNli = latest.nli_loss > 0;
        const hasWn = latest.wn_noun_loss > 0 || latest.wn_axis_loss > 0 || latest.wn_tropo_loss > 0;
        if (hasNli || hasWn) {
            html += `\n<span class="label"> ── Semantic Structure ─────────────────</span>\n`;
            if (hasNli)
                html += `<span class="label"> NLI Graded:</span> <span class="value">${fmt(latest.nli_loss, 4)}</span>\n`;
            if (latest.wn_noun_loss > 0)
                html += `<span class="label"> WN Noun:</span>    <span class="value">${fmt(latest.wn_noun_loss, 4)}</span>\n`;
            if (latest.wn_axis_loss > 0)
                html += `<span class="label"> WN Axis:</span>    <span class="value">${fmt(latest.wn_axis_loss, 4)}</span>\n`;
            if (latest.wn_tropo_loss > 0)
                html += `<span class="label"> WN Tropo:</span>   <span class="value">${fmt(latest.wn_tropo_loss, 4)}</span>\n`;
        }
        // V20: NLI Probe results
        if (le.nli_entail != null) {
            html += `\n<span class="label"> NLI Probe:</span>`;
            html += ` E=<span class="${rating(le.nli_entail, 0.75, 0.5)}">${fmt(le.nli_entail)}</span>`;
            html += ` N=<span class="value">${fmt(le.nli_neutral)}</span>`;
            html += ` C=<span class="${rating(1 - (le.nli_contra || 0), 0.75, 0.5)}">${fmt(le.nli_contra)}</span>`;
            html += ` <span class="${le.nli_ordered === 1 ? 'good' : 'bad'}">${le.nli_ordered === 1 ? 'ORDERED' : 'UNORDERED'}</span>\n`;
        }
    } else if (hasFr) {
        html += `\n<span class="label">           EN Recon        FR Translation</span>\n`;
        html += `<span class="label"> Loss:</span>    <span class="value">${fmt(latest.recon, 4).padEnd(16)}</span>`;
        html += `  <span class="value">${fmt(latest.fr_loss, 4)}</span>\n`;
        if (le.token_acc != null || le.fr_token_acc != null) {
            html += `<span class="label"> Tok Acc:</span> <span class="${rating(le.token_acc, 0.95, 0.8)}">${pct(le.token_acc).padEnd(16)}</span>`;
            html += `  <span class="${rating(le.fr_token_acc, 0.5, 0.2)}">${pct(le.fr_token_acc)}</span>\n`;
        }
        if (le.exact_match != null)
            html += `<span class="label"> EM:</span>      <span class="${rating(le.exact_match, 0.9, 0.5)}">${pct(le.exact_match)}</span>\n`;
        html += `<span class="label"> EM EMA:</span>  <span class="${rating(latest.em_ema, 0.9, 0.5)}">${pct(latest.em_ema)}</span>\n`;
    } else if (isV10) {
        html += `\n<span class="label"> Recon:</span>     <span class="value">${fmt(latest.recon)}</span>\n`;
        html += `<span class="label"> EM EMA:</span>    <span class="${rating(latest.em_ema, 0.9, 0.5)}">${pct(latest.em_ema)}</span>\n`;
        if (le.token_acc != null)
            html += `<span class="label"> Token Acc:</span> <span class="${rating(le.token_acc, 0.95, 0.8)}">${pct(le.token_acc)}</span>\n`;
        if (le.exact_match != null)
            html += `<span class="label"> Exact Match:</span> <span class="${rating(le.exact_match, 0.9, 0.5)}">${pct(le.exact_match)}</span>\n`;
    } else {
        html += `\n<span class="label"> Recon:</span>     <span class="value">${fmt(latest.recon)}</span>\n`;
        html += `<span class="label"> Total:</span>     <span class="value">${fmt(latest.total_loss)}</span>\n`;
        if (latest.geo_gate != null && latest.geo_gate > 0)
            html += `<span class="label"> Geo Gate:</span>  <span class="${rating(latest.geo_gate, 0.8, 0.3)}">${pct(latest.geo_gate)}</span>\n`;
    }

    // --- Geometry ---
    if (le.clustering_gap != null || le.analogy_avg != null) {
        html += `\n<span class="label"> Geometry:</span>\n`;
        if (le.analogy_avg != null)
            html += `  Analogy:  <span class="${rating(le.analogy_avg, 0.8, 0.6)}">${fmt(le.analogy_avg)}</span> (>0.8)\n`;
        html += `  Gap:      <span class="${rating(le.clustering_gap, 0.1, 0.03)}">${fmt(le.clustering_gap, 4)}</span> (>0.05)\n`;
        if (le.dir_consistency != null)
            html += `  Dir Con:  <span class="${rating(le.dir_consistency, 0.5, 0.2)}">${fmt(le.dir_consistency)}</span> (>0.3)\n`;
        if (le.word_order_sim != null)
            html += `  WO Sim:   <span class="${rating(1 - le.word_order_sim, 0.15, 0.05)}">${fmt(le.word_order_sim)}</span> (<0.85)\n`;
        if (le.rank90 != null)
            html += `  Rank:     <span class="value">${le.rank90}/${le.rank95 || '?'}</span> (90%/95%)\n`;
    }

    // --- Comparison ---
    if (hasCmp && compare_steps.length) {
        const cl = compare_steps[compare_steps.length - 1];
        const ce = compare_evals.length ? compare_evals[compare_evals.length - 1] : {};
        html += `\n<span class="label"> ${compare_run} (${cl.step.toLocaleString()} steps):</span>\n`;
        html += `  Recon: <span class="value">${fmt(cl.recon)}</span>`;
        if (cl.em_ema != null) html += `  EM: <span class="value">${pct(cl.em_ema)}</span>`;
        html += `\n`;
        if (ce.analogy_avg != null)
            html += `  Analogy: <span class="value">${fmt(ce.analogy_avg)}</span>  `;
        if (ce.clustering_gap != null)
            html += `Gap: <span class="value">${fmt(ce.clustering_gap, 4)}</span>  `;
        if (ce.dir_consistency != null)
            html += `Dir: <span class="value">${fmt(ce.dir_consistency)}</span>`;
        if (ce.analogy_avg != null || ce.clustering_gap != null)
            html += `\n`;
        if (ce.word_order_sim != null)
            html += `  WO: <span class="value">${fmt(ce.word_order_sim)}</span>  `;
        if (ce.rank90 != null)
            html += `Rank: <span class="value">${ce.rank90}/${ce.rank95 || '?'}</span>`;
        if (ce.word_order_sim != null || ce.rank90 != null)
            html += `\n`;
    }

    document.getElementById('stats-panel').innerHTML = html;

    // 11. Training stage banner
    const banner = document.getElementById('stage-banner');
    const geo = latest.geo_gate;
    const phase = latest.phase || '';
    if (isV10 || geo != null || phase) {
        banner.style.display = 'block';
        const phaseEl = document.getElementById('stage-phase');
        const detailEl = document.getElementById('stage-detail');
        const geoPctEl = document.getElementById('stage-geo-pct');
        const geoBarEl = document.getElementById('stage-geo-bar');
        const reconEmaEl = document.getElementById('stage-recon-ema');

        // Phase display
        let phaseText = '', phaseColor = '#aaa', detailText = '';
        if (isV10) {
            const emEma = latest.em_ema || 0;
            const progress = latest.progress || 0;
            if (isHydra) {
                const heads = ['EN', hasFr ? 'FR' : '', hasEs ? 'ES' : '', hasDe ? 'DE' : '', hasPt ? 'PT' : '', hasZh ? 'ZH' : '', hasJa ? 'JA' : '', hasPara ? 'Para' : '', hasParse ? 'Parse' : ''].filter(Boolean);
                phaseText = `Hydra (${heads.length} Heads)`;
                phaseColor = '#ff7043';
                detailText = `${heads.join('+')} | EN: ${fmt(latest.recon, 3)} | EM: ${(emEma*100).toFixed(1)}%`;
            } else if (hasFr) {
                phaseText = 'Dual Decoder (EN + FR)';
                phaseColor = '#ab47bc';
                detailText = `EN recon: ${fmt(latest.recon, 3)} | FR trans: ${fmt(latest.fr_loss, 3)} | EM: ${(emEma*100).toFixed(1)}%`;
            } else {
                phaseText = 'Pure Reconstruction';
                phaseColor = '#4fc3f7';
                detailText = `Parallel decoder, diverse data. EM EMA: ${(emEma*100).toFixed(1)}%`;
            }
            // Use bar for training progress
            geoPctEl.textContent = progress.toFixed(1) + '%';
            geoBarEl.style.width = Math.min(progress, 100) + '%';
            geoBarEl.style.background = progress >= 75 ? '#66bb6a' : progress >= 25 ? '#ffa726' : '#4fc3f7';
            reconEmaEl.textContent = `LR: ${(latest.lr || 0).toExponential(2)} | Step ${latest.step.toLocaleString()} / 600K`;
        } else if (phase.startsWith('P1') && geo != null && geo === 0) {
            phaseText = 'Phase 0: Pure Recon';
            phaseColor = '#ef5350';
            detailText = 'Decoder-only training. Geometry fully suppressed until recon loss drops.';
        } else if (phase.startsWith('P1')) {
            phaseText = 'Phase 1: Recon + Geometry Ramp';
            phaseColor = '#42a5f5';
            detailText = 'Reconstruction driving training, geometry ramping up as recon improves.';
        } else if (phase.startsWith('P2')) {
            const slotMatch = phase.match(/P2s(\d+)/);
            const slotNum = slotMatch ? slotMatch[1] : '?';
            const slotNames = ['subject','object','animacy','age','size','color','shape','material',
                'weight','temperature','action','manner','speed','direction','location','spatial',
                'distance','tense','duration','time_ref','number','degree','sentiment','emotion',
                'arousal','quality','difficulty','negation','certainty','causation','formality','speech_act'];
            const slotName = slotNames[parseInt(slotNum)] || slotNum;
            phaseText = 'Phase 2: Slot Focus';
            phaseColor = '#ffa726';
            detailText = `Training slot ${slotNum} (${slotName}) — full gradient on focus, 5% on others. ${slotNum}/32 done.`;
        } else if (phase.startsWith('P3')) {
            phaseText = 'Phase 3: Joint Fine-tune';
            phaseColor = '#66bb6a';
            detailText = 'All slots equal, balanced refinement.';
        }
        phaseEl.textContent = phaseText;
        phaseEl.style.color = phaseColor;
        detailEl.textContent = detailText;

        // Geo gate bar
        const geoPct = geo != null ? geo : 0;
        geoPctEl.textContent = (geoPct * 100).toFixed(0) + '%';
        geoBarEl.style.width = (geoPct * 100) + '%';
        geoBarEl.style.background = geoPct >= 0.9 ? '#66bb6a' : geoPct >= 0.5 ? '#ffa726' : '#ef5350';

        // Recon EMA estimate (geo_scale = (0.5 - ema) / 0.4, so ema = 0.5 - geo_scale * 0.4)
        if (geo != null) {
            const estEma = (0.5 - geoPct * 0.4).toFixed(3);
            reconEmaEl.textContent = `Recon EMA ≈ ${estEma} (need < 0.1 for Phase 2)`;
        }
    } else {
        banner.style.display = 'none';
    }
}

async function refresh() {
    try {
        // Static mode: use embedded data
        if (window.__STATIC_DATA__) {
            updateDashboard(window.__STATIC_DATA__);
            return;
        }
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
if (window.__STATIC_DATA__) {
    // Static mode: no polling, no run loading, hide compare controls
    document.querySelector('.controls').style.display = 'none';
    document.querySelector('.header .status .live').textContent = '\u25cf';
    document.querySelector('.header .status .live').style.color = '#888';
    refresh();
} else {
    loadRuns();
    refresh();
    setInterval(refresh, 30000);
}
</script>
</body>
</html>
"""


def export_static(output_path, run=None):
    """Generate a self-contained static HTML dashboard with embedded data."""
    available = list_available_runs()
    if not run:
        run = detect_run()
    log_path = available.get(run)
    if not log_path:
        log_path = os.path.join(LOG_DIR, f"concept_{run}.log")

    step_data = downsample(parse_step_data(log_path))
    eval_data = parse_eval_data(log_path)

    data = {
        "run": run,
        "steps": step_data,
        "evals": eval_data,
        "compare_run": None,
        "compare_steps": [],
        "compare_evals": [],
    }

    # Inject static data into HTML before closing </head>
    data_script = f'<script>window.__STATIC_DATA__ = {json.dumps(data, separators=(",", ":"))};</script>'
    html = DASHBOARD_HTML.replace('</head>', data_script + '\n</head>', 1)

    # Update title to indicate static snapshot
    html = html.replace('<title>flm Dashboard</title>',
                        f'<title>flm Dashboard — {run.upper()} (static)</title>')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    step_count = step_data[-1]['step'] if step_data else 0
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported static dashboard: {output_path}")
    print(f"  Run: {run.upper()}, Steps: {step_count:,}, Size: {size_kb:.0f} KB")


def main():
    parser = argparse.ArgumentParser(description="flm web training dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--export", type=str, metavar="PATH",
                        help="Export static HTML dashboard (e.g. docs/dashboard.html)")
    parser.add_argument("--run", type=str, help="Run to display (default: auto-detect latest)")
    args = parser.parse_args()

    if args.export:
        export_static(args.export, run=args.run)
    else:
        print(f"flm Training Dashboard: http://localhost:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
