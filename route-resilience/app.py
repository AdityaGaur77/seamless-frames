"""
app.py
------
Streamlit + Leaflet (Folium) dashboard for Route Resilience.

Run with:
    streamlit run app.py

What you can do
---------------
1.  Browse a tile from the dataset (clean / occluded / predicted mask).
2.  See the healed road graph overlaid on a Leaflet map.
3.  Inspect the resilience curve (global efficiency vs. # of nodes ablated).
4.  Click any node on the map to *disable* it and see the new
    shortest-path / efficiency instantly.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import List, Optional

import cv2
import folium
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

from analyze import (AnalyzeConfig, betweenness, criticality_scores,
                     edge_criticality, global_efficiency, resilience_curve,
                     to_networkx, top_gatekeepers)
from heal import HealConfig, heal_graph
from model import AttentionUNet
from pipeline import load_model, predict_mask
from skeletonize import extract_skeleton, render_overlay


# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Route Resilience Dashboard",
                   layout="wide", page_icon="🛣️")

st.title("🛣️ Route Resilience — Occlusion-Robust Road Criticality")
st.caption("Phase IV dashboard.  Drag the slider to ablate the top-N "
           "gatekeeper nodes; the map re-routes instantly.")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Config")
    ckpt = st.text_input("Checkpoint", value="checkpoints/best.pt")
    base = st.number_input("Base channels", value=16, min_value=8, max_value=64,
                           step=8)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    d_max = st.slider("Heal: max gap (px)", 5, 80, 35)
    ang_max = st.slider("Heal: angle gate (°)", 5.0, 90.0, 35.0)
    n_ablate = st.slider("Ablate top-N nodes", 1, 50, 5)


# --------------------------------------------------------------------------- #
# Load model (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource
def _load(ckpt_path: str, base: int, device: str):
    return load_model(ckpt_path, base=base, device=device)


model = _load(ckpt, int(base), device)

# --------------------------------------------------------------------------- #
# Choose tile
# --------------------------------------------------------------------------- #
img_dir = Path("data/synth/images")
mask_dir = Path("data/synth/masks")
occ_dir = Path("data/synth/occluded")
tiles = sorted([p.stem for p in img_dir.glob("*.png")]) if img_dir.exists() else []

if not tiles:
    st.error("No tiles in data/synth/images.  Run `python synth_data.py` first.")
    st.stop()

tile = st.selectbox("Tile", tiles)

col1, col2, col3 = st.columns(3)
clean = cv2.imread(str(img_dir / f"{tile}.png"))
clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
occ = cv2.imread(str(occ_dir / f"{tile}.png"))
occ_rgb = cv2.cvtColor(occ, cv2.COLOR_BGR2RGB)
mask = cv2.imread(str(mask_dir / f"{tile}.png"), cv2.IMREAD_GRAYSCALE)

with col1:
    st.markdown("**Clean**")
    st.image(clean_rgb, use_column_width=True)
with col2:
    st.markdown("**Occluded** (input)")
    st.image(occ_rgb, use_column_width=True)
with col3:
    st.markdown("**Ground truth mask**")
    st.image(mask, use_column_width=True)

# --------------------------------------------------------------------------- #
# Run prediction + analysis
# --------------------------------------------------------------------------- #
with st.spinner("Running model + healing + analysis…"):
    pred = predict_mask(model, occ, device=device)
    skel, nodes, edges = extract_skeleton(pred)
    nodes, edges, heal_info = heal_graph(nodes, edges,
                                        HealConfig(d_max=d_max, ang_max=ang_max))
    G = to_networkx(nodes, edges)
    betw = betweenness(G, AnalyzeConfig())
    nscore, ordered = criticality_scores(G, betw)
    ec = edge_criticality(G, nscore)
    base_eff = global_efficiency(G)
    full_curve = resilience_curve(G, ordered, AnalyzeConfig(top_k_ablations=n_ablate))

# Top stats
m1, m2, m3, m4 = st.columns(4)
m1.metric("Nodes", len(nodes))
m2.metric("Edges", len(edges))
m3.metric("Bridges added", heal_info["added"])
m4.metric("Baseline efficiency", f"{base_eff:.4f}")

# --------------------------------------------------------------------------- #
# Folium map
# --------------------------------------------------------------------------- #
st.subheader("🗺️  Routable graph — click a node to disable it")

# Map size
h, w = skel.shape
# Scale pixel coords to "fake lat/lon" centred on (12.97, 77.59) — Bengaluru
def to_latlon(y: int, x: int) -> tuple[float, float]:
    return (12.97 + (y - h / 2) * 1e-5,
            77.59 + (x - w / 2) * 1e-5)

disabled = st.session_state.setdefault("disabled", set())

m = folium.Map(location=(12.97, 77.59), zoom_start=17,
               tiles="CartoDB positron",
               width="100%", height="500px")

# Draw edges — colour by criticality
max_ec = max(ec.values()) if ec else 1.0
for (u, v), sc in ec.items():
    if u not in G.nodes or v not in G.nodes:
        continue
    y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
    y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
    intensity = sc / max_ec if max_ec > 0 else 0.0
    color = f"#{int(255*(1-intensity)):02x}{int(80*intensity):02x}00"
    weight = 1.5 + 3.5 * intensity
    folium.PolyLine(
        locations=[to_latlon(y1, x1), to_latlon(y2, x2)],
        color=color, weight=weight, opacity=0.7,
    ).add_to(m)

# Draw nodes
for nid, n in enumerate(nodes):
    if nid in disabled:
        continue
    sc = nscore.get(nid, 0.0)
    radius = 4 + 8 * sc
    color = "red" if n.kind == "junction" else "blue" if n.kind == "endpoint" else "orange"
    folium.CircleMarker(
        location=to_latlon(n.y, n.x),
        radius=radius,
        color=color, fill=True, fill_opacity=0.6,
        tooltip=f"#{nid}  {n.kind}  score={sc:.2f}",
    ).add_to(m)

# Show disabled nodes in grey
for nid in disabled:
    if nid >= len(nodes):
        continue
    n = nodes[nid]
    folium.CircleMarker(
        location=to_latlon(n.y, n.x),
        radius=5, color="black", fill=True, fill_opacity=0.9,
        tooltip=f"#{nid}  DISABLED",
    ).add_to(m)

# Map click → disable nearest node
map_data = st_folium(m, height=500, width=None,
                     returned_objects=["last_clicked"])
if map_data and map_data.get("last_clicked"):
    click_lat = map_data["last_clicked"]["lat"]
    click_lng = map_data["last_clicked"]["lng"]
    # Find the nearest node in our pixel coords
    best, best_d = None, 1e9
    for nid, n in enumerate(nodes):
        if nid in disabled:
            continue
        lat, lon = to_latlon(n.y, n.x)
        d = (lat - click_lat) ** 2 + (lon - click_lng) ** 2
        if d < best_d:
            best, best_d = nid, d
    if best is not None and best_d < 1e-9:
        st.session_state["disabled"].add(best)
        st.rerun()

# --------------------------------------------------------------------------- #
# Resilience curve
# --------------------------------------------------------------------------- #
st.subheader("📉  Resilience curve (sequential node ablation)")
if full_curve:
    import pandas as pd
    df = pd.DataFrame(full_curve)
    st.line_chart(df.set_index("step")["resilience_index"], height=250)

# --------------------------------------------------------------------------- #
# Apply disabled set → recompute efficiency
# --------------------------------------------------------------------------- #
if st.session_state["disabled"]:
    Gp = G.copy()
    for nid in st.session_state["disabled"]:
        if nid in Gp:
            Gp.remove_node(nid)
    eff_p = global_efficiency(Gp)
    ri = (eff_p / base_eff) if base_eff else 0.0
    st.warning(
        f"**{len(st.session_state['disabled'])}** node(s) disabled by you -> "
        f"efficiency **{eff_p:.4f}** (resilience index = **{ri:.2f}**).  "
        f"Network split into **{__import__('networkx').number_connected_components(Gp)}** "
        f"connected component(s)."
    )
    if st.button("Reset disabled set"):
        st.session_state["disabled"] = set()
        st.rerun()
else:
    eff_p = base_eff

# --------------------------------------------------------------------------- #
# Top gatekeepers
# --------------------------------------------------------------------------- #
st.subheader("🚧  Top gatekeeper nodes (highest betweenness)")
gk = top_gatekeepers(ordered, nodes, k=10)
gdf = [{"id": g["id"], "kind": g["kind"], "y": g["y"], "x": g["x"],
        "score": round(nscore.get(g["id"], 0.0), 3)} for g in gk]
st.table(gdf)

# --------------------------------------------------------------------------- #
# JSON download
# --------------------------------------------------------------------------- #
report = {
    "tile": tile,
    "nodes": len(nodes),
    "edges": len(edges),
    "bridges_added": heal_info["added"],
    "lcc": heal_info["lcc_after"],
    "baseline_efficiency": base_eff,
    "current_efficiency": eff_p if st.session_state["disabled"] else base_eff,
    "resilience_index": (eff_p / base_eff) if (base_eff and st.session_state["disabled"]) else 1.0,
    "disabled": sorted(st.session_state["disabled"]),
    "top_gatekeepers": gdf,
}
st.download_button("💾 Download JSON report", data=json.dumps(report, indent=2),
                   file_name=f"{tile}_resilience.json", mime="application/json")
