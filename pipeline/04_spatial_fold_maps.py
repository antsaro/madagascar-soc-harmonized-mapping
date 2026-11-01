# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3b — MAPS OF THE SPATIAL BLOCK FOLDS ACTUALLY USED (from soc_spatial_cv)
#  Fig. S5 : first grid position per target + test→train distance curves
#  Fig. S6 : all grid positions used as repeats
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point, Polygon as ShPoly
from shapely.ops import unary_union

import nature_style as NS
import soc_config as C
import soc_ml as M
from soc_spatial import ORIGIN, block_keys, grid_shifts, test_train_dist

FONT = NS.setup_arial(verbose=False)
plt.rcParams.update({'axes.spines.top': True, 'axes.spines.right': True})
GEOJSON, FIG_DPI, WORK_DIR = C.GEOJSON, 600, C.FIG_DIR
os.makedirs(WORK_DIR, exist_ok=True)
TARGETS = [('SOC 0–30 cm', 'SOC30'), ('SOC 30–100 cm', 'DeepSOC'), ('SOC 0–100 cm', 'SOC100')]
TARGETS = [(lab, None, key, C.SPATIAL_DESIGN[key]) for lab, key in TARGETS]

FOLD_COLS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
             "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#8C8C8C"]
LINE_COLS = {"map": "#000000", "cv": "#4E79A7", "blk": "#1B9E77"}

# land polygon in km and a 5 km grid of map cells
to_utm = Transformer.from_crs(4326, 32738, always_xy=True)
land = unary_union([shape(f["geometry"]) for f in json.load(open(GEOJSON))["features"]])
try:
    import shapely
    land_km = shapely.transform(land, lambda c: np.column_stack(to_utm.transform(c[:, 0], c[:, 1])) / 1000)
    inside_fn = lambda x, y: shapely.contains_xy(land_km, x, y)
except AttributeError:
    from shapely.ops import transform as sh_transform
    from shapely.prepared import prep
    land_km = sh_transform(lambda x, y, z=None: tuple(np.asarray(v) / 1000 for v in to_utm.transform(x, y)), land)
    inside_fn = lambda x, y: np.array([prep(land_km).contains(Point(a, b)) for a, b in zip(x, y)])

minx, miny, maxx, maxy = land_km.bounds
gx, gy = np.meshgrid(np.arange(np.floor(minx / 5) * 5 + 2.5, maxx, 5),
                     np.arange(np.floor(miny / 5) * 5 + 2.5, maxy, 5))
gx, gy = gx.ravel(), gy.ravel()
map_cells = np.c_[gx, gy][inside_fn(gx, gy)]


def block_polys(xy, fold, shape_, size, shift):
    keys = block_keys(xy, shape_, size, shift)
    o = ORIGIN + np.asarray(shift)
    out = []
    for a, b in np.unique(keys, axis=0):
        f = int(np.bincount(fold[(keys[:, 0] == a) & (keys[:, 1] == b)]).argmax())
        if shape_ == "square":
            x0, y0 = o + np.array([a, b]) * size
            poly = ShPoly([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])
        else:
            r = size / np.sqrt(3)
            cx, cy = o[0] + r * np.sqrt(3) * (a + b / 2), o[1] + r * 1.5 * b
            ang = np.radians(30 + 60 * np.arange(6))
            poly = ShPoly(np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)])
        g = poly.intersection(land_km)
        if not g.is_empty:
            out.append((g, f))
    return out, len(np.unique(keys, axis=0))


def patches(g, **kw):
    parts = g.geoms if hasattr(g, "geoms") else [g]
    return [Polygon(np.asarray(p.exterior.coords), closed=True, **kw) for p in parts if isinstance(p, ShPoly)]


def draw_map(ax, P, rep, point_size):
    ax.set_xlim(minx - 8, maxx + 8)
    ax.set_ylim(miny - 8, maxy + 8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#333333")
    for p in patches(land_km, fc="#EFEFEF", ec="#8c8c8c", lw=0.25):
        ax.add_patch(p)
    for g, f in rep["polys"]:
        for p in patches(g, fc=FOLD_COLS[f - 1], ec="#404040", lw=0.3, alpha=0.30):
            ax.add_patch(p)
    order = np.random.default_rng(1).permutation(len(P))
    ax.scatter(P[order, 0], P[order, 1], c=[FOLD_COLS[f - 1] for f in rep["fold"][order]],
               s=point_size, linewidths=0.12, edgecolors="#262626", zorder=3)
    ax.plot([maxx - 196, maxx + 4], [miny + 22] * 2, "k-", lw=1.6, solid_capstyle="butt", zorder=5)
    ax.text(maxx - 96, miny + 30, "200 km", ha="center", va="bottom", fontsize=6.5, zorder=5)


def fold_legend(fig, height):
    ax = fig.add_axes([0, 0, 1, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.72, "Test fold (spatial blocks)", ha="center", va="center", fontsize=10, fontweight="bold")
    xs = 0.5 + (np.arange(10) - 4.5) * 0.072
    ax.scatter(xs, [0.32] * 10, c=FOLD_COLS, s=50, edgecolor="#262626", linewidth=0.4)
    for i, x in enumerate(xs):
        ax.text(x + 0.013, 0.32, str(i + 1), va="center", fontsize=9)


# read back the folds that were used in the models
fig_data = []
for label, _, response, (shape_, size) in TARGETS:
    path = os.path.join(M.results_dir(response, "spatial"), "spatial_folds.csv")
    if not os.path.exists(path):
        continue
    t = pd.read_csv(path)
    P = t[["x_km", "y_km"]].values
    rep_cols = sorted([c for c in t.columns if c.startswith("fold_rep")], key=lambda c: int(c[8:]))
    random_folds = np.random.default_rng(1).permutation(np.resize(np.arange(1, 11), len(P)))
    d = dict(label=label, shape=shape_, size=size, P=P,
             map_dist=cKDTree(P).query(map_cells)[0], random_dist=test_train_dist(P, random_folds), reps=[])
    for i, c in enumerate(rep_cols):
        fold = t[c].values.astype(int)
        shift = grid_shifts(size)[i]
        polys, n_blocks = block_polys(P, fold, shape_, size, shift)
        d["reps"].append(dict(fold=fold, polys=polys, n_blocks=n_blocks, shift=shift, dist=test_train_dist(P, fold)))
    fig_data.append(d)

letters = "abcdefghijkl"

# figure 1: first grid position for each target, with the distance curves
nc = len(fig_data)
fig = plt.figure(figsize=(7.2, 8.6))
w = 0.96 / nc - 0.02
for j, d in enumerate(fig_data):
    rep, x0 = d["reps"][0], 0.02 + j * 0.96 / nc
    fig.text(x0 + w / 2, 0.975, d["label"], ha="center", va="center", fontsize=10.5, fontweight="bold")
    fig.text(x0 + w / 2, 0.952, f"n = {len(d['P']):,} plots", ha="center", va="center", fontsize=7.5, color="#4d4d4d")
    fig.text(x0, 0.915, letters[j], ha="left", va="center", fontsize=19, fontweight="bold")
    fig.text(x0 + 0.06, 0.925, f"Median dist.: {np.median(rep['dist']):.1f} km", ha="left", va="center", fontsize=7)
    fig.text(x0 + 0.06, 0.905, f"{d['shape']}s {d['size']:g} km | {rep['n_blocks']} blocks",
             ha="left", va="center", fontsize=6.6, color="#4d4d4d")
    draw_map(fig.add_axes([x0, 0.37, w, 0.52]), d["P"], rep, 2.0 if len(d["P"]) > 3000 else 3.0)

    fig.text(x0, 0.315, letters[nc + j], ha="left", va="center", fontsize=19, fontweight="bold")
    fig.text(x0 + 0.06, 0.315, f"Map → plot: {np.median(d['map_dist']):.0f} km", ha="left", va="center", fontsize=7)
    ax = fig.add_axes([x0 + 0.055, 0.115, w - 0.07, 0.175])
    ax.set_xscale("log")
    ax.set_xlim(0.01, 600)
    ax.set_ylim(0, 1)
    for key, dist, lw, lab in [("map", d["map_dist"], 2.0, "Map cells"),
                               ("cv", d["random_dist"], 1.2, "Random CV"),
                               ("blk", rep["dist"], 1.6, "Spatial blocks")]:
        v = np.sort(np.maximum(dist, 0.01))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=LINE_COLS[key], lw=lw, label=lab, drawstyle="steps-post")
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"], fontsize=7)
    ax.tick_params(axis="y", labelsize=7, length=2.5)
    ax.tick_params(axis="x", length=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("km (log scale)", fontsize=7.5, labelpad=2)
    if j == 0:
        ax.set_ylabel("Cumulative share", fontsize=7.5, labelpad=2)
    ax.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.9, edgecolor="none",
              handlelength=1.3, labelspacing=0.15, borderaxespad=0.1, borderpad=0.25)
fig.text(0.5, 0.345, "Distance from test plots to the nearest training plot",
         ha="center", va="center", fontsize=9, fontweight="bold")
fold_legend(fig, 0.065)
NS.save(fig, os.path.join(WORK_DIR, "FigS5_spatialCV_blocks_used"))
plt.show()

# figure 2: all grid positions used as repeats
ncol = max(len(d["reps"]) for d in fig_data)
if ncol > 1:
    wm, hm = 1.55, 1.55 * 1.95
    W, H = 0.45 + ncol * (wm + 0.08), 0.35 + len(fig_data) * (hm + 0.45) + 0.7
    fig = plt.figure(figsize=(W, H))
    k = 0
    for r, d in enumerate(fig_data):
        yb = H - 0.35 - (r + 1) * (hm + 0.45)
        fig.text(0.2 / W, (yb + hm / 2) / H, d["label"], rotation=90, ha="center", va="center",
                 fontsize=10.5, fontweight="bold")
        for c, rep in enumerate(d["reps"]):
            xb = 0.45 + c * (wm + 0.08)
            if r == 0:
                fig.text((xb + wm / 2) / W, (H - 0.18) / H, f"Repeat {c + 1}", ha="center", va="center",
                         fontsize=10.5, fontweight="bold")
            fig.text(xb / W, (yb + hm + 0.22) / H, letters[k] if k < len(letters) else "",
                     ha="left", va="center", fontsize=15, fontweight="bold")
            fig.text((xb + 0.28) / W, (yb + hm + 0.28) / H,
                     f"shift {rep['shift'][0]:g}, {rep['shift'][1]:g} km | {rep['n_blocks']} blocks",
                     ha="left", va="center", fontsize=6.3)
            fig.text((xb + 0.28) / W, (yb + hm + 0.12) / H, f"Median dist.: {np.median(rep['dist']):.1f} km",
                     ha="left", va="center", fontsize=6.3, color="#4d4d4d")
            draw_map(fig.add_axes([xb / W, yb / H, wm / W, hm / H]), d["P"], rep,
                     1.2 if len(d["P"]) > 3000 else 1.8)
            k += 1
    fold_legend(fig, 0.6 / H)
    NS.save(fig, os.path.join(WORK_DIR, "FigS6_spatialCV_all_grid_positions"))
    plt.show()