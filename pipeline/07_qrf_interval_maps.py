# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — QRF PREDICTION WITH PREDICTION-INTERVAL RASTERS (90 % PI)
#  For each depth: mean, q05, q50, q95, PI width (q95 − q05) and relative width
#  GeoTIFFs + Fig. 8 (lower bound, median, upper bound, PI width).
#  Runs on CPU (quantile-forest); the three depths run in parallel processes.
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
import os
import subprocess
import sys
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

import nature_style as NS
import soc_config as C

RUN_QRF_MAPS = True                 # False → only re-draw the figure
TARGETS_TO_MAP = ['SOC30', 'DeepSOC', 'SOC100']
DECIM = C.QRF_MAP_DECIM             # e.g. 2 or 4 to predict on a coarser grid
N_PARALLEL = 1                      # processes at once (each uses all cores)

if RUN_QRF_MAPS:
    SRC = os.path.dirname(os.path.abspath(C.__file__))
    os.makedirs(C.MAP_DIR, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = SRC + os.pathsep + env.get('PYTHONPATH', '')
    queue, running = list(TARGETS_TO_MAP), {}
    while queue or running:
        while queue and len(running) < N_PARALLEL:
            t = queue.pop(0)
            log = open(os.path.join(C.MAP_DIR, f'log_qrf_{t}.txt'), 'w')
            p = subprocess.Popen([sys.executable, '-u', f'{SRC}/map_qrf_intervals.py',
                                  '--target', t, '--decim', str(DECIM)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, cwd=SRC)
            running[t] = (p, log)
            print(f'Launched QRF interval map for {t}')
        time.sleep(30)
        for t, (p, log) in list(running.items()):
            lines = [l.strip() for l in open(log.name) if l.strip()]
            print(f'[QRF {t:<8}] {lines[-1] if lines else ""}', flush=True)
            if p.poll() is not None:
                log.close()
                print(f'QRF {t}: exit {p.returncode}')
                running.pop(t)

# ── totals ────────────────────────────────────────────────────────────────────
print('\nQRF national totals (mean map; bounds = sums of pixel quantiles)')
for d in NS.DEPTH_ORDER:
    f = os.path.join(C.MAP_DIR, f'{d}_QRF', 'map_stats.json')
    if os.path.exists(f):
        s = json.load(open(f))
        print(f'  {d:<8} {s["total_stock_PgC"]:7.3f} Pg C   '
              f'[{s["total_stock_PgC_sum_of_lower"]:.3f}, {s["total_stock_PgC_sum_of_upper"]:.3f}]')

# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 8 — QRF: lower bound, median, upper bound, 90 % PI width
#  q05 / q50 / q95 share one colour scale per depth so they can be compared.
# ═══════════════════════════════════════════════════════════════════════════════
FONT = NS.setup_arial(verbose=False)
PLOT_DECIM = 8
lo_q, hi_q = C.QRF_PI
level = int(round(100 * (hi_q - lo_q)))
COLS = [(f'q{100 * lo_q:02.0f}', f'Lower bound (q{100 * lo_q:02.0f})'),
        ('q50', 'Median (q50)'),
        (f'q{100 * hi_q:02.0f}', f'Upper bound (q{100 * hi_q:02.0f})'),
        (f'piw{level}', f'{level}% PI width')]
SOC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'soc', ['#FFF7E3', '#F6D58A', '#D99A3E', '#9C5A1E', '#5A2F12', '#2B1708'])
PIW_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'piw', ['#FCFBFD', '#DADAEB', '#9E9AC8', '#6A51A3', '#3F007D'])


def read_small(path):
    with rasterio.open(path) as src:
        arr = src.read(1, out_shape=(max(1, src.height // PLOT_DECIM), max(1, src.width // PLOT_DECIM)),
                       resampling=Resampling.nearest).astype(float)
        b, nd = src.bounds, src.nodata
    arr[(arr == nd) | ~np.isfinite(arr)] = np.nan
    return arr, (b.left, b.right, b.bottom, b.top)


fig, axs = plt.subplots(3, 4, figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.86),
                        gridspec_kw=dict(wspace=0.04, hspace=0.12))
k = 0
for r, d in enumerate(NS.DEPTH_ORDER):
    paths = {key: os.path.join(C.MAP_DIR, f'{d}_QRF', f'{d}_QRF_{key}.tif') for key, _ in COLS}
    if not all(os.path.exists(p) for p in paths.values()):
        for ax in axs[r]:
            ax.set_axis_off()
        continue
    data = {key: read_small(p) for key, p in paths.items()}
    vmax_soc = np.nanpercentile(data[COLS[2][0]][0], 98)
    for c, (key, title) in enumerate(COLS):
        ax = axs[r, c]
        arr, ext = data[key]
        is_w = key.startswith('piw')
        im = ax.imshow(arr, extent=ext, cmap=PIW_CMAP if is_w else SOC_CMAP, vmin=0,
                       vmax=np.nanpercentile(arr, 98) if is_w else vmax_soc,
                       interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if c in (2, 3):              # one bar for the shared SOC scale, one for width
            cax = ax.inset_axes([0.84, 0.06, 0.035, 0.36])
            cb = fig.colorbar(im, cax=cax, extend='max')
            cb.outline.set_linewidth(0.3)
            cb.ax.tick_params(labelsize=5.2, width=0.3, length=1.5)
            cb.set_label(NS.UNIT, fontsize=5.4, labelpad=1)
            cb.ax.yaxis.set_label_position('left')
        if r == 0:
            ax.set_title(title, loc='center', pad=3)
        if c == 0:
            ax.text(-0.04, 0.5, NS.DEPTH_LABEL[d], transform=ax.transAxes, rotation=90,
                    va='center', ha='right', fontsize=7)
        NS.panel_letter(ax, 'abcdefghijkl'[k], x=0.0, y=0.93)
        k += 1
fig.suptitle(f'Quantile regression forest · {level}% prediction interval',
             x=0.13, y=0.935, ha='left', fontsize=7.5, fontweight='bold')
NS.save(fig, os.path.join(C.FIG_DIR, 'Fig8_QRF_prediction_interval_maps'))
plt.show()
