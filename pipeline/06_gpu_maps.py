# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — GPU MAPPING (launch) + MAP FIGURES
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

RUN_MAPPING = True                       # False → only re-draw figures
TARGETS_TO_MAP = ['SOC30', 'DeepSOC', 'SOC100']
QUEUES = {0: 'RandomForest', 1: 'XGBoost'}
N_BOOT = C.N_BOOT_MAP

# ── 5a. launch: RF on GPU 0, XGBoost on GPU 1 ─────────────────────────────────
if RUN_MAPPING:
    try:
        import cupy as cp
        n_gpu = cp.cuda.runtime.getDeviceCount()
    except Exception:                                            # noqa: BLE001
        n_gpu = 0
    os.makedirs(C.MAP_DIR, exist_ok=True)
    SRC = os.path.dirname(os.path.abspath(C.__file__))
    procs = {}
    for gpu, model in QUEUES.items():
        cmd = ' ; '.join(f'{sys.executable} -u {SRC}/map_soc_gpu.py --target {t} '
                          f'--model {model} --n-boot {N_BOOT}' for t in TARGETS_TO_MAP)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu if n_gpu > 1 else 0)
        env['PYTHONPATH'] = SRC + os.pathsep + env.get('PYTHONPATH', '')
        log = open(os.path.join(C.MAP_DIR, f'log_map_{model}.txt'), 'w')
        procs[model] = (subprocess.Popen(cmd, shell=True, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, cwd=SRC), log)
        print(f'Launched mapping {model} on GPU {env["CUDA_VISIBLE_DEVICES"]}')
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(60)
        for model, (p, log) in procs.items():
            lines = [l.strip() for l in open(log.name) if l.strip()]
            print(f'[{model:<12}] {lines[-1] if lines else ""}', flush=True)
    for model, (p, log) in procs.items():
        log.close()
        print(f'{model}: exit {p.returncode}')

# ── 5b. stock totals + additivity check (SOC30 + SOC30–100 vs SOC0–100) ──────
print('\nNational totals (bootstrap-mean maps)')
for m in QUEUES.values():
    tot = {}
    for d in NS.DEPTH_ORDER:
        f = os.path.join(C.MAP_DIR, f'{d}_{m}', 'map_stats.json')
        if os.path.exists(f):
            tot[d] = json.load(open(f))['total_stock_PgC']
            print(f'  {m:<12} {d:<8} {tot[d]:7.3f} Pg C')
    if len(tot) == 3:
        diff = tot['SOC30'] + tot['DeepSOC'] - tot['SOC100']
        print(f'  {m:<12} SOC30+Deep − SOC100 = {diff:+.3f} Pg C '
              f'({100 * diff / tot["SOC100"]:+.1f} %)')

# ── 5c. Figure — 3 depths × (mean, SD, CV), one per model ────────────────────
FONT = NS.setup_arial(verbose=False)
DECIM = 8                                      # read every 8th pixel for plotting

CMAPS = {
    'mean': mcolors.LinearSegmentedColormap.from_list(
        'soc', ['#FFF7E3', '#F6D58A', '#D99A3E', '#9C5A1E', '#5A2F12', '#2B1708']),
    'sd': mcolors.LinearSegmentedColormap.from_list(
        'sd', ['#F7FBFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B']),
    'cv': mcolors.LinearSegmentedColormap.from_list(
        'cv', ['#FCFBFD', '#DADAEB', '#9E9AC8', '#6A51A3', '#3F007D']),
}
STAT_LABEL = {'mean': f'Mean ({NS.UNIT})', 'sd': f'SD ({NS.UNIT})', 'cv': 'CV (%)'}


def read_small(path):
    with rasterio.open(path) as src:
        arr = src.read(1, out_shape=(src.height // DECIM, src.width // DECIM),
                       resampling=Resampling.nearest).astype(float)
        b = src.bounds
        nd = src.nodata
    arr[(arr == nd) | ~np.isfinite(arr)] = np.nan
    return arr, (b.left, b.right, b.bottom, b.top)


for m in QUEUES.values():
    fig, axs = plt.subplots(3, 3, figsize=(NS.DOUBLE_COL * 0.82, NS.DOUBLE_COL * 1.12),
                            gridspec_kw=dict(wspace=0.05, hspace=0.12))
    k = 0
    for r, d in enumerate(NS.DEPTH_ORDER):
        for c, s in enumerate(['mean', 'sd', 'cv']):
            ax = axs[r, c]
            p = os.path.join(C.MAP_DIR, f'{d}_{m}', f'{d}_{m}_{s}.tif')
            if not os.path.exists(p):
                ax.set_axis_off()
                continue
            arr, ext = read_small(p)
            vmax = np.nanpercentile(arr, 98)
            im = ax.imshow(arr, extent=ext, cmap=CMAPS[s], vmin=0, vmax=vmax,
                           interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            cax = ax.inset_axes([0.84, 0.06, 0.035, 0.36])
            cb = fig.colorbar(im, cax=cax, extend='max')
            cb.outline.set_linewidth(0.3)
            cb.ax.tick_params(labelsize=5.2, width=0.3, length=1.5)
            cb.set_label(STAT_LABEL[s], fontsize=5.4, labelpad=1)
            cb.ax.yaxis.set_label_position('left')
            if r == 0:
                ax.set_title({'mean': 'Mean', 'sd': 'Standard deviation',
                              'cv': 'Coefficient of variation'}[s], loc='center', pad=3)
            if c == 0:
                ax.text(-0.04, 0.5, NS.DEPTH_LABEL[d], transform=ax.transAxes,
                        rotation=90, va='center', ha='right', fontsize=7)
            NS.panel_letter(ax, 'abcdefghi'[k], x=0.0, y=0.93)
            k += 1
    fig.suptitle(f'{NS.MODEL_LABEL[m]} · bootstrap ensemble (B = {N_BOOT})',
                 x=0.13, y=0.915, ha='left', fontsize=7.5, fontweight='bold')
    NS.save(fig, os.path.join(C.FIG_DIR, f'Fig7_maps_{m}'))
    plt.show()
