#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════════
#  map_qrf_intervals.py — QRF prediction-interval rasters (CPU)
#
#  python map_qrf_intervals.py --target SOC100 [--decim 1] [--cv random]
#
#  Uses the final QRF tuned in the nested CV (results_<target>[_spatialCV]/models/
#  QRF.pkl) and predicts, tile by tile, for every valid pixel:
#     mean      conditional mean (point map)
#     q05, q50, q95   conditional quantiles (C.QRF_PI gives the interval bounds)
#     piw90     q95 − q05             (width of the 90 % prediction interval)
#     rpiw90    100 · piw90 / q50 (%) (relative width)
#  plus national totals. Pixels are masked exactly like in map_soc_gpu.py.
# ═══════════════════════════════════════════════════════════════════════════════
import argparse
import json
import os
import time

import joblib
import numpy as np
import rasterio

import soc_config as C
import soc_ml as M
from soc_raster import RasterStack, pixel_area_ha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    ap.add_argument('--cv', default=C.FINAL_MODEL_CV, choices=['random', 'spatial'])
    ap.add_argument('--decim', type=int, default=C.QRF_MAP_DECIM,
                    help='predict on a grid N times coarser (1 = native)')
    a = ap.parse_args()

    lo_q, hi_q = C.QRF_PI
    level = int(round(100 * (hi_q - lo_q)))
    names = ['mean', f'q{100 * lo_q:02.0f}', 'q50', f'q{100 * hi_q:02.0f}',
             f'piw{level}', f'rpiw{level}']
    bundle = joblib.load(os.path.join(M.results_dir(a.target, a.cv), 'models', 'QRF.pkl'))
    model, encoders = bundle['model'], bundle['encoders']
    df, _, _ = M.load_harmonized_table()
    modes = {c: int(df[f'{c}_enc'].mode()[0]) for c in C.CATEGORICAL_PREDICTORS}

    out_dir = os.path.join(C.MAP_DIR, f'{a.target}_QRF')
    os.makedirs(out_dir, exist_ok=True)
    print('=' * 70)
    print(f' QRF prediction intervals | {a.target} | {level} % PI [{lo_q}, {hi_q}] | '
          f'decim={a.decim} | params from {a.cv} CV')
    print('=' * 70, flush=True)

    stack = RasterStack(encoders, modes, tile=C.QRF_MAP_TILE, decim=a.decim)
    print(f'   output grid {stack.H}×{stack.W}, {len(stack.tiles)} tiles', flush=True)
    paths = {n: os.path.join(out_dir, f'{a.target}_QRF_{n}.tif') for n in names}
    dsts = {n: rasterio.open(p, 'w', **stack.profile) for n, p in paths.items()}
    tot = dict(mean=0.0, lo=0.0, hi=0.0, area=0.0, n=0)
    t0 = time.time()
    for ti, (r0, c0) in enumerate(stack.tiles):
        X, flat, win = stack.read(r0, c0)
        th, tw = int(win.height), int(win.width)
        layers = {n: np.full((th, tw), C.NODATA_OUT, np.float32) for n in names}
        if X is not None:
            mean = np.empty(len(X), np.float32)
            q = np.empty((len(X), 3), np.float32)
            for s in range(0, len(X), C.QRF_MAP_BATCH):
                xb = X[s:s + C.QRF_MAP_BATCH]
                mean[s:s + len(xb)] = M.predict(model, xb)
                q[s:s + len(xb)] = M.predict_quantiles(model, xb, [lo_q, 0.5, hi_q])
            mean, q = np.maximum(mean, 0), np.maximum(q, 0)
            width = q[:, 2] - q[:, 0]
            rel = np.where(q[:, 1] > 0, 100 * width / np.maximum(q[:, 1], 1e-6), C.NODATA_OUT)
            rr, cc = np.unravel_index(flat, (th, tw))
            for n, v in zip(names, [mean, q[:, 0], q[:, 1], q[:, 2], width, rel]):
                layers[n][rr, cc] = v
            area = pixel_area_ha(stack.transform, stack.crs, r0 + rr)
            tot['mean'] += float(np.sum(mean * area)) / 1e9
            tot['lo'] += float(np.sum(q[:, 0] * area)) / 1e9
            tot['hi'] += float(np.sum(q[:, 2] * area)) / 1e9
            tot['area'] += float(area.sum())
            tot['n'] += len(X)
        for n in names:
            dsts[n].write(layers[n], 1, window=win)
        el = (time.time() - t0) / 60
        print(f'   tile {ti + 1}/{len(stack.tiles)}  valid={0 if X is None else len(X):,}  '
              f'{el:.1f} min, ETA {el / (ti + 1) * (len(stack.tiles) - ti - 1):.1f} min',
              flush=True)
    for d in dsts.values():
        d.close()
    stack.close()
    stats = dict(target=a.target, model='QRF', cv_params=a.cv, decim=a.decim,
                 interval=[lo_q, hi_q], n_pixels=tot['n'],
                 total_stock_PgC=tot['mean'],
                 total_stock_PgC_sum_of_lower=tot['lo'],
                 total_stock_PgC_sum_of_upper=tot['hi'],
                 note='sums of pixel quantiles bound the total; they are not '
                      'quantiles of the national total',
                 mapped_area_Mha=tot['area'] / 1e6,
                 minutes=(time.time() - t0) / 60)
    json.dump(stats, open(os.path.join(out_dir, 'map_stats.json'), 'w'), indent=2)
    print(f'   ✅ {out_dir}\n   total stock (mean) = {tot["mean"]:.3f} Pg C '
          f'[{tot["lo"]:.3f}, {tot["hi"]:.3f}] over {tot["area"] / 1e6:.2f} Mha', flush=True)


if __name__ == '__main__':
    main()
