#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════════
#  map_soc_gpu.py — bootstrap ensemble mapping on GPU (mean, SD, CV rasters)
#
#  python map_soc_gpu.py --target SOC100 --model RandomForest [--n-boot 100]
#
#  Phase 1 (GPU): B bootstrap replicates of the training table. For DeepSOC /
#    SOC100 the Jobbágy β are RE-ESTIMATED on each bootstrap sample, so the
#    uncertainty of the depth harmonisation propagates into the map SD.
#    RF → cuML RandomForestRegressor   XGBoost → device='cuda'
#  Phase 2 (GPU): tiles → feature matrix → predictions of every replicate,
#    Welford running mean/variance → mean, SD and CV (%) GeoTIFFs + national
#    stock total (Pg C).
#
#  Categorical rasters are mapped raster code → class name → training-encoder
#  index (soc_raster.build_cat_lut).
#  QRF prediction intervals are mapped separately by map_qrf_intervals.py.
# ═══════════════════════════════════════════════════════════════════════════════
import argparse
import gc
import json
import math
import os
import pickle
import time

import numpy as np
import rasterio

import soc_config as C
import soc_harmonize as H
import soc_ml as M
from soc_raster import RasterStack, pixel_area_ha


# ── helpers ───────────────────────────────────────────────────────────────────
def gpu_predict(model, model_name, X, use_gpu):
    if not use_gpu:
        return M.predict(model, X).astype(np.float32)
    import cupy as cp
    out = np.empty(len(X), np.float32)
    for s in range(0, len(X), C.PRED_BATCH):
        Xg = cp.asarray(X[s:s + C.PRED_BATCH])
        if model_name == 'XGBoost':
            p = model.get_booster().inplace_predict(Xg)
        else:
            p = model.predict(Xg)
        out[s:s + C.PRED_BATCH] = cp.asnumpy(cp.asarray(p).ravel())
        del Xg, p
    cp.get_default_memory_pool().free_all_blocks()
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — bootstrap training
# ═══════════════════════════════════════════════════════════════════════════════
def phase1(a, out_dir, use_gpu):
    bundle = __import__('joblib').load(
        os.path.join(M.results_dir(a.target, a.cv), 'models', f'{a.model}.pkl'))
    print(f'   hyper-parameters from {a.cv}-CV tuning: {bundle["best_params"]}')
    params = bundle['best_params']
    df, features, encoders = M.load_harmonized_table()
    X = df[features].values.astype(np.float32)
    n = len(df)
    mdir = os.path.join(out_dir, 'boot_models')
    os.makedirs(mdir, exist_ok=True)
    rng = np.random.RandomState(42)
    y_ref, _ = M.full_target(df, a.target)          # reference for OOB diagnostics
    modes = {c: int(df[f'{c}_enc'].mode()[0]) for c in C.CATEGORICAL_PREDICTORS}
    diag = []
    t0 = time.time()
    for b in range(a.n_boot):
        f = os.path.join(mdir, f'b{b:03d}.pkl')
        bi = rng.randint(0, n, n)                       # always draw → reproducible
        if os.path.exists(f):
            continue
        boot = df.iloc[bi].reset_index(drop=True)
        if a.target == 'SOC30':
            y = boot['SC30'].values
            k_used = {}
        else:
            p_b, _ = H.fit_params(boot, k0=C.SHRINK_K0)    # β re-estimated
            y = H.harmonize(boot, p_b)[C.TARGETS[a.target]].values
            k_used = p_b
        mdl = M.make_model(a.model, params, use_gpu, seed=1000 + b)
        mdl.fit(X[bi], y)
        oob = np.setdiff1d(np.arange(n), np.unique(bi))
        m = M.calc_metrics(y_ref[oob], gpu_predict(mdl, a.model, X[oob], use_gpu))
        diag.append(dict(b=b, oob_r2=m['r2'], oob_rmse=m['rmse'], **{f'k_{g}': v for g, v in k_used.items()}))
        with open(f, 'wb') as fh:
            pickle.dump(mdl, fh)
        del mdl
        gc.collect()
        print(f'   boot {b + 1:3d}/{a.n_boot}  OOB R²={m["r2"]:.3f}  '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
    if diag:
        import pandas as pd
        pd.DataFrame(diag).to_csv(os.path.join(out_dir, 'bootstrap_diagnostics.csv'),
                                  index=False, mode='a',
                                  header=not os.path.exists(
                                      os.path.join(out_dir, 'bootstrap_diagnostics.csv')))
    return features, encoders, modes


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — tiled GPU prediction (batch-outer / tile-inner, Welford)
# ═══════════════════════════════════════════════════════════════════════════════
def phase2(a, out_dir, features, encoders, modes, use_gpu):
    stack = RasterStack(encoders, modes, tile=C.TILE_SIZE)
    tiles = stack.tiles
    print(f'   grid {stack.H}×{stack.W}, {len(tiles)} tiles, CRS {stack.crs}')
    build_X = stack.read
    prof = stack.profile

    mdir = os.path.join(out_dir, 'boot_models')
    files = sorted(os.listdir(mdir))[:a.n_boot]
    batch = a.batch_models or (len(files) if a.model == 'XGBoost' else 10)
    n_batches = math.ceil(len(files) / batch)
    acc = {}                                           # tile → [mean, m2, k, flat, win]
    t0 = time.time()
    for bi in range(n_batches):
        fb = files[bi * batch:(bi + 1) * batch]
        models = [pickle.load(open(os.path.join(mdir, f), 'rb')) for f in fb]
        for ti, (r0, c0) in enumerate(tiles):
            X, flat, win = build_X(r0, c0)
            if X is None:
                continue
            if ti not in acc:
                acc[ti] = [np.zeros(len(X), np.float32), np.zeros(len(X), np.float32),
                           0, flat, win]
            mean, m2, k = acc[ti][0], acc[ti][1], acc[ti][2]
            for mdl in models:
                p = np.maximum(gpu_predict(mdl, a.model, X, use_gpu), 0)
                k += 1
                d = p - mean
                mean += d / k
                m2 += d * (p - mean)
            acc[ti][2] = k
            del X
        del models
        gc.collect()
        el = (time.time() - t0) / 60
        print(f'   batch {bi + 1}/{n_batches} done | {el:.1f} min | '
              f'ETA {el / (bi + 1) * (n_batches - bi - 1):.1f} min', flush=True)

    paths = {s: os.path.join(out_dir, f'{a.target}_{a.model}_{s}.tif')
             for s in ('mean', 'sd', 'cv')}
    total_pg, total_sd_pg, area_tot = 0.0, 0.0, 0.0
    with rasterio.open(paths['mean'], 'w', **prof) as fm, \
            rasterio.open(paths['sd'], 'w', **prof) as fs, \
            rasterio.open(paths['cv'], 'w', **prof) as fc:
        for ti, (r0, c0) in enumerate(tiles):
            win = stack.window(r0, c0)
            mt = np.full((win.height, win.width), C.NODATA_OUT, np.float32)
            st, ct = mt.copy(), mt.copy()
            if ti in acc:
                mean, m2, k, flat, _ = acc[ti]
                sd = np.sqrt(np.maximum(m2 / max(k - 1, 1), 0))
                rr, cc = np.unravel_index(flat, (win.height, win.width))
                mt[rr, cc], st[rr, cc] = mean, sd
                ct[rr, cc] = np.where(mean > 0, 100 * sd / np.maximum(mean, 1e-6), C.NODATA_OUT)
                area = pixel_area_ha(stack.transform, stack.crs, r0 + rr)
                total_pg += float(np.sum(mean * area)) / 1e9
                total_sd_pg += float(np.sum(sd * area)) / 1e9   # upper bound (full correlation)
                area_tot += float(area.sum())
            fm.write(mt, 1, window=win)
            fs.write(st, 1, window=win)
            fc.write(ct, 1, window=win)
    stack.close()
    stats = dict(target=a.target, model=a.model, n_models=len(files),
                 total_stock_PgC=total_pg, total_sd_PgC_upper=total_sd_pg,
                 mapped_area_Mha=area_tot / 1e6,
                 mean_density_MgC_ha=total_pg * 1e9 / max(area_tot, 1),
                 minutes=(time.time() - t0) / 60)
    json.dump(stats, open(os.path.join(out_dir, 'map_stats.json'), 'w'), indent=2)
    print(f'   ✅ {paths["mean"]}\n   total stock = {total_pg:.3f} Pg C '
          f'over {area_tot / 1e6:.2f} Mha', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    ap.add_argument('--model', required=True, choices=['RandomForest', 'XGBoost'])
    ap.add_argument('--n-boot', type=int, default=C.N_BOOT_MAP)
    ap.add_argument('--batch-models', type=int, default=0,
                    help='models held on the GPU at once (0 = auto)')
    ap.add_argument('--cv', default=C.FINAL_MODEL_CV, choices=['random', 'spatial'],
                    help='which tuned final model to bootstrap')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    use_gpu = (not a.cpu) and M.gpu_available()
    out_dir = os.path.join(C.MAP_DIR, f'{a.target}_{a.model}')
    os.makedirs(out_dir, exist_ok=True)
    print('=' * 70)
    print(f' MAP {a.model} | {a.target} | B={a.n_boot} | GPU={use_gpu}')
    print('=' * 70, flush=True)
    features, encoders, modes = phase1(a, out_dir, use_gpu)
    phase2(a, out_dir, features, encoders, modes, use_gpu)


if __name__ == '__main__':
    main()
