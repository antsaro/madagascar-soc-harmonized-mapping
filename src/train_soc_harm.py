#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════════
#  train_soc_harm.py — repeated nested CV (GPU) on the depth-harmonised table
#
#  python train_soc_harm.py --target SOC100 --model XGBoost --cv random
#  python train_soc_harm.py --target SOC100 --model XGBoost --cv spatial
#  (the launcher sets CUDA_VISIBLE_DEVICES so each process sees one GPU)
#
#  --cv random  : outer 10 × 10 KFold, inner 4-fold KFold (Optuna TPE + pruning)
#  --cv spatial : outer 10 folds of hexagonal blocks × 4 grid positions,
#                 inner 4 spatial-block folds built from the outer-training plots
#                 only (same design as soc_spatial_cv.ipynb)
#  • For DeepSOC / SOC100 the Jobbágy β are re-fitted on every outer-training
#    fold, so held-out profiles never inform their own harmonised target.
#  • Validation is reported on (i) measured 0–100 cm profiles only (primary,
#    independent) and (ii) all held-out points (measured + harmonised).
#  • --model QRF (quantile regression forest, CPU): the conditional quantiles of
#    every held-out point are also stored, and the coverage (PICP) and mean width
#    (MPIW) of the 50/80/90/95 % central prediction intervals are reported.
# ═══════════════════════════════════════════════════════════════════════════════
import argparse
import json
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold

import soc_config as C
import soc_ml as M
import soc_spatial as S

optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner_search(model_name, X, y, splits, n_trials, use_gpu, seed=42):
    """Optuna search; `splits` = list of (train, val) index arrays into X."""
    def objective(trial):
        params = M.suggest_params(trial, model_name)
        scores = []
        for i, (tr, va) in enumerate(splits):
            mdl = M.make_model(model_name, params, use_gpu)
            mdl.fit(X[tr], y[tr])
            p = M.predict(mdl, X[va])
            scores.append(float(np.sqrt(np.mean((y[va] - p) ** 2))))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def random_kfold(n, k, seed):
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))


def outer_plan(a, df, xy):
    """Yields (repeat, fold, train_idx, test_idx, inner_splits_factory, info)."""
    n = len(df)
    if a.cv == 'random':
        for rep in range(a.repeats):
            for fold, (tr, te) in enumerate(random_kfold(n, a.outer_folds, rep * 100 + 42), 1):
                yield rep + 1, fold, tr, te, (lambda tr_=tr: random_kfold(len(tr_), a.inner_folds, 0)), {}
        return
    shape_, size = C.SPATIAL_DESIGN[a.target]
    w = None
    if a.target != 'SOC30':                    # balance folds on measured 1 m profiles
        w = np.where(df['measured100'].values, 1.0, C.SPATIAL_HARMONIZED_WEIGHT)
    for rep, shift in enumerate(S.grid_shifts(size)[:a.spatial_repeats], 1):
        outer, folds = S.spatial_splits(xy, shape_, size, a.outer_folds, shift, w)
        sizes = np.bincount(folds['ids'])[1:]
        n_meas = [int(df['measured100'].values[te].sum()) for _, te in outer]
        print(f'  repeat {rep}: shift {shift} km, block {folds["size"]:.0f} km, '
              f'{folds["n_blocks"]} blocks, fold sizes {sizes.min()}–{sizes.max()}, '
              f'measured-1m per fold {min(n_meas)}–{max(n_meas)}', flush=True)
        yield ('folds', rep, folds['ids'])
        for fold, (tr, te) in enumerate(outer, 1):
            def inner(tr_=tr, shift_=shift):
                sp, inf = S.spatial_splits(xy[tr_], shape_, size, a.inner_folds, shift_,
                                           None if w is None else w[tr_])
                return sp
            yield rep, fold, tr, te, inner, dict(block_km=folds['size'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    ap.add_argument('--model', required=True, choices=C.MODELS)
    ap.add_argument('--cv', default='random', choices=['random', 'spatial'])
    ap.add_argument('--outer-folds', type=int, default=C.N_OUTER_FOLDS)
    ap.add_argument('--repeats', type=int, default=C.N_OUTER_REPEATS)
    ap.add_argument('--spatial-repeats', type=int, default=C.N_SPATIAL_REPEATS)
    ap.add_argument('--inner-folds', type=int, default=C.N_INNER_FOLDS)
    ap.add_argument('--trials', type=int, default=None,
                    help=f'Optuna trials (default {C.OPTUNA_TRIALS}; QRF {C.QRF_OPTUNA_TRIALS})')
    ap.add_argument('--cpu', action='store_true', help='force CPU (testing)')
    a = ap.parse_args()

    if a.trials is None:
        a.trials = C.QRF_OPTUNA_TRIALS if a.model == 'QRF' else C.OPTUNA_TRIALS
    use_gpu = (not a.cpu) and M.gpu_available() and a.model != 'QRF'
    is_qrf = a.model == 'QRF'
    QS = C.QRF_QUANTILES
    PIS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}
    out_dir = M.results_dir(a.target, a.cv)
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    tag = f'{a.model}|{a.target}|{a.cv}'
    n_rep = a.repeats if a.cv == 'random' else a.spatial_repeats
    print('=' * 78)
    print(f' {a.model} | {a.target} | {a.cv.upper()} CV | GPU={use_gpu} | '
          f'outer {a.outer_folds}×{n_rep}, inner {a.inner_folds}, {a.trials} trials')
    if a.cv == 'spatial':
        print(f' blocks: {C.SPATIAL_DESIGN[a.target][0]}s {C.SPATIAL_DESIGN[a.target][1]} km')
    print('=' * 78, flush=True)

    df, features, encoders = M.load_harmonized_table()
    X = df[features].values.astype(np.float32)
    n = len(df)
    xy = S.lonlat_to_km(df[C.LON_COL].values, df[C.LAT_COL].values)
    src = df['SOC100_src'].values if a.target != 'SOC30' else np.array(['measured'] * n)
    print(f' n = {n}  | measured 0–100 cm = {int(df.measured100.sum())}', flush=True)

    pred_rows, fold_rows, param_rows, fold_table = [], [], [], {}
    t0 = time.time()
    run, total = 0, a.outer_folds * n_rep
    for item in outer_plan(a, df, xy):
        if item[0] == 'folds':
            fold_table[f'fold_rep{item[1]}'] = item[2]
            continue
        rep, fold, tr, te, inner_factory, info = item
        run += 1
        y_tr, y_te_all, y_te_meas, _ = M.targets_for_fold(df, a.target, tr, te)
        best, best_rmse = inner_search(a.model, X[tr], y_tr, inner_factory(),
                                       a.trials, use_gpu, seed=42 + run)
        mdl = M.make_model(a.model, best, use_gpu, seed=42 + run)
        mdl.fit(X[tr], y_tr)
        p = M.predict(mdl, X[te])
        dist = cKDTree(xy[tr]).query(xy[te])[0]

        m_meas = M.calc_metrics(y_te_meas, p)
        m_all = M.calc_metrics(y_te_all, p)
        qcols, pi_stats = {}, {}
        if is_qrf:
            qp = M.predict_quantiles(mdl, X[te], QS)
            qcols = {f'q{q:.3f}': qp[:, i] for i, q in enumerate(QS)}
            lo, hi = qp[:, QS.index(0.05)], qp[:, QS.index(0.95)]
            pi_stats['meas_picp90'], pi_stats['meas_mpiw90'] = M.interval_scores(y_te_meas, lo, hi)
            pi_stats['all_picp90'], pi_stats['all_mpiw90'] = M.interval_scores(y_te_all, lo, hi)
        fold_rows.append(dict(repeat=rep, fold=fold, n_test=len(te), **pi_stats,
                              median_test_train_km=float(np.median(dist)),
                              **{f'meas_{k}': v for k, v in m_meas.items()},
                              **{f'all_{k}': v for k, v in m_all.items()}))
        param_rows.append(dict(repeat=rep, fold=fold, inner_rmse=best_rmse, **info, **best))
        pred_rows.append(pd.DataFrame(dict(
            row_id=te, repeat=rep, fold=fold, predicted=p,
            observed_all=y_te_all, observed_meas=y_te_meas, source=src[te],
            HGROUP=df['HGROUP'].values[te], x_km=xy[te, 0], y_km=xy[te, 1],
            dist_to_train_km=dist, **qcols)))
        el = (time.time() - t0) / 60
        print(f' [{tag}] fold {run:3d}/{total}  n={len(te):4d}  '
              f'dist={np.median(dist):6.1f} km  R²(meas)={m_meas["r2"]:.3f}  '
              f'R²(all)={m_all["r2"]:.3f}  RMSE(meas)={m_meas["rmse"]:.1f}  | '
              f'{el:.1f} min, ETA {el / run * (total - run):.1f} min', flush=True)

    preds = pd.concat(pred_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    pooled_meas = M.calc_metrics(preds['observed_meas'], preds['predicted'])
    pooled_all = M.calc_metrics(preds['observed_all'], preds['predicted'])

    # QRF: coverage of central prediction intervals (pooled over all outer folds)
    qrf_cov = []
    if is_qrf:
        for lev, (ql, qh) in PIS.items():
            lo, hi = preds[f'q{ql:.3f}'].values, preds[f'q{qh:.3f}'].values
            for scope, col in [('measured', 'observed_meas'), ('all', 'observed_all')]:
                picp, mpiw = M.interval_scores(preds[col].values, lo, hi)
                qrf_cov.append(dict(nominal=lev, scope=scope, picp=100 * picp, mpiw=mpiw))
        pd.DataFrame(qrf_cov).to_csv(os.path.join(out_dir, 'qrf_interval_coverage.csv'),
                                     index=False)
        print(pd.DataFrame(qrf_cov).round(2).to_string(index=False))

    # OOF mean per sample (averaged over repeats) → used for scatter plots
    oof = (preds.groupby('row_id')
           .agg(predicted=('predicted', 'mean'), pred_sd=('predicted', 'std'),
                observed_all=('observed_all', 'mean'),
                observed_meas=('observed_meas', 'first'),
                source=('source', 'first'), HGROUP=('HGROUP', 'first'),
                dist_to_train_km=('dist_to_train_km', 'median'))
           .reset_index())

    # ── final model on all data (tuned with the same CV scheme) ───────────────
    y_full, harm_params = M.full_target(df, a.target)
    if a.cv == 'random':
        full_splits = random_kfold(n, a.inner_folds, 0)
    else:
        w = None if a.target == 'SOC30' else np.where(
            df['measured100'].values, 1.0, C.SPATIAL_HARMONIZED_WEIGHT)
        full_splits, _ = S.spatial_splits(xy, *C.SPATIAL_DESIGN[a.target],
                                          a.inner_folds, weights=w)
    best, best_rmse = inner_search(a.model, X, y_full, full_splits, a.trials, use_gpu)
    final = M.make_model(a.model, best, use_gpu)
    final.fit(X, y_full)

    bundle = dict(model=final, best_params=best, features=features,
                  encoders=encoders, target=a.target, cv=a.cv,
                  target_col=C.TARGETS[a.target], harm_params=harm_params,
                  gpu=use_gpu)
    joblib.dump(bundle, os.path.join(out_dir, 'models', f'{a.model}.pkl'))
    if a.model == 'XGBoost':
        final.save_model(os.path.join(out_dir, 'models', 'XGBoost.ubj'))

    preds.to_csv(os.path.join(out_dir, f'cv_predictions_{a.model}.csv'), index=False)
    oof.to_csv(os.path.join(out_dir, f'oof_mean_{a.model}.csv'), index=False)
    folds.to_csv(os.path.join(out_dir, f'fold_metrics_{a.model}.csv'), index=False)
    pd.DataFrame(param_rows).to_csv(
        os.path.join(out_dir, f'best_params_per_fold_{a.model}.csv'), index=False)
    if fold_table:                              # spatial folds actually used
        ft = pd.concat([df[[C.LON_COL, C.LAT_COL, 'SC30', 'measured100']],
                        pd.DataFrame(xy, columns=['x_km', 'y_km']),
                        pd.DataFrame(fold_table)], axis=1)
        ft.to_csv(os.path.join(out_dir, 'spatial_folds.csv'), index=False)

    def _mean_sd(prefix):
        ks = ('r2', 'rmse', 'mae', 'bias', 'rpd', 'rpiq', 'ccc')
        return {f'{prefix}{k}_mean': float(np.nanmean(folds[f'{prefix}{k}'])) for k in ks} | \
               {f'{prefix}{k}_sd': float(np.nanstd(folds[f'{prefix}{k}'])) for k in ks}

    summary = dict(model=a.model, target=a.target, cv_scheme=a.cv, n=n,
                   n_measured_100=int(df.measured100.sum()),
                   cv=f'{n_rep}x{a.outer_folds} outer, {a.inner_folds} inner, '
                      f'{a.trials} trials', gpu=use_gpu,
                   design=(f'{C.SPATIAL_DESIGN[a.target][0]} '
                           f'{C.SPATIAL_DESIGN[a.target][1]} km' if a.cv == 'spatial' else 'random'),
                   median_test_train_km=float(preds['dist_to_train_km'].median()),
                   pooled_measured=pooled_meas, pooled_all=pooled_all,
                   **_mean_sd('meas_'), **_mean_sd('all_'),
                   final_params={k: (v.item() if isinstance(v, np.generic) else v)
                                 for k, v in best.items()},
                   final_inner_rmse=best_rmse, qrf_coverage=qrf_cov,
                   minutes=(time.time() - t0) / 60)
    with open(os.path.join(out_dir, f'summary_{a.model}.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'\n POOLED (measured): R²={pooled_meas["r2"]:.3f} RMSE={pooled_meas["rmse"]:.2f} '
          f'CCC={pooled_meas["ccc"]:.3f} n={pooled_meas["n"]}')
    print(f' POOLED (all)     : R²={pooled_all["r2"]:.3f} RMSE={pooled_all["rmse"]:.2f} '
          f'CCC={pooled_all["ccc"]:.3f} n={pooled_all["n"]}')
    print(f' median test→train distance: {summary["median_test_train_km"]:.1f} km')
    print(f' ✅ saved → {out_dir}  ({summary["minutes"]:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
