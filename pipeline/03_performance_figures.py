# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — MODEL SUMMARY TABLE + OBSERVED-vs-PREDICTED SCATTERPLOTS
#  for both CV schemes (random KFold and spatial block CV)
#  Fig. 2a/2b : RF / XGBoost × 3 depths, measured profiles — random / spatial CV
#  Fig. S2a/b : same, all held-out points (measured + harmonised)
#  Fig. 2c    : random vs spatial CV comparison
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import nature_style as NS
import soc_config as C
import soc_ml as M

FONT = NS.setup_arial(verbose=False)
MODELS = C.MODELS
LETTERS = 'abcdefghijklmnopqr'

# ── summary table ─────────────────────────────────────────────────────────────
rows = []
for cvs, d, m in [(cvs, d, m) for cvs in C.CV_SCHEMES for d in NS.DEPTH_ORDER for m in MODELS]:
        p = os.path.join(M.results_dir(d, cvs), f'summary_{m}.json')
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        for scope, key in [('measured', 'meas_'), ('all', 'all_')]:
            if d == 'SOC30' and scope == 'all':
                continue
            pooled = s['pooled_measured' if scope == 'measured' else 'pooled_all']
            rows.append({'CV': cvs, 'Depth': d, 'Model': m, 'Validation': scope,
                         'n': pooled['n'],
                         'Median_test_train_km': round(s.get('median_test_train_km', np.nan), 1),
                         'R2': f"{s[f'{key}r2_mean']:.3f} ± {s[f'{key}r2_sd']:.3f}",
                         'RMSE': f"{s[f'{key}rmse_mean']:.1f} ± {s[f'{key}rmse_sd']:.1f}",
                         'MAE': f"{s[f'{key}mae_mean']:.1f} ± {s[f'{key}mae_sd']:.1f}",
                         'Bias': f"{s[f'{key}bias_mean']:.2f} ± {s[f'{key}bias_sd']:.2f}",
                         'CCC': f"{s[f'{key}ccc_mean']:.3f} ± {s[f'{key}ccc_sd']:.3f}",
                         'RPIQ': f"{s[f'{key}rpiq_mean']:.2f} ± {s[f'{key}rpiq_sd']:.2f}",
                         'Pooled_R2': round(pooled['r2'], 3),
                         'Pooled_RMSE': round(pooled['rmse'], 2)})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(C.RESULTS_DIR, 'Table_model_performance.csv'), index=False)
print(summary.to_string(index=False))


def load_oof(depth, model, cvs):
    return pd.read_csv(os.path.join(M.results_dir(depth, cvs), f'oof_mean_{model}.csv'))


def density(x, y):
    try:
        z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    except Exception:                                            # noqa: BLE001
        z = np.ones_like(x)
    return (z - z.min()) / (np.ptp(z) + 1e-12)


def panel(ax, obs, pred, model, depth, letter, extra=None, cmap='viridis'):
    dz = density(obs, pred)
    o = dz.argsort()
    ax.scatter(obs[o], pred[o], c=dz[o], cmap=cmap, s=2.2,
               lw=0, alpha=0.9, rasterized=True, zorder=2)
    if extra is not None:                         # harmonised points (Fig. S2)
        ax.scatter(extra[0], extra[1], s=1.5, color=NS.GREY, lw=0, alpha=0.35,
                   rasterized=True, zorder=1)
    hi = np.nanpercentile(np.r_[obs, pred], 99.7) * 1.08
    ax.plot([0, hi], [0, hi], color=NS.INK, lw=0.6, ls='--', zorder=3)
    z = np.polyfit(obs, pred, 1)
    xx = np.linspace(0, hi, 50)
    ax.plot(xx, np.polyval(z, xx), color=NS.MODEL_COLOR[model], lw=1.0, zorder=4)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect('equal')
    ax.locator_params(nbins=4)
    mt = (M.calc_metrics(obs, pred) if extra is None else
          M.calc_metrics(np.r_[obs, extra[0]], np.r_[pred, extra[1]]))
    ax.text(0.04, 0.97,
            f"$R^2$ = {mt['r2']:.2f}\nRMSE = {mt['rmse']:.1f}\nMAE = {mt['mae']:.1f}\n"
            f"Bias = {mt['bias']:.1f}\nCCC = {mt['ccc']:.2f}\n$n$ = {mt['n']:,}",
            transform=ax.transAxes, va='top', ha='left', fontsize=5.6,
            linespacing=1.3, zorder=5,
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
    ax.set_title(f'{NS.MODEL_LABEL[model]} · {NS.DEPTH_LABEL[depth]}', fontsize=6.5, pad=3)
    NS.panel_letter(ax, letter, x=-0.30, y=1.02)


def make_figure(scope, cvs, fname):
    nr = len(MODELS)
    fig, axs = plt.subplots(nr, 3, figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.33 * nr),
                            squeeze=False,
                            gridspec_kw=dict(wspace=0.42, hspace=0.38))
    k = 0
    for r, m in enumerate(MODELS):
        for c, d in enumerate(NS.DEPTH_ORDER):
            ax = axs[r, c]
            try:
                oof = load_oof(d, m, cvs)
            except FileNotFoundError:
                ax.set_axis_off()
                continue
            extra = None
            if scope == 'measured' or d == 'SOC30':
                s = oof[oof.observed_meas.notna()]
                obs, pred = s.observed_meas.values, s.predicted.values
            else:
                s = oof[oof.source == 'measured']
                h = oof[oof.source == 'harmonized']
                obs, pred = s.observed_meas.values, s.predicted.values
                extra = (h.observed_all.values, h.predicted.values)
            panel(ax, obs, pred, m, d, LETTERS[k], extra)
            if r == nr - 1:
                ax.set_xlabel(f'Observed ({NS.UNIT})')
            if c == 0:
                ax.set_ylabel(f'Predicted ({NS.UNIT})')
            k += 1
    fig.suptitle({'random': 'Random 10-fold CV (10 repeats)',
                  'spatial': 'Spatial block CV (hexagons, 4 grid positions)'}[cvs],
                 x=0.06, y=0.985, ha='left', fontsize=7.5, fontweight='bold')
    if scope == 'all':
        fig.legend(handles=[
            Line2D([], [], marker='o', ls='', color=NS.NPG[3], ms=2.5, label='Measured profile'),
            Line2D([], [], marker='o', ls='', color=NS.GREY, ms=2.5, label='Harmonised (Jobbágy)')],
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))
    NS.save(fig, os.path.join(C.FIG_DIR, fname))
    plt.show()


for cvs in C.CV_SCHEMES:
    tag = {'random': 'randomCV', 'spatial': 'spatialCV'}[cvs]
    make_figure('measured', cvs, f'Fig2_obs_vs_pred_measured_{tag}')
    make_figure('all', cvs, f'FigS2_obs_vs_pred_all_points_{tag}')

# ═══════════════════════════════════════════════════════════════════════════════
#  Random vs spatial CV — table + Fig. 2c (fold-level mean ± SD, measured profiles)
# ═══════════════════════════════════════════════════════════════════════════════
comp = []
for d in NS.DEPTH_ORDER:
    for m in MODELS:
        row = {'Depth': d, 'Model': m}
        for cvs in C.CV_SCHEMES:
            p = os.path.join(M.results_dir(d, cvs), f'summary_{m}.json')
            if os.path.exists(p):
                s = json.load(open(p))
                for k in ('r2', 'rmse', 'ccc'):
                    row[f'{cvs}_{k}'] = s[f'meas_{k}_mean']
                    row[f'{cvs}_{k}_sd'] = s[f'meas_{k}_sd']
                row[f'{cvs}_pooled_r2'] = s['pooled_measured']['r2']
                row[f'{cvs}_median_dist_km'] = s.get('median_test_train_km', np.nan)
        comp.append(row)
comp = pd.DataFrame(comp)
if {'random_r2', 'spatial_r2'} <= set(comp.columns):
    comp['delta_R2'] = comp['spatial_r2'] - comp['random_r2']
    comp['delta_RMSE_pct'] = 100 * (comp['spatial_rmse'] / comp['random_rmse'] - 1)
comp.to_csv(os.path.join(C.RESULTS_DIR, 'random_vs_spatialCV.csv'), index=False)
print(comp.round(3).to_string(index=False))

if {'random_r2', 'spatial_r2'} <= set(comp.columns):
    CVCOL = {'random': NS.GREY, 'spatial': NS.NPG[1]}
    nm = len(MODELS)
    fig, axs = plt.subplots(1, 3, figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.30),
                            gridspec_kw=dict(wspace=0.45))
    for ax, (k, lab) in zip(axs, [('r2', '$R^2$'), ('rmse', f'RMSE ({NS.UNIT})'),
                                  ('ccc', 'CCC')]):
        xt, xl = [], []
        for i, d in enumerate(NS.DEPTH_ORDER):
            for j, m in enumerate(MODELS):
                r = comp[(comp.Depth == d) & (comp.Model == m)].iloc[0]
                x0 = i * (nm + 1) + j
                for o, cvs in zip((-0.17, 0.17), C.CV_SCHEMES):
                    ax.errorbar(x0 + o, r[f'{cvs}_{k}'], yerr=r[f'{cvs}_{k}_sd'],
                                fmt=NS.MODEL_MARKER[m], ms=3.2,
                                color=CVCOL[cvs], mec='white', mew=0.3,
                                elinewidth=0.6, capsize=1.2)
                ax.plot([x0 - 0.17, x0 + 0.17], [r[f'random_{k}'], r[f'spatial_{k}']],
                        color=NS.LIGHT, lw=0.6, zorder=0)
            xt.append(i * (nm + 1) + (nm - 1) / 2)
            xl.append(NS.DEPTH_LABEL[d])
        ax.set_xticks(xt)
        ax.set_xticklabels(xl)
        ax.set_ylabel(lab)
        ax.tick_params(axis='x', length=0)
        NS.panel_letter(ax, 'abc'[list(axs).index(ax)], x=-0.30, y=1.02)
    from matplotlib.lines import Line2D as L2
    fig.legend(handles=[L2([], [], marker='o', ls='', color=NS.GREY, ms=3, label='Random CV'),
                        L2([], [], marker='o', ls='', color=NS.NPG[1], ms=3, label='Spatial block CV')]
                       + [L2([], [], marker=NS.MODEL_MARKER[m], ls='', color=NS.INK, ms=3,
                             label=NS.MODEL_LABEL[m]) for m in MODELS],
               loc='upper center', ncol=2 + len(MODELS), bbox_to_anchor=(0.5, 1.13))
    NS.save(fig, os.path.join(C.FIG_DIR, 'Fig2c_random_vs_spatialCV'))
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. S7 — QRF prediction-interval coverage (calibration of the uncertainty)
#  Empirical coverage (PICP) of the 50/80/90/95 % central intervals vs nominal,
#  held-out measured profiles, random and spatial CV. On the 1:1 line = well
#  calibrated; below = intervals too narrow; above = too wide.
# ═══════════════════════════════════════════════════════════════════════════════
if 'QRF' in MODELS:
    rows = []
    for cvs in C.CV_SCHEMES:
        for d in NS.DEPTH_ORDER:
            f = os.path.join(M.results_dir(d, cvs), 'qrf_interval_coverage.csv')
            if os.path.exists(f):
                rows.append(pd.read_csv(f).assign(CV=cvs, Depth=d))
    if rows:
        cov = pd.concat(rows, ignore_index=True)
        cov.to_csv(os.path.join(C.RESULTS_DIR, 'Table_QRF_interval_coverage.csv'), index=False)
        print(cov[cov.scope == 'measured'].round(2).to_string(index=False))
        CVCOL = {'random': NS.GREY, 'spatial': NS.NPG[1]}
        fig, axs = plt.subplots(1, 3, figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.33),
                                gridspec_kw=dict(wspace=0.35))
        for c, d in enumerate(NS.DEPTH_ORDER):
            ax = axs[c]
            ax.plot([40, 100], [40, 100], color=NS.INK, lw=0.6, ls='--')
            for cvs in C.CV_SCHEMES:
                s = cov[(cov.Depth == d) & (cov.CV == cvs) & (cov.scope == 'measured')]
                if len(s):
                    ax.plot(s.nominal, s.picp, '-o', color=CVCOL[cvs], ms=3, lw=0.9,
                            label={'random': 'Random CV', 'spatial': 'Spatial block CV'}[cvs])
                    w90 = s.loc[s.nominal == 90, 'mpiw'].values
                    if len(w90):
                        ax.text(0.97, 0.05 + 0.09 * C.CV_SCHEMES.index(cvs),
                                f'MPIW$_{{90}}$ = {w90[0]:.0f}', transform=ax.transAxes,
                                ha='right', fontsize=5.8, color=CVCOL[cvs])
            ax.set_xlim(45, 100)
            ax.set_ylim(40, 100)
            ax.set_xlabel('Nominal coverage (%)')
            if c == 0:
                ax.set_ylabel('Empirical coverage, PICP (%)')
                ax.legend(loc='upper left', fontsize=5.8)
            ax.set_title(f'QRF · {NS.DEPTH_LABEL[d]}', pad=3)
            NS.panel_letter(ax, 'abc'[c], x=-0.25, y=1.02)
        NS.save(fig, os.path.join(C.FIG_DIR, 'FigS7_QRF_interval_coverage'))
        plt.show()
