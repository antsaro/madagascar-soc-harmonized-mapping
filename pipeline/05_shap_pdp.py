# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — SHAP (out-of-fold) + PARTIAL DEPENDENCE, Nature-Reviews style
#   XGBoost : GPU TreeSHAP  (booster.predict(..., pred_contribs=True), device=cuda)
#   RF      : sklearn twin with the same tuned hyper-parameters + shap.TreeExplainer
#  Fig. 3  : mean |SHAP| importance, RF vs XGBoost, 3 depths
#  Fig. 4  : SHAP beeswarm, 2 × 3
#  Fig. 5/6: SHAP dependence (LOWESS ± 95% CI) with centred PDP overlay — per model
#  Fig. S3/S4 : SHAP by land-cover / soil class
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import os
import textwrap
import warnings

import joblib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from statsmodels.nonparametric.smoothers_lowess import lowess

import nature_style as NS
import soc_config as C
import soc_ml as M

warnings.filterwarnings('ignore')
FONT = NS.setup_arial(verbose=False)

MODELS = C.MODELS                  # QRF is explained through its conditional mean
N_FOLDS, N_SEEDS = (3, 1) if C.FAST else (10, 3)   # OOF SHAP: 3 × 10-fold
TOP_K_DEP = 4                     # numeric features shown in dependence figure
PDP_GRID, PDP_SAMPLE = (15, 100) if C.FAST else (40, 600)
SHAP_DIR = os.path.join(C.RESULTS_DIR, 'SHAP')
os.makedirs(SHAP_DIR, exist_ok=True)
USE_GPU = M.gpu_available()
print(f'GPU TreeSHAP for XGBoost: {USE_GPU} | final models from {C.FINAL_MODEL_CV} CV')

NUM = C.NUMERIC_PREDICTORS
CAT_ENC = [f'{c}_enc' for c in C.CATEGORICAL_PREDICTORS]


# ── SHAP for one fold ─────────────────────────────────────────────────────────
def fold_shap(model_name, params, X, y, tr, te, seed):
    if model_name == 'XGBoost':
        mdl = M.make_model('XGBoost', params, USE_GPU, seed=seed)
        mdl.fit(X[tr], y[tr])
        bst = mdl.get_booster()
        contrib = bst.predict(xgb.DMatrix(X[te]), pred_contribs=True)
        return contrib[:, :-1], contrib[:, -1]
    import shap
    mdl = M.to_sklearn_rf(params, seed=seed)
    mdl.set_params(n_jobs=1)
    mdl.fit(X[tr], y[tr])
    ex = shap.TreeExplainer(mdl, feature_perturbation='tree_path_dependent')
    sv = ex.shap_values(X[te], check_additivity=False)
    return np.asarray(sv), np.full(len(te), float(np.ravel(ex.expected_value)[0]))


def oof_shap(model_name, params, X, y):
    tasks = []
    for s in range(N_SEEDS):
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=137 * s + 1).split(X):
            tasks.append((s, tr, te))
    n_jobs = 1 if (model_name == 'XGBoost' and USE_GPU) else -1
    res = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(fold_shap)(model_name, params, X, y, tr, te, 1000 * s + i)
        for i, (s, tr, te) in enumerate(tasks))
    sv = np.vstack([r[0] for r in res])
    base = np.concatenate([r[1] for r in res])
    idx = np.concatenate([t[2] for t in tasks])
    return sv, base, idx


def pdp_curves(model, X, feat_idx, rng):
    """Centred partial dependence (mean ICE − mean prediction) + 10–90 % ICE band."""
    sub = X[rng.choice(len(X), min(PDP_SAMPLE, len(X)), replace=False)]
    base = M.predict(model, sub).mean()
    out = {}
    for j in feat_idx:
        grid = np.linspace(*np.percentile(X[:, j], [2, 98]), PDP_GRID)
        ice = np.empty((len(grid), len(sub)))
        Xg = np.repeat(sub[None], len(grid), 0)
        Xg[:, :, j] = grid[:, None]
        ice[:] = M.predict(model, Xg.reshape(-1, X.shape[1])).reshape(len(grid), -1)
        ice_c = ice - ice.mean(0, keepdims=True) + (ice.mean() - base)
        out[j] = dict(grid=grid, pdp=ice.mean(1) - base,
                      lo=np.percentile(ice_c, 10, 1), hi=np.percentile(ice_c, 90, 1))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Compute
# ═══════════════════════════════════════════════════════════════════════════════
df, features, encoders = M.load_harmonized_table()
X = df[features].values.astype(np.float32)
R = {}
for d in NS.DEPTH_ORDER:
    y, _ = M.full_target(df, d)
    for m in MODELS:
        pkl = os.path.join(M.results_dir(d, C.FINAL_MODEL_CV), 'models', f'{m}.pkl')
        params = joblib.load(pkl)['best_params']
        print(f'SHAP  {m:<12} | {d:<7} | params {params}', flush=True)
        sv, base, idx = oof_shap(m, params, X, y)
        full = (M.to_sklearn_rf(params) if m in ('RandomForest', 'QRF')
                else M.make_model(m, params, USE_GPU)).fit(X, y)
        rng = np.random.default_rng(0)
        imp = np.abs(sv).mean(0)
        top_num = [features.index(f) for f in
                   sorted(NUM, key=lambda f: -imp[features.index(f)])[:TOP_K_DEP]]
        pdp = pdp_curves(full, X, top_num, rng)
        R[(m, d)] = dict(sv=sv, Xv=X[idx], idx=idx, imp=imp,
                         imp_sd=np.array([np.abs(sv[s * len(X):(s + 1) * len(X)]).mean(0)
                                         for s in range(N_SEEDS)]).std(0),
                         top_num=top_num, pdp=pdp)
        pd.DataFrame({'feature': features, 'label': [NS.flabel(f) for f in features],
                      'mean_abs_shap': imp,
                      'share_pct': 100 * imp / imp.sum()}).sort_values(
            'mean_abs_shap', ascending=False).to_csv(
            os.path.join(SHAP_DIR, f'shap_importance_{m}_{d}.csv'), index=False)
        np.savez_compressed(os.path.join(SHAP_DIR, f'shap_values_{m}_{d}.npz'),
                            shap=sv, X=X[idx], row_id=idx, base=base, features=features)

# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 3 — importance (grouped horizontal bars, RF vs XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(1, 3, figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.36),
                        gridspec_kw=dict(wspace=0.75))
for c, d in enumerate(NS.DEPTH_ORDER):
    ax = axs[c]
    rel = {m: 100 * R[(m, d)]['imp'] / R[(m, d)]['imp'].sum() for m in MODELS}
    rel_sd = {m: 100 * R[(m, d)]['imp_sd'] / R[(m, d)]['imp'].sum() for m in MODELS}
    order = np.argsort(-sum(rel[m] for m in MODELS))
    ypos = np.arange(len(features))[::-1]
    hb = 0.8 / len(MODELS)
    for k, m in enumerate(MODELS):
        ax.barh(ypos + (len(MODELS) - 1) / 2 * hb - k * hb, rel[m][order], height=hb * 0.95,
                xerr=rel_sd[m][order], color=NS.MODEL_COLOR[m], label=NS.MODEL_LABEL[m],
                error_kw=dict(lw=0.5, capsize=1, capthick=0.5, ecolor=NS.INK))
    ax.set_yticks(ypos)
    ax.set_yticklabels([NS.flabel(features[i]) for i in order])
    ax.set_xlabel('Relative importance,\nmean |SHAP| (%)')
    ax.set_title(NS.DEPTH_LABEL[d], pad=3)
    ax.tick_params(axis='y', length=0)
    NS.panel_letter(ax, 'abc'[c], x=-0.55, y=1.02)
axs[0].legend(loc='lower right', fontsize=5.8)
NS.save(fig, os.path.join(C.FIG_DIR, 'Fig3_SHAP_importance'))
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 4 — beeswarm (2 × 3)
# ═══════════════════════════════════════════════════════════════════════════════
BEE = mcolors.LinearSegmentedColormap.from_list('bee', ['#3C5488', '#B7C3DE', '#F2C6BE', '#E64B35'])


def beeswarm(ax, sv, Xv, max_pts=2500, rng=np.random.default_rng(3)):
    imp = np.abs(sv).mean(0)
    order = np.argsort(imp)                      # least important at bottom
    for row, j in enumerate(order):
        s = sv[:, j]
        sel = rng.choice(len(s), min(max_pts, len(s)), replace=False)
        s, xv = s[sel], Xv[sel, j]
        # jitter proportional to local density (histogram-based)
        bins = np.linspace(s.min(), s.max() + 1e-9, 60)
        b = np.digitize(s, bins)
        cnt = np.bincount(b, minlength=len(bins) + 1)[b]
        jit = rng.uniform(-1, 1, len(s)) * 0.38 * cnt / cnt.max()
        if features[j] in CAT_ENC:
            col = NS.GREY
        else:
            lo, hi = np.percentile(xv, [5, 95])
            col = BEE(np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1))
        ax.scatter(s, row + jit, c=col, s=0.8, lw=0, rasterized=True)
    ax.axvline(0, color=NS.INK, lw=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NS.flabel(features[j]) for j in order])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)


fig, axs = plt.subplots(len(MODELS), 3, squeeze=False,
                        figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.31 * len(MODELS)),
                        gridspec_kw=dict(wspace=0.65, hspace=0.42))
k = 0
for r, m in enumerate(MODELS):
    for c, d in enumerate(NS.DEPTH_ORDER):
        ax = axs[r, c]
        beeswarm(ax, R[(m, d)]['sv'], R[(m, d)]['Xv'])
        ax.set_title(f'{NS.MODEL_LABEL[m]} · {NS.DEPTH_LABEL[d]}', pad=3)
        if r == len(MODELS) - 1:
            ax.set_xlabel(f'SHAP value ({NS.UNIT})')
        NS.panel_letter(ax, 'abcdefghi'[k], x=-0.52, y=1.02)
        k += 1
cax = fig.add_axes([0.93, 0.35, 0.008, 0.3])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=BEE), cax=cax, ticks=[0, 1])
cb.ax.set_yticklabels(['Low', 'High'])
cb.set_label('Feature value', labelpad=-4)
cb.outline.set_linewidth(0.4)
NS.save(fig, os.path.join(C.FIG_DIR, 'Fig4_SHAP_beeswarm'))
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. 5/6 — SHAP dependence + centred PDP (rows = depth, cols = top features)
# ═══════════════════════════════════════════════════════════════════════════════
def lowess_ci(x, y, frac=0.3, n_boot=100, rng=np.random.default_rng(7)):
    grid = np.linspace(*np.percentile(x, [2, 98]), 60)
    fit = lambda xx, yy: np.interp(grid, *lowess(yy, xx, frac=frac, return_sorted=True).T)
    main = fit(x, y)
    sub = min(len(x), 2000)
    boots = []
    for _ in range(n_boot):
        i = rng.choice(len(x), sub, replace=True)
        boots.append(fit(x[i], y[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return grid, main, lo, hi


def dependence_figure(m):
    fig = plt.figure(figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.78))
    outer = gridspec.GridSpec(3, TOP_K_DEP, figure=fig, wspace=0.42, hspace=0.62)
    k = 0
    for r, d in enumerate(NS.DEPTH_ORDER):
        res = R[(m, d)]
        for c, j in enumerate(res['top_num']):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, c],
                                                     height_ratios=[1, 4.5], hspace=0.05)
            axh, ax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            x, s = res['Xv'][:, j], res['sv'][:, j]
            lo_x, hi_x = np.percentile(x, [1, 99])
            keep = (x >= lo_x) & (x <= hi_x)
            x, s = x[keep], s[keep]
            ax.scatter(x, s, s=1.2, color=NS.MODEL_COLOR[m], alpha=0.18, lw=0,
                       rasterized=True, zorder=1)
            g, mu, lo, hi = lowess_ci(x, s)
            ax.fill_between(g, lo, hi, color=NS.INK, alpha=0.18, lw=0, zorder=2)
            ax.plot(g, mu, color=NS.INK, lw=1.0, zorder=3, label='SHAP (LOWESS ± 95% CI)')
            p = res['pdp'][j]
            # align PDP with SHAP: zero mean over the observed distribution of x
            shift = np.mean(np.interp(x, p['grid'], p['pdp']))
            p = dict(grid=p['grid'], pdp=p['pdp'] - shift, lo=p['lo'] - shift,
                     hi=p['hi'] - shift)
            ax.fill_between(p['grid'], p['lo'], p['hi'], color=NS.NPG[2], alpha=0.12,
                            lw=0, zorder=1)
            ax.plot(p['grid'], p['pdp'], color=NS.NPG[2], lw=1.0, ls='--', zorder=4,
                    label='Centred PDP (ICE 10–90%)')
            ax.axhline(0, color=NS.GREY, lw=0.4, ls=':', zorder=0)
            ylo, yhi = np.percentile(s, [0.5, 99.5])
            pad = 0.1 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)
            ax.set_xlim(lo_x, hi_x)
            ax.set_xlabel(NS.flabel(features[j]), labelpad=1.5)
            ax.locator_params(nbins=4)
            if c == 0:
                ax.set_ylabel(f'{NS.DEPTH_LABEL[d]}\nSHAP ({NS.UNIT})')
            axh.hist(x, bins=35, color=NS.LIGHT, edgecolor='none')
            axh.set_xlim(lo_x, hi_x)
            axh.axis('off')
            NS.panel_letter(axh, 'abcdefghijkl'[k], x=-0.30 if c else -0.42, y=0.4)
            k += 1
            if r == 0 and c == TOP_K_DEP - 1:
                h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(NS.MODEL_LABEL[m], x=0.125, y=0.975, ha='left', fontsize=7.5,
                 fontweight='bold')
    NS.save(fig, os.path.join(C.FIG_DIR, f'Fig5{"abc"[MODELS.index(m)]}_SHAP_PDP_dependence_{m}'))
    plt.show()


for m in MODELS:
    dependence_figure(m)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. S3/S4 — SHAP by category (land cover, soil type)
# ═══════════════════════════════════════════════════════════════════════════════
def categorical_figure(m):
    fig, axs = plt.subplots(len(CAT_ENC), 3,
                            figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.75),
                            gridspec_kw=dict(wspace=0.12, hspace=0.35))
    k = 0
    for r, (cenc, craw) in enumerate(zip(CAT_ENC, C.CATEGORICAL_PREDICTORS)):
        classes = encoders[craw].classes_
        j = features.index(cenc)
        # common class order (median SHAP for SOC100)
        ref = R[(m, 'SOC100')]
        med = [np.median(ref['sv'][ref['Xv'][:, j] == i, j]) if (ref['Xv'][:, j] == i).any()
               else 0 for i in range(len(classes))]
        order = np.argsort(med)
        for c, d in enumerate(NS.DEPTH_ORDER):
            ax = axs[r, c]
            res = R[(m, d)]
            data = [res['sv'][res['Xv'][:, j] == i, j] for i in order]
            bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False,
                            patch_artist=True, medianprops=dict(color=NS.INK, lw=0.8),
                            whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                            boxprops=dict(lw=0.5))
            for patch, dd in zip(bp['boxes'], data):
                med_v = np.median(dd) if len(dd) else 0
                patch.set_facecolor(NS.NPG[0] if med_v > 0 else NS.NPG[3])
                patch.set_alpha(0.55)
            ax.axvline(0, color=NS.GREY, lw=0.5, ls=':')
            n_per = [len(dd) for dd in data]
            if c == 0:
                ax.set_yticklabels([f'{textwrap.shorten(classes[i], 30)} ({n})'
                                    for i, n in zip(order, n_per)], fontsize=5.6)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            ax.set_title(f'{NS.flabel(craw)} · {NS.DEPTH_LABEL[d]}', pad=3)
            if r == len(CAT_ENC) - 1:
                ax.set_xlabel(f'SHAP value ({NS.UNIT})')
            NS.panel_letter(ax, 'abcdef'[k], x=-0.08 if c else -0.75, y=1.02)
            k += 1
    fig.suptitle(NS.MODEL_LABEL[m], x=0.02, y=0.99, ha='left', fontsize=7.5,
                 fontweight='bold')
    NS.save(fig, os.path.join(C.FIG_DIR, f'FigS3{"abc"[MODELS.index(m)]}_SHAP_categorical_{m}'))
    plt.show()


for m in MODELS:
    categorical_figure(m)
print('✅ SHAP done →', SHAP_DIR)
