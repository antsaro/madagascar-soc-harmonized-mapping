# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DEPTH HARMONISATION (Jobbágy & Jackson 2000, locally calibrated)
#  → SOC_harmonized_table.csv  +  validation tables  +  Fig. S1 (Nature style)
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import nature_style as NS
import soc_config as C
import soc_harmonize as H

FONT = NS.setup_arial()
os.makedirs(C.HARM_DIR, exist_ok=True)

# ── 1. Load full point table & apply the hand corrections ─────────────────────
raw = pd.read_csv(C.PREDICTORS_CSV).replace(-9999, np.nan)
raw = raw.rename(columns={'SOC100_ten': 'SOC100', 'SOC0_30': 'SC30', 'SOC0_100': 'SOC100'})
raw, n_cor = H.apply_corrections(raw, C.CORRECTIONS_CSV)
print(f'Points: {len(raw)} | SC30: {raw.SC30.notna().sum()} | '
      f'SOC100 measured: {raw.SOC100.notna().sum()} | corrected rows applied: {n_cor}')
bad = (raw.SOC100 < raw.SC30).sum()
print(f'Profiles still with SOC100 < SC30: {bad} → kept as measured (Deep = 0), '
      f'excluded from β calibration')

# ── 2. Groups, per-profile decay rate, calibration ───────────────────────────
tab = H.prepare_table(raw, calibrate_on=C.CALIBRATE_ON)
params, info = H.fit_params(tab, k0=C.SHRINK_K0)
tab = H.harmonize(tab, params)
tab, boot_k = H.bootstrap_uncertainty(tab, n_boot=C.N_BOOT_HARM, k0=C.SHRINK_K0)

# bootstrap CI for every group's β / f30
for g in info['group']:
    if g in boot_k:
        kb = boot_k[g].values
        info.loc[info.group == g, 'f30_p05'] = np.percentile(H.frac(kb, 30.), 5)
        info.loc[info.group == g, 'f30_p95'] = np.percentile(H.frac(kb, 30.), 95)
info['n_harmonized'] = info['group'].map(
    tab[tab.SOC100_src == 'harmonized'].groupby('HGROUP').size()).fillna(0).astype(int)
info['JJ2000_f30'] = info['group'].map(
    lambda g: H.frac(H.JJ2000_K[H.GROUP_TO_JJ[g]], 30.) if H.GROUP_TO_JJ.get(g) else np.nan)
info.to_csv(C.HARM_PARAMS, index=False)
print('\nJobbágy depth-model parameters (k: cm⁻¹, β = e^-k, f30 = share of 0–100 cm in 0–30 cm)')
print(info.round(4).to_string(index=False))

# ── 3. Validation on paired profiles (10 × 10-fold) ──────────────────────────
oof_h, val = H.validate(tab, n_splits=10, n_repeats=2 if C.FAST else 10, k0=C.SHRINK_K0)
val.to_csv(f'{C.HARM_DIR}/harmonization_validation.csv', index=False)
oof_h.to_csv(f'{C.HARM_DIR}/harmonization_oof.csv', index=False)
print('\nValidation (out-of-fold, paired profiles)')
print(val.round(3).to_string(index=False))

# ── 4. Save harmonised table ──────────────────────────────────────────────────
tab.to_csv(C.HARM_TABLE, index=False)
print(f'\n✅ Harmonised table → {C.HARM_TABLE}')
print(tab.groupby(['HGROUP', 'SOC100_src']).agg(
    n=('SC30', 'size'), SC30=('SC30', 'mean'), SOC100=('SOC100_h', 'mean'),
    Deep=('DeepSOC30_100_h', 'mean'), sd=('SOC100_h_sd', 'mean')).round(1).to_string())
print('\nGroup source for harmonised points:',
      tab[tab.SOC100_src == 'harmonized'].GROUP_SRC.value_counts().to_dict())

# ═══════════════════════════════════════════════════════════════════════════════
#  Fig. S1 — harmonisation (183 mm, 2 × 2)
# ═══════════════════════════════════════════════════════════════════════════════
ORDER = ['Forest', 'Agroforestry', 'Fallow', 'Shrubland', 'Grassland',
         'Cropland', 'Rice/Wetland', 'Mangrove']
GCOL = dict(zip(ORDER, NS.NPG[:7] + ['#7E6148']))
gi = info[info.level == 'group'].set_index('group')

fig, axs = plt.subplots(2, 2, figsize=(NS.DOUBLE_COL, NS.DOUBLE_COL * 0.78),
                        gridspec_kw=dict(wspace=0.38, hspace=0.45))

# a — observed f30 per group + calibrated value + JJ2000
ax = axs[0, 0]
cal = tab[tab.calib]
rng = np.random.default_rng(1)
for i, g in enumerate(ORDER):
    v = cal.loc[cal.HGROUP == g, 'ratio30_100'].values
    if len(v):
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=2.5, color=GCOL[g],
                   alpha=0.35, lw=0, rasterized=True)
        ax.boxplot(v, positions=[i], widths=0.55, showfliers=False,
                   medianprops=dict(color=NS.INK, lw=0.8),
                   boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                   capprops=dict(lw=0.6))
    f, lo, hi = gi.loc[g, ['f30_100', 'f30_p05', 'f30_p95']]
    ax.errorbar(i + 0.38, f, yerr=[[f - lo], [hi - f]], fmt='D', ms=3,
                color='black', elinewidth=0.6, capsize=1.5, zorder=5)
    jj = gi.loc[g, 'JJ2000_f30']
    if np.isfinite(jj):
        ax.plot([i - 0.3, i + 0.3], [jj, jj], color=NS.GREY, ls='--', lw=0.8, zorder=4)
ax.axhline(0.30, color=NS.LIGHT, lw=0.6, zorder=0)
ax.text(-0.45, 0.285, 'uniform profile', fontsize=5.5, color=NS.GREY,
        ha='left', va='top')
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([f'{g}\n(n={int(gi.loc[g, "n"])})' for g in ORDER],
                   rotation=45, ha='right', rotation_mode='anchor', fontsize=5.8)
ax.set_ylabel(r'SOC$_{\mathregular{0–30}}$ / SOC$_{\mathregular{0–100}}$')
ax.set_ylim(0, 1.18)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.plot([], [], 'D', color='black', ms=3, label='Calibrated (90% CI)')
ax.plot([], [], '--', color=NS.GREY, lw=0.8, label='Jobbágy & Jackson (2000)')
ax.legend(loc='upper center', fontsize=5.5, ncol=2, columnspacing=1.0)
NS.panel_letter(ax, 'a', x=-0.16)

# b — cumulative SOC distribution with depth
ax = axs[0, 1]
d = np.linspace(0, 100, 201)
for g in ORDER:
    k = gi.loc[g, 'k']
    y = np.array([H.frac(k, di) if di > 0 else 0.0 for di in d])
    ax.plot(y, d, color=GCOL[g], lw=1.0, label=g)
for lab, kk in H.JJ2000_K.items():
    y = np.array([H.frac(kk, di) if di > 0 else 0.0 for di in d])
    ax.plot(y, d, color=NS.GREY, lw=0.6, ls='--')
ax.plot([0, 1], [0, 100], color=NS.LIGHT, lw=0.6, zorder=0)
ax.axhline(30, color=NS.INK, lw=0.4, ls=':')
ax.set_ylim(100, 0)
ax.set_xlim(0, 1)
ax.set_xlabel('Cumulative fraction of SOC$_{\\mathregular{0–100}}$')
ax.set_ylabel('Depth (cm)')
ax.plot([], [], color=NS.GREY, lw=0.6, ls='--', label='JJ2000 (F, G, S)')
ax.legend(fontsize=5.5, loc='lower left', handlelength=1.0)
NS.panel_letter(ax, 'b', x=-0.16)


def dens_scatter(ax, x, y, cmap='viridis'):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    try:
        z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    except Exception:                                            # noqa: BLE001
        z = np.ones_like(x)
    o = z.argsort()
    ax.scatter(x[o], y[o], c=z[o], s=3, cmap=cmap, lw=0, rasterized=True)
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, color=NS.INK, lw=0.6, ls='--')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')


# c — OOF validation (non-mangrove profiles, local calibration)
ax = axs[1, 0]
nm = oof_h[oof_h.HGROUP != 'Mangrove']
dens_scatter(ax, nm.SOC100.values, nm.SOC100_local.values)
r = val[(val.subset == 'Excl. mangroves') & (val.target == 'SOC100')].set_index('scheme')
ax.text(0.04, 0.97,
        f"Local calibration\n$R^2$ = {r.loc['local', 'r2']:.2f}\n"
        f"RMSE = {r.loc['local', 'rmse']:.1f}\nCCC = {r.loc['local', 'ccc']:.2f}\n"
        f"JJ2000 $R^2$ = {r.loc['literature', 'r2']:.2f}\n$n$ = {int(r.loc['local', 'n'])}",
        transform=ax.transAxes, va='top', fontsize=5.8, linespacing=1.35)
ax.set_xlabel(f'Measured SOC$_{{\\mathregular{{0–100}}}}$ ({NS.UNIT})')
ax.set_ylabel(f'Harmonised SOC$_{{\\mathregular{{0–100}}}}$ ({NS.UNIT})')
NS.panel_letter(ax, 'c', x=-0.16)

# d — measured vs harmonised distributions per group
ax = axs[1, 1]
pos, ticks = 0, []
for g in [g for g in ORDER if g != 'Mangrove']:
    for j, (s, colr) in enumerate([('measured', NS.GREY), ('harmonized', GCOL[g])]):
        v = tab.loc[(tab.HGROUP == g) & (tab.SOC100_src == s), 'SOC100_h'].values
        if len(v) > 2:
            vp = ax.violinplot(v, positions=[pos + j * 0.42], widths=0.4,
                               showextrema=False, showmedians=False)
            for b in vp['bodies']:
                b.set_facecolor(colr)
                b.set_edgecolor('none')
                b.set_alpha(0.8)
            ax.plot(pos + j * 0.42, np.median(v), 'o', ms=1.8, color='white',
                    mec=NS.INK, mew=0.4)
    ticks.append(pos + 0.21)
    pos += 1.2
ax.set_xticks(ticks)
ax.set_xticklabels([g for g in ORDER if g != 'Mangrove'], rotation=45, ha='right',
                   rotation_mode='anchor', fontsize=5.8)
ax.set_ylabel(f'SOC$_{{\\mathregular{{0–100}}}}$ ({NS.UNIT})')
ax.fill_between([], [], color=NS.GREY, label='Measured')
ax.fill_between([], [], color=NS.NPG[0], label='Harmonised')
ax.legend(fontsize=5.5, loc='upper right')
NS.panel_letter(ax, 'd', x=-0.16)

NS.save(fig, f'{C.HARM_DIR}/FigS1_depth_harmonization')
plt.show()
