# ═══════════════════════════════════════════════════════════════════════════════
#  soc_harmonize.py — depth harmonisation 0–30 cm → 0–100 cm
#  Jobbágy & Jackson (2000) exponential depth model, calibrated on local profiles
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Model (Gale & Grigal 1987; Jobbágy & Jackson 2000, Ecol. Appl. 10:423-436)
#      Y(d) = 1 − β^d                 cumulative fraction of SOC above depth d (cm)
#  Writing β = exp(−k) (k = decay rate, cm⁻¹), the fraction of the 0–100 cm stock
#  held in 0–30 cm is
#      f₃₀ = (1 − β³⁰) / (1 − β¹⁰⁰) = expm1(−30k) / expm1(−100k)
#  so for a profile with only SOC₀₋₃₀:
#      SOC₀₋₁₀₀ = SOC₀₋₃₀ / f₃₀      SOC₃₀₋₁₀₀ = SOC₀₋₁₀₀ − SOC₀₋₃₀
#
#  k (β) is estimated per vegetation group from the paired profiles (SC30 & SOC100
#  both measured). Small groups are shrunk towards their parent group (woody /
#  herbaceous), and parents towards the global Jobbágy & Jackson values
#  (share of 0–100 cm SOC in 0–20 cm: forests 50 %, grasslands 42 %,
#  shrublands 33 %). k < 0 (β > 1) is allowed: SOC increasing with depth
#  (f₃₀ < 0.30), which is common in mangroves and hydromorphic soils.
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# ── Field land-use label → harmonisation group ────────────────────────────────
LU_TO_GROUP = {
    'Forest': 'Forest', 'Closed Canopy': 'Forest', 'Pine': 'Forest',
    'Reforestation': 'Forest',
    'AgroF': 'Agroforestry', 'AGroF': 'Agroforestry', 'Agroforesterie': 'Agroforestry',
    'Tree fallow': 'Fallow', 'Shrub fallow': 'Fallow', 'Jachere': 'Fallow',
    'Shrubland': 'Shrubland', 'Pseudo-steppe': 'Shrubland',
    'Grassland': 'Grassland', 'Tany Maty': 'Grassland', 'Tany maty': 'Grassland',
    'Bare': 'Grassland',
    'Crop': 'Cropland', 'Cropland': 'Cropland', 'Culture': 'Cropland',
    'Rice': 'Rice/Wetland',
    'Mangroves': 'Mangrove',
}
# ── ESA WorldCover class (fallback when LU is 'undetermined') ─────────────────
ESA_TO_GROUP = {
    'Tree Cover': 'Forest', 'Shrubland': 'Shrubland', 'Grassland': 'Grassland',
    'Cropland': 'Cropland', 'Built-up': 'Grassland',
    'Bare / Sparse Vegetation': 'Grassland', 'Herbaceous Wetland': 'Rice/Wetland',
    'Permanent Water Bodies': 'Rice/Wetland', 'Mangroves': 'Mangrove',
    'Moss and Lichen': 'Grassland', 'Snow and Ice': 'Grassland',
}
PARENT = {
    'Forest': 'Woody', 'Agroforestry': 'Woody', 'Fallow': 'Woody',
    'Shrubland': 'Woody',
    'Grassland': 'Herbaceous', 'Cropland': 'Herbaceous', 'Rice/Wetland': 'Herbaceous',
    'Mangrove': 'Mangrove',
}
# Jobbágy & Jackson (2000) global share of 0–100 cm SOC found in 0–20 cm
JJ2000_F20 = {'Forest': 0.50, 'Grassland': 0.42, 'Shrubland': 0.33}
GROUP_TO_JJ = {
    'Forest': 'Forest', 'Agroforestry': 'Forest', 'Fallow': 'Forest',
    'Woody': 'Forest', 'Shrubland': 'Shrubland',
    'Grassland': 'Grassland', 'Cropland': 'Grassland', 'Rice/Wetland': 'Grassland',
    'Herbaceous': 'Grassland', 'Mangrove': None,
}

K_BOUNDS = (-0.05, 0.50)     # decay rate bounds (cm⁻¹); β ∈ [0.61, 1.05]


# ═══════════════════════════════════════════════════════════════════════════════
#  Depth-model algebra
# ═══════════════════════════════════════════════════════════════════════════════
def frac(k, d_top, d_bot=100.0):
    """Fraction of the 0–d_bot stock contained in 0–d_top for decay rate k."""
    k = np.asarray(k, dtype=float)
    out = np.empty_like(k)
    small = np.abs(k) < 1e-9
    out[small] = d_top / d_bot                         # uniform profile limit
    ks = k[~small]
    out[~small] = np.expm1(-d_top * ks) / np.expm1(-d_bot * ks)
    return out if out.ndim else float(out)


def k_from_ratio(r, d_top=30.0, d_bot=100.0):
    """Invert f(k) = r for one profile (vectorised via loop; ~1 µs each)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    lo, hi = K_BOUNDS
    f_lo, f_hi = frac(lo, d_top, d_bot), frac(hi, d_top, d_bot)
    out = np.full(r.shape, np.nan)
    for i, ri in enumerate(r):
        if not np.isfinite(ri):
            continue
        if ri <= f_lo:
            out[i] = lo
        elif ri >= f_hi:
            out[i] = hi
        else:
            out[i] = brentq(lambda kk: frac(kk, d_top, d_bot) - ri, lo, hi, xtol=1e-10)
    return out


def k_from_f20(f20):
    return float(k_from_ratio([f20], d_top=20.0)[0])


JJ2000_K = {g: k_from_f20(v) for g, v in JJ2000_F20.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Table preparation
# ═══════════════════════════════════════════════════════════════════════════════
def assign_groups(df, lu_col='LU', esa_col='lulc'):
    """HGROUP from the field LU label; ESA WorldCover class when LU is unknown."""
    g = df[lu_col].astype(str).str.strip().map(LU_TO_GROUP) if lu_col in df else None
    src = pd.Series('LU', index=df.index)
    if g is None:
        g = pd.Series(np.nan, index=df.index, dtype=object)
    miss = g.isna()
    if esa_col in df.columns:
        g.loc[miss] = df.loc[miss, esa_col].astype(str).map(ESA_TO_GROUP)
        src.loc[miss] = 'ESA'
    still = g.isna()
    g.loc[still] = 'Grassland'                     # last-resort default
    src.loc[still] = 'default'
    return g.astype(str), src


def apply_corrections(df, corrected_csv=None, key='num'):
    """Overwrite SC30/SOC100 with the manually corrected values (rows where
    SOC100 < SC30 had been fixed by hand), matched on `key`."""
    df = df.copy()
    if corrected_csv is None:
        return df, 0
    try:
        cor = pd.read_csv(corrected_csv, sep=None, engine='python')
    except FileNotFoundError:
        print(f'[harm] corrections file not found: {corrected_csv} — skipped')
        return df, 0
    if key not in cor.columns:
        print(f"[harm] '{key}' missing in corrections file — skipped")
        return df, 0
    for c in ('SC30', 'SOC100'):
        if c in cor.columns:
            cor[c] = pd.to_numeric(cor[c].astype(str).str.replace(',', '.'),
                                   errors='coerce')
    cor = cor.drop_duplicates(key).set_index(key)
    hit = df[key].isin(cor.index)
    for c in ('SC30', 'SOC100'):
        if c in cor.columns:
            df.loc[hit, c] = df.loc[hit, key].map(cor[c]).values
    return df, int(hit.sum())


def prepare_table(df, lu_col='LU', esa_col='lulc', r_max=0.98, r_min=0.02,
                  calibrate_on='LU'):
    """Adds HGROUP, GROUP_SRC, ratio, k_obs and the calibration flag."""
    df = df.copy()
    df['HGROUP'], df['GROUP_SRC'] = assign_groups(df, lu_col, esa_col)
    df['PARENT'] = df['HGROUP'].map(PARENT)
    meas = df['SOC100'].notna() & df['SC30'].notna() & (df['SOC100'] > 0)
    df['measured100'] = meas
    df['ratio30_100'] = np.where(meas, df['SC30'] / df['SOC100'], np.nan)
    df['k_obs'] = np.nan
    df.loc[meas, 'k_obs'] = k_from_ratio(df.loc[meas, 'ratio30_100'].values)
    calib = meas & df['ratio30_100'].between(r_min, r_max)
    if calibrate_on == 'LU':                       # only profiles with a field LU
        calib &= df['GROUP_SRC'].eq('LU')
    df['calib'] = calib
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Calibration (hierarchical shrinkage of the median decay rate)
# ═══════════════════════════════════════════════════════════════════════════════
def fit_params(df_cal, k0=10.0, stat='median'):
    """Returns {group: k} for every group in PARENT (plus parents)."""
    agg = np.median if stat == 'median' else np.mean
    d = df_cal[df_cal['calib']]
    params, info = {}, []
    # parents first: shrink towards Jobbágy & Jackson (2000)
    parent_k = {}
    for p in sorted(set(PARENT.values())):
        sub = d.loc[d['PARENT'] == p, 'k_obs'].dropna()
        jj = GROUP_TO_JJ.get(p)
        prior = JJ2000_K[jj] if jj else (agg(sub) if len(sub) else 0.0)
        n = len(sub)
        kp = (n * agg(sub) + k0 * prior) / (n + k0) if n else prior
        parent_k[p] = kp
        info.append(dict(level='parent', group=p, n=n,
                         k_local=agg(sub) if n else np.nan, k_prior=prior, k=kp))
    for g, p in PARENT.items():
        sub = d.loc[d['HGROUP'] == g, 'k_obs'].dropna()
        n = len(sub)
        kg = (n * agg(sub) + k0 * parent_k[p]) / (n + k0) if n else parent_k[p]
        params[g] = kg
        info.append(dict(level='group', group=g, n=n,
                         k_local=agg(sub) if n else np.nan, k_prior=parent_k[p], k=kg))
    info = pd.DataFrame(info)
    info['beta'] = np.exp(-info['k'])
    info['f30_100'] = frac(info['k'].values, 30.0)
    info['f20_100'] = frac(info['k'].values, 20.0)
    return params, info


def literature_params():
    out = {}
    for g in PARENT:
        jj = GROUP_TO_JJ.get(g)
        out[g] = JJ2000_K[jj] if jj else np.nan
    return out


def harmonize(df, params, keep_measured=True, suffix='_h'):
    """Fill SOC100 / Deep for profiles that only have SC30."""
    df = df.copy()
    k = df['HGROUP'].map(params).astype(float)
    f30 = frac(k.values, 30.0)
    soc100_pred = df['SC30'].values / f30
    df['k_used'] = k.values
    df['f30_100'] = f30
    df['SOC100_jj'] = soc100_pred                      # model value for every row
    if keep_measured:
        use_meas = df['measured100'].values
        df[f'SOC100{suffix}'] = np.where(use_meas, df['SOC100'], soc100_pred)
        df['SOC100_src'] = np.where(use_meas, 'measured', 'harmonized')
    else:
        df[f'SOC100{suffix}'] = soc100_pred
        df['SOC100_src'] = 'harmonized'
    df[f'SOC100{suffix}'] = np.maximum(df[f'SOC100{suffix}'], df['SC30'])
    df[f'DeepSOC30_100{suffix}'] = df[f'SOC100{suffix}'] - df['SC30']
    return df


def bootstrap_uncertainty(df, n_boot=500, k0=10.0, seed=42, suffix='_h'):
    """Bootstrap the calibration profiles → sd / 90 % interval of the
    harmonised SOC100 for every harmonised row (measured rows get 0)."""
    rng = np.random.default_rng(seed)
    cal_idx = np.where(df['calib'].values)[0]
    groups = df['HGROUP'].values
    sc30 = df['SC30'].values
    sims = np.empty((n_boot, len(df)), dtype=np.float32)
    ks = []
    for b in range(n_boot):
        bi = rng.choice(cal_idx, size=len(cal_idx), replace=True)
        boot = df.iloc[bi]
        p, _ = fit_params(boot, k0=k0)
        ks.append(p)
        k = np.array([p[g] for g in groups])
        sims[b] = sc30 / frac(k, 30.0)
    sd = sims.std(0)
    lo, hi = np.percentile(sims, [5, 95], axis=0)
    harm = df['SOC100_src'].values == 'harmonized'
    df = df.copy()
    df[f'SOC100{suffix}_sd'] = np.where(harm, sd, 0.0)
    df[f'SOC100{suffix}_p05'] = np.where(harm, lo, df[f'SOC100{suffix}'])
    df[f'SOC100{suffix}_p95'] = np.where(harm, hi, df[f'SOC100{suffix}'])
    return df, pd.DataFrame(ks)


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation on the paired profiles (repeated K-fold)
# ═══════════════════════════════════════════════════════════════════════════════
def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 3:
        return dict(n=len(y), r2=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan, ccc=np.nan)
    res = y - p
    r2 = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
    ccc = 2 * np.cov(y, p, ddof=0)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    return dict(n=len(y), r2=r2, rmse=np.sqrt(np.mean(res ** 2)),
                mae=np.mean(np.abs(res)), bias=np.mean(p - y), ccc=ccc)


def validate(df, n_splits=10, n_repeats=10, k0=10.0, seed=42):
    """Out-of-fold SOC100 for every calibration profile under 3 schemes:
    'local' (group-calibrated JJ), 'literature' (global JJ2000 β),
    'single' (one β for all profiles)."""
    from sklearn.model_selection import KFold
    cal = df[df['calib']].copy()
    idx = np.arange(len(cal))
    oof = {s: np.zeros(len(cal)) for s in ('local', 'single')}
    cnt = np.zeros(len(cal))
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(idx):
            p, _ = fit_params(cal.iloc[tr], k0=k0)
            k_te = cal.iloc[te]['HGROUP'].map(p).values
            oof['local'][te] += cal.iloc[te]['SC30'].values / frac(k_te, 30.0)
            k_single = np.median(cal.iloc[tr]['k_obs'])
            oof['single'][te] += cal.iloc[te]['SC30'].values / frac(
                np.full(len(te), k_single), 30.0)
            cnt[te] += 1
    for s in oof:
        oof[s] /= np.maximum(cnt, 1)
    lit = literature_params()
    k_lit = cal['HGROUP'].map(lit).values
    oof['literature'] = cal['SC30'].values / frac(k_lit, 30.0)
    out = cal[['HGROUP', 'SC30', 'SOC100']].copy()
    for s, v in oof.items():
        out[f'SOC100_{s}'] = v
    rows = []
    for subset, mask in [('All profiles', np.ones(len(out), bool)),
                         ('Excl. mangroves', out['HGROUP'].ne('Mangrove').values)]:
        for s in ('local', 'literature', 'single'):
            for tgt in ('SOC100', 'DeepSOC'):
                y = out['SOC100'].values[mask]
                p = out[f'SOC100_{s}'].values[mask]
                if tgt == 'DeepSOC':
                    y = y - out['SC30'].values[mask]
                    p = p - out['SC30'].values[mask]
                rows.append(dict(subset=subset, scheme=s, target=tgt, **_metrics(y, p)))
    for g in sorted(out['HGROUP'].unique()):
        m = out['HGROUP'].eq(g).values
        rows.append(dict(subset=g, scheme='local', target='SOC100',
                         **_metrics(out['SOC100'].values[m], out['SOC100_local'].values[m])))
    return out, pd.DataFrame(rows)


def run_harmonization(df, k0=10.0, calibrate_on='LU', n_boot=500, seed=42):
    """Convenience: prepare → fit → harmonise → bootstrap. Returns (table, info)."""
    t = prepare_table(df, calibrate_on=calibrate_on)
    params, info = fit_params(t, k0=k0)
    t = harmonize(t, params)
    if n_boot:
        t, _ = bootstrap_uncertainty(t, n_boot=n_boot, k0=k0, seed=seed)
    return t, params, info
