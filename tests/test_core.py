"""Fast unit tests for the depth model and the spatial blocks (pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import soc_harmonize as H  # noqa: E402
import soc_spatial as S    # noqa: E402
import soc_ml as M         # noqa: E402


def test_uniform_profile_limit():
    assert H.frac(0.0, 30.0) == pytest.approx(0.30)
    assert H.frac(1e-12, 30.0) == pytest.approx(0.30)


def test_frac_matches_beta_form():
    k = 0.025
    beta = np.exp(-k)
    assert H.frac(k, 30.0) == pytest.approx((1 - beta ** 30) / (1 - beta ** 100))


@pytest.mark.parametrize('r', [0.25, 0.30, 0.45, 0.60, 0.85])
def test_inversion_round_trip(r):
    k = H.k_from_ratio([r])[0]
    assert H.frac(k, 30.0) == pytest.approx(r, abs=1e-8)


def test_jobbagy_priors():
    # shares of 0-100 cm SOC in 0-20 cm reported by Jobbagy & Jackson (2000)
    for g, f20 in H.JJ2000_F20.items():
        assert H.frac(H.JJ2000_K[g], 20.0) == pytest.approx(f20, abs=1e-8)
    assert H.JJ2000_K['Forest'] > H.JJ2000_K['Grassland'] > H.JJ2000_K['Shrubland']


def _toy_table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    lu = rng.choice(['Forest', 'Grassland', 'Crop', 'undetermined'], n)
    sc30 = rng.lognormal(4, 0.3, n)
    f30 = np.where(lu == 'Forest', 0.55, 0.60)
    soc100 = np.where(rng.random(n) < 0.5, sc30 / f30, np.nan)
    return pd.DataFrame(dict(LU=lu, lulc='Grassland', SC30=sc30, SOC100=soc100))


def test_harmonize_keeps_measured_and_is_consistent():
    t, params, info = H.run_harmonization(_toy_table(), n_boot=0)
    meas = t['measured100']
    assert np.allclose(t.loc[meas, 'SOC100_h'], np.maximum(t.loc[meas, 'SOC100'], t.loc[meas, 'SC30']))
    assert (t['SOC100_h'] >= t['SC30']).all()
    assert np.allclose(t['DeepSOC30_100_h'], t['SOC100_h'] - t['SC30'])
    assert H.frac(params['Forest'], 30.0) == pytest.approx(0.55, abs=0.03)


def test_shrinkage_pulls_small_groups():
    t = H.prepare_table(_toy_table())
    p, info = H.fit_params(t, k0=10)
    crop = info[(info.level == 'group') & (info.group == 'Cropland')].iloc[0]
    lo, hi = sorted([crop.k_local, crop.k_prior])
    assert lo <= crop.k <= hi


def test_hex_blocks_and_folds():
    rng = np.random.default_rng(1)
    xy = np.c_[rng.uniform(300, 1000, 2000), rng.uniform(7100, 8700, 2000)]
    keys = S.block_keys(xy, 'hexagon', 125)
    assert keys.shape == (2000, 2)
    splits, folds = S.spatial_splits(xy, 'hexagon', 125, 10)
    assert len(splits) == 10
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0
        # no block is split between train and test
        test_blocks = {tuple(k) for k in folds['keys'][te]}
        train_blocks = {tuple(k) for k in folds['keys'][tr]}
        assert not test_blocks & train_blocks


def test_qrf_quantiles_and_interval_scores():
    pytest.importorskip('quantile_forest')
    rng = np.random.default_rng(2)
    X = rng.random((600, 3)).astype(np.float32)
    y = 10 * X[:, 0] + rng.normal(0, 1, 600)
    params = dict(n_estimators=100, max_depth=None, min_samples_split=2,
                  min_samples_leaf=5, max_features=1.0)
    m = M.make_model('QRF', params, use_gpu=True).fit(X[:400], y[:400])
    q = M.predict_quantiles(m, X[400:], [0.05, 0.5, 0.95])
    assert q.shape == (200, 3)
    assert (np.diff(q, axis=1) >= 0).all()
    picp, mpiw = M.interval_scores(y[400:], q[:, 0], q[:, 2])
    assert 0.75 < picp <= 1.0 and mpiw > 0
    assert M.predict(m, X[400:]).shape == (200,)
