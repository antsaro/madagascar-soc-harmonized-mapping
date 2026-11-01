# ═══════════════════════════════════════════════════════════════════════════════
#  soc_ml.py — data loading, encoders, GPU model factory, fold-wise harmonisation
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import soc_config as C
import soc_harmonize as H


# ── GPU availability ──────────────────────────────────────────────────────────
def gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:                                            # noqa: BLE001
        return False


# ── Data ──────────────────────────────────────────────────────────────────────
def load_harmonized_table(path=C.HARM_TABLE):
    df = pd.read_csv(path).replace(-9999, np.nan)
    df = df[df['SC30'].notna() & (df['SC30'] < C.SOC_UPPER_LIMIT)]
    df = df[~(df['SOC100'] >= C.SOC_UPPER_LIMIT)].reset_index(drop=True)
    for col in C.NUMERIC_PREDICTORS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    encoders = {}
    for col in C.CATEGORICAL_PREDICTORS:
        df[col] = df[col].astype(str).fillna('Unknown')
        le = LabelEncoder().fit(df[col])
        df[f'{col}_enc'] = le.transform(df[col])
        encoders[col] = le
    features = C.NUMERIC_PREDICTORS + [f'{c}_enc' for c in C.CATEGORICAL_PREDICTORS]
    df['calib'] = df['calib'].astype(bool)
    df['measured100'] = df['measured100'].astype(bool)
    return df, features, encoders


def targets_for_fold(df, target, train_idx, test_idx, k0=C.SHRINK_K0):
    """Target vectors with β re-fitted on the training fold only (no leakage).
    Returns y_train, y_test_all (harmonised with train β), y_test_meas (NaN
    where the profile was not measured to 1 m), and the fold's group params."""
    if target == 'SOC30':
        y = df['SC30'].values
        return y[train_idx], y[test_idx], y[test_idx], None
    params, _ = H.fit_params(df.iloc[train_idx], k0=k0)
    h = H.harmonize(df, params)
    col = C.TARGETS[target]
    y_all = h[col].values
    meas = df['measured100'].values
    if target == 'SOC100':
        y_meas = np.where(meas, df['SOC100'].values, np.nan)
    else:
        y_meas = np.where(meas, np.maximum(df['SOC100'].values - df['SC30'].values, 0),
                          np.nan)
    return y_all[train_idx], y_all[test_idx], y_meas[test_idx], params


def full_target(df, target, k0=C.SHRINK_K0):
    if target == 'SOC30':
        return df['SC30'].values, None
    params, _ = H.fit_params(df, k0=k0)
    return H.harmonize(df, params)[C.TARGETS[target]].values, params


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model(model_name, params, use_gpu=True, seed=42):
    p = dict(params)
    if model_name == 'QRF':
        use_gpu = False
    if model_name == 'RandomForest':
        if use_gpu:
            from cuml.ensemble import RandomForestRegressor as cuRF
            return cuRF(**p, random_state=seed, n_streams=1)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**p, random_state=seed, n_jobs=-1)
    if model_name == 'QRF':
        from quantile_forest import RandomForestQuantileRegressor
        return RandomForestQuantileRegressor(**p, max_samples_leaf=None,
                                             default_quantiles='mean',
                                             random_state=seed, n_jobs=-1)
    if model_name == 'XGBoost':
        import xgboost as xgb
        return xgb.XGBRegressor(**p, random_state=seed, verbosity=0,
                                tree_method='hist',
                                device='cuda' if use_gpu else 'cpu')
    raise ValueError(model_name)


def to_sklearn_rf(params, seed=42):
    """Same hyper-parameters on sklearn (needed by shap.TreeExplainer)."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**params, random_state=seed, n_jobs=-1)


def is_qrf(model):
    return type(model).__name__ == 'RandomForestQuantileRegressor'


def predict(model, X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if is_qrf(model):
        return np.asarray(model.predict(X, quantiles='mean'), dtype=np.float64).ravel()
    out = model.predict(X)
    try:
        import cupy as cp
        if isinstance(out, cp.ndarray):
            out = cp.asnumpy(out)
    except ImportError:
        pass
    if hasattr(out, 'to_numpy'):
        out = out.to_numpy()
    return np.asarray(out, dtype=np.float64).ravel()


def predict_quantiles(model, X, quantiles):
    """QRF conditional quantiles, shape (n, len(quantiles)), monotone per row."""
    q = model.predict(np.ascontiguousarray(X, dtype=np.float32), quantiles=list(quantiles))
    q = np.asarray(q, dtype=np.float64).reshape(len(X), -1)
    return np.maximum.accumulate(q, axis=1)


def interval_scores(y, lo, hi):
    """PICP (coverage) and MPIW (mean width) of [lo, hi]."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.nan, np.nan
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))), float(np.mean(hi[m] - lo[m]))


def suggest_params(trial, model_name):
    if model_name in ('RandomForest', 'QRF'):
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 5, 24),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 15),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 8),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
        )
    return dict(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 2.0),
        reg_lambda=trial.suggest_float('reg_lambda', 0.5, 5.0),
    )


def calc_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    keys = ('rmse', 'r2', 'mae', 'bias', 'rpd', 'rpiq', 'ccc')
    if len(y_true) < 3:
        return dict(n=len(y_true), **{k: np.nan for k in keys})
    res = y_true - y_pred
    rmse = float(np.sqrt(np.mean(res ** 2)))
    sst = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum(res ** 2) / sst) if sst > 0 else np.nan
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    ccc_den = y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
    ccc = 2 * np.cov(y_true, y_pred, ddof=0)[0, 1] / ccc_den if ccc_den > 0 else np.nan
    return dict(n=int(len(y_true)), rmse=rmse, r2=r2, mae=float(np.mean(np.abs(res))),
                bias=float(np.mean(y_pred - y_true)),
                rpd=float(y_true.std() / rmse) if rmse > 0 else np.nan,
                rpiq=float(iqr / rmse) if rmse > 0 else np.nan, ccc=float(ccc))


def results_dir(target, cv='random'):
    import os
    return os.path.join(C.RESULTS_DIR, f'results_{target}' + ('_spatialCV' if cv == 'spatial' else ''))
