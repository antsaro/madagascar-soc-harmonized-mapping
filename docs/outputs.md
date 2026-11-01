# Output files

Paths are relative to `SOC_WORK`. `<target>` is `SOC30`, `DeepSOC` (30–100 cm) or
`SOC100`; `<model>` is `RandomForest`, `XGBoost` or `QRF`.

## Harmonization/

**`SOC_harmonized_table.csv`**: the input table plus

| Column | Meaning |
|---|---|
| `HGROUP` | harmonisation group |
| `GROUP_SRC` | where the group came from: `LU`, `ESA` or `default` |
| `PARENT` | woody, herbaceous or mangrove |
| `measured100` | SOC100 was measured |
| `ratio30_100` | SC30 / SOC100 for measured profiles |
| `k_obs` | decay rate of the measured profile (cm⁻¹) |
| `calib` | used for calibration |
| `k_used`, `f30_100` | group decay rate and 0–30 cm share applied to the profile |
| `SOC100_jj` | model value SC30 / f30 for every profile, measured or not |
| `SOC100_h` | measured SOC100, or the harmonised value |
| `DeepSOC30_100_h` | `SOC100_h − SC30` |
| `SOC100_src` | `measured` or `harmonized` |
| `SOC100_h_sd`, `SOC100_h_p05`, `SOC100_h_p95` | bootstrap SD and 90 % interval (0 / measured value for measured profiles) |

**`jobbagy_group_parameters.csv`**: one row per parent and group: number of
calibration profiles `n`, local median `k_local`, prior `k_prior`, final `k`,
`beta`, `f30_100`, `f20_100`, bootstrap 90 % interval of `f30` (`f30_p05`,
`f30_p95`), number of profiles harmonised with it, and the JJ2000 `f30` for
comparison.

**`harmonization_validation.csv`**: out-of-fold metrics (`n`, `r2`, `rmse`, `mae`,
`bias`, `ccc`) per subset (all profiles, without mangroves, each group), scheme
(`local`, `literature`, `single`) and target (SOC100, DeepSOC).

**`harmonization_oof.csv`**: out-of-fold SOC100 of each calibration profile under
the three schemes.

## ML_SOC_Harmonized/results_<target>[ _spatialCV]/

| File | Content |
|---|---|
| `cv_predictions_<model>.csv` | every outer prediction: `row_id`, `repeat`, `fold`, `predicted`, `observed_all`, `observed_meas` (empty if not measured to 1 m), `source`, `HGROUP`, `x_km`, `y_km`, `dist_to_train_km` |
| `oof_mean_<model>.csv` | per profile: mean and SD of its out-of-fold predictions across repeats |
| `fold_metrics_<model>.csv` | metrics per outer fold, `meas_*` (measured profiles) and `all_*`, plus `median_test_train_km` |
| `best_params_per_fold_<model>.csv` | tuned hyper-parameters and inner RMSE per outer fold |
| `summary_<model>.json` | fold mean ± SD and pooled metrics, final parameters, run time |
| `models/<model>.pkl` | final model bundle: model, parameters, feature list, encoders, harmonisation parameters |
| `models/XGBoost.ubj` | native XGBoost model file |
| `spatial_folds.csv` | spatial CV only: coordinates and the fold of every profile for each grid position |
| `qrf_interval_coverage.csv` | QRF only: PICP (%) and MPIW of the 50/80/90/95 % central intervals, measured and all profiles |

For the QRF, `cv_predictions_QRF.csv` has extra columns `q0.025` … `q0.975` (the
conditional quantiles of each held-out profile) and `fold_metrics_QRF.csv` has
`meas_picp90`, `meas_mpiw90`, `all_picp90`, `all_mpiw90`.

Also in `ML_SOC_Harmonized/`: `Table_model_performance.csv`,
`random_vs_spatialCV.csv`, `Table_QRF_interval_coverage.csv`, `logs/`.

## ML_SOC_Harmonized/SHAP/

| File | Content |
|---|---|
| `shap_importance_<model>_<target>.csv` | mean \|SHAP\| and share per feature |
| `shap_values_<model>_<target>.npz` | arrays `shap`, `X`, `row_id`, `base`, `features` |

## Maps_Harmonized/<target>_<model>/ (Random Forest, XGBoost)

| File | Content |
|---|---|
| `<target>_<model>_mean.tif` | bootstrap mean, Mg C ha⁻¹ |
| `<target>_<model>_sd.tif` | bootstrap SD, Mg C ha⁻¹ |
| `<target>_<model>_cv.tif` | coefficient of variation, % |
| `bootstrap_diagnostics.csv` | out-of-bag R² and RMSE and decay rates used by each replicate |
| `boot_models/` | the bootstrap models (can be deleted after mapping) |
| `map_stats.json` | national stock (Pg C), upper-bound SD, mapped area, mean density |

## Maps_Harmonized/<target>_QRF/

| File | Content |
|---|---|
| `<target>_QRF_mean.tif` | conditional mean, Mg C ha⁻¹ |
| `<target>_QRF_q05.tif` | 5th percentile (lower bound of the 90 % PI) |
| `<target>_QRF_q50.tif` | median |
| `<target>_QRF_q95.tif` | 95th percentile (upper bound) |
| `<target>_QRF_piw90.tif` | interval width q95 − q05, Mg C ha⁻¹ |
| `<target>_QRF_rpiw90.tif` | relative width, % of the median |
| `map_stats.json` | national total from the mean map, sums of the pixel q05 / q95 (bounds, not quantiles of the total), area, decimation factor |

The interval bounds follow `QRF_PI` in `soc_config.py`; with other values the file
suffixes change accordingly (e.g. `q10`, `q90`, `piw80`).
