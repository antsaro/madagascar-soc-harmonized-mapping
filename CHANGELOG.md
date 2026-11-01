# Changelog

## 1.1.0 (2026-11-01)

- Quantile regression forest (QRF, `quantile-forest`) added as a third learner,
  tuned and evaluated in both nested CV schemes (CPU, in parallel with the GPU
  queues). Conditional quantiles of every held-out profile are stored and the
  coverage (PICP) and width (MPIW) of the 50/80/90/95 % intervals are reported.
- New step `07_qrf_interval_maps.py` / `src/map_qrf_intervals.py`: QRF prediction
  with interval rasters (mean, q05, q50, q95, 90 % interval width, relative width)
  and an optional coarsening factor.
- New figures: QRF interval coverage (Fig. S7) and interval maps (Fig. 8). All
  performance and SHAP figures now show the three models.
- Raster reading moved to `src/soc_raster.py`, shared by both mapping scripts.

## 1.0.0 (2026-11-01)

First public release.

- Depth harmonisation of SOC 0–100 cm and 30–100 cm with the Jobbágy & Jackson
  (2000) exponential model, calibrated per vegetation group with hierarchical
  shrinkage, bootstrap uncertainty and out-of-fold validation.
- Nested cross-validation on GPU for Random Forest (cuML) and XGBoost, with two
  schemes: random K-fold (10 × 10 outer, 4 inner) and spatial hexagonal blocks
  (10 folds × 4 grid positions outer, 4 spatial inner folds). Group decay rates are
  re-fitted inside each outer training fold.
- Out-of-fold SHAP values, beeswarm, SHAP dependence with centred partial
  dependence, SHAP by land-cover and soil class.
- Bootstrap ensemble maps (mean, SD, CV) predicted tile by tile on GPU, with
  harmonisation uncertainty propagated and national stock totals.
- Fix: categorical rasters are now translated from raster code to class name to
  encoder index. An earlier internal version set every pixel's land cover and
  soil code to 0.
