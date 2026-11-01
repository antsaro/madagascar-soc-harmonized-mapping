#!/usr/bin/env bash
# End-to-end run on synthetic data with tiny settings (CPU is fine, ~10-20 min).
# Usage: bash tests/smoke_test.sh [work_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/_smoke}"

python "$ROOT/tests/make_synthetic_data.py" "$WORK"

export SOC_WORK="$WORK"
export SOC_PRED_CSV="$WORK/SOC30_predictors_table.csv"
export SOC_CORR_CSV="none"
export SOC_RASTER_DIR="$WORK/rasters"
export SOC_GEOJSON="$WORK/outline.geojson"
export SOC_FAST=1
export MPLBACKEND=Agg

cd "$ROOT"
python pipeline/01_harmonize_depth.py
python pipeline/02_run_nested_cv.py
python pipeline/03_performance_figures.py
python pipeline/04_spatial_fold_maps.py
python pipeline/05_shap_pdp.py
python pipeline/06_gpu_maps.py
python pipeline/07_qrf_interval_maps.py

echo "Smoke test finished. Outputs in $WORK"
