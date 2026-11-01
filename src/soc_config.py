# ═══════════════════════════════════════════════════════════════════════════════
#  soc_config.py — single place for paths & settings (edit here only)
# ═══════════════════════════════════════════════════════════════════════════════
import os

WORK = os.environ.get('SOC_WORK', '/kaggle/working')

# ── Inputs ────────────────────────────────────────────────────────────────────
# Full point table with covariates (all 4,184 points; SC30 + SOC100 where measured)
PREDICTORS_CSV = os.environ.get('SOC_PRED_CSV', f'{WORK}/SOC30_predictors_table.csv')
# Hand-corrected rows where SOC100 < SC30 (set to None if not used)
CORRECTIONS_CSV = os.environ.get(
    'SOC_CORR_CSV',
    '/kaggle/input/datasets/antsasarobidyran/deepsoc-data-corrected/'
    'DeepSOC_invalid_negatives_corrected.csv')

# ── Outputs ───────────────────────────────────────────────────────────────────
HARM_DIR = f'{WORK}/Harmonization'
HARM_TABLE = f'{HARM_DIR}/SOC_harmonized_table.csv'
HARM_PARAMS = f'{HARM_DIR}/jobbagy_group_parameters.csv'
RESULTS_DIR = f'{WORK}/ML_SOC_Harmonized'
FIG_DIR = f'{RESULTS_DIR}/Figures'
MAP_DIR = f'{WORK}/Maps_Harmonized'

# ── Harmonisation settings ────────────────────────────────────────────────────
SHRINK_K0 = 10.0         # pseudo-count pulling small groups to the parent / JJ2000
CALIBRATE_ON = 'LU'      # 'LU' = only profiles with a field land-use label
N_BOOT_HARM = 500

# ── Targets (column names produced by the harmonisation step) ─────────────────
TARGETS = {
    'SOC30':   'SC30',
    'DeepSOC': 'DeepSOC30_100_h',
    'SOC100':  'SOC100_h',
}
SOC_UPPER_LIMIT = 1000

# ── Predictors ────────────────────────────────────────────────────────────────
NUMERIC_PREDICTORS = ['NDVI', 'NDWI', 'NIRI', 'elevation', 'slope',
                      'mat', 'map', 'treecover']
CATEGORICAL_PREDICTORS = ['lulc', 'soil_type']

# ── Models ────────────────────────────────────────────────────────────────────
#   RandomForest (cuML, GPU) · XGBoost (GPU) · QRF = quantile regression forest
#   (quantile-forest package, CPU; point prediction = conditional mean)
MODELS = ['RandomForest', 'XGBoost', 'QRF']
QRF_OPTUNA_TRIALS = 30               # QRF is tuned on CPU → fewer trials by default
QRF_QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
QRF_PI = (0.05, 0.95)                # interval mapped as rasters (90 % PI)

# ── Nested CV (same design as the original notebook) ──────────────────────────
N_OUTER_FOLDS = 10
N_OUTER_REPEATS = 10
N_INNER_FOLDS = 4
OPTUNA_TRIALS = 100

# ── Nested SPATIAL block CV (from soc_spatial_cv.ipynb) ───────────────────────
CV_SCHEMES = ['random', 'spatial']
LON_COL, LAT_COL = 'Longitude', 'Latitude'
SPATIAL_DESIGN = {                   # (block shape, flat-to-flat size in km)
    'SOC30':   ('hexagon', 125),
    'DeepSOC': ('hexagon', 150),     # 150 km from the block-design analysis of
    'SOC100':  ('hexagon', 150),     # the 1,520 profiles measured to 1 m
}
N_SPATIAL_REPEATS = 4                # grid positions (half-block shifts), 1–4
# For DeepSOC / SOC100, balance outer folds on the measured 1 m profiles
# (validation set) rather than on all plots: weight measured = 1, harmonised = w
SPATIAL_HARMONIZED_WEIGHT = 0.1
GEOJSON = os.environ.get('SOC_GEOJSON') or '/kaggle/input/datasets/antsasarobidyran/madaforblockcv/madagascar_ne10m.geojson'
FINAL_MODEL_CV = 'random'            # which tuned final models feed SHAP & maps

# ── Rasters for mapping ───────────────────────────────────────────────────────
_K = '/kaggle/input/datasets/antsasarobidyran'
RASTERS = {
    'NDVI':      f'{_K}/merged-covariates-geotiff-v2/NDVI.tif',
    'NDWI':      f'{_K}/merged-covariates-geotiff-v2/NDWI.tif',
    'NIRI':      f'{_K}/merged-covariates-geotiff-v2/NIRI.tif',
    'elevation': f'{_K}/merged-covariates-geotiff-v6/ELEV_ALOS.tif',
    'lulc':      f'{_K}/merged-covariates-geotiff-v3/lulc.tif',
    'map':       f'{_K}/merged-covariates-geotiff-v3/map_corrected.tif',
    'mat':       f'{_K}/merged-covariates-geotiff-v3/mat_corrected.tif',
    'slope':     f'{_K}/merged-covariates-geotiff-v6/SLOPE_ALOS.tif',
    'soil_type': f'{_K}/merged-covariates-geotiff-v3/soil_type.tif',
    'treecover': f'{_K}/merged-covariates-geotiff-v3/treecover.tif',
}
if os.environ.get('SOC_RASTER_DIR'):      # local testing override
    RASTERS = {k: f"{os.environ['SOC_RASTER_DIR']}/{k}.tif" for k in RASTERS}
MAT_DIVIDE_BY_10 = True                  # raster MAT stored ×10
ZERO_IS_NODATA = ('mat', 'map')

# Raster integer code → class name used in the training table
SOIL_CLASSES = {
    1: 'Andosols', 2: 'Bare Rock', 3: 'Ferrallitic Soils', 4: 'Ferruginous Soils',
    5: 'Fersiallitic Soils', 6: 'Hydromorphic Soils', 7: 'Lithic Raw Mineral Soils',
    8: 'Alluvial Weakly Developed Soils',
    9: 'Erosional Weakly Developed Soils / Lithosols',
    10: 'Podzolic Soils and Podzols', 11: 'Vertisols',
}
LULC_CLASSES = {
    10: 'Tree Cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
    50: 'Built-up', 60: 'Bare / Sparse Vegetation', 70: 'Snow and Ice',
    80: 'Permanent Water Bodies', 90: 'Herbaceous Wetland', 95: 'Mangroves',
    100: 'Moss and Lichen',
}
RASTER_CLASS_MAPS = {'lulc': LULC_CLASSES, 'soil_type': SOIL_CLASSES}
# Pixels whose class never occurs in the training data:
#   'mask' → NoData in the map,  'mode' → most frequent training class
UNSEEN_CLASS_POLICY = {'lulc': 'mask', 'soil_type': 'mode'}
MASK_LULC_CODES = (80,)                  # permanent water → always NoData

# ── Quick test mode (SOC_FAST=1): tiny settings for the smoke test ──────────
FAST = os.environ.get('SOC_FAST') == '1'
if FAST:
    N_BOOT_HARM, N_OUTER_FOLDS, N_OUTER_REPEATS = 20, 3, 1
    N_INNER_FOLDS, OPTUNA_TRIALS, N_SPATIAL_REPEATS = 2, 2, 1
    QRF_OPTUNA_TRIALS = 2

# ── Mapping (bootstrap) ───────────────────────────────────────────────────────
N_BOOT_MAP = 3 if FAST else 100

# ── QRF prediction-interval maps (CPU) ────────────────────────────────────────
QRF_MAP_DECIM = 1                    # >1 predicts on a coarser grid (factor)
QRF_MAP_TILE = 2048                  # output tile size (pixels)
QRF_MAP_BATCH = 250_000              # pixels per predict call
TILE_SIZE = 4096
PRED_BATCH = 2_000_000
NODATA_OUT = -9999.0
