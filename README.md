# Depth-harmonised soil organic carbon mapping for Madagascar

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23857981.svg)](https://doi.org/10.5281/zenodo.23857981)

Code for mapping soil organic carbon (SOC) stocks in Madagascar for three depth
intervals, 0–30 cm, 30–100 cm and 0–100 cm, from a national compilation of soil
profiles. Most profiles in the compilation were only sampled to 30 cm, so the deeper
stocks are completed with the Jobbágy & Jackson (2000) exponential depth model,
calibrated on the profiles that were measured to 1 m. Three learners, Random
Forest, XGBoost and a quantile regression forest (QRF), are then tuned and
evaluated with two nested cross-validation schemes (random K-fold and spatial
blocks) and interpreted with SHAP and partial dependence. National maps come with
two kinds of uncertainty: bootstrap mean/SD/CV maps for Random Forest and XGBoost,
and 90 % prediction-interval rasters (lower bound, median, upper bound, interval
width) from the QRF. Random Forest and XGBoost training, SHAP for XGBoost and the
bootstrap maps run on GPU; the QRF runs on CPU.

**Authors:** Antsa Sarobidy Randrianantenaina and Nandrianina Ramifehiarivo<br>
**License:** MIT

---

## Contents

1. [What the pipeline does](#what-the-pipeline-does)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Input data](#input-data)
6. [How to run](#how-to-run)
7. [Methods](#methods)
   - [7.1 Depth harmonisation](#71-depth-harmonisation-jobbágy--jackson-2000)
   - [7.2 Predictors and models](#72-predictors-and-models)
   - [7.3 Nested cross-validation](#73-nested-cross-validation)
   - [7.4 Accuracy metrics](#74-accuracy-metrics)
   - [7.5 Model interpretation: SHAP and partial dependence](#75-model-interpretation-shap-and-partial-dependence)
   - [7.6 Mapping and uncertainty](#76-mapping-and-uncertainty)
   - [7.7 Quantile regression forest and prediction intervals](#77-quantile-regression-forest-and-prediction-intervals)
8. [Outputs](#outputs)
9. [Configuration](#configuration)
10. [Run time](#run-time)
11. [Limitations](#limitations)
12. [How to cite](#how-to-cite)
13. [References](#references)

---

## What the pipeline does

| Step | Script | What happens |
|---|---|---|
| 1 | `pipeline/01_harmonize_depth.py` | Assigns every profile to a vegetation group, fits the Jobbágy depth model per group on the paired 0–30 / 0–100 cm profiles, fills SOC 0–100 and 30–100 cm where they were not measured, quantifies the uncertainty by bootstrap and validates the procedure out of fold. |
| 2 | `pipeline/02_run_nested_cv.py` | Launches the nested cross-validation for the three models, both CV schemes and the three depths. Random Forest runs on GPU 0, XGBoost on GPU 1 and the QRF on the CPU, all three at the same time. |
| 3 | `pipeline/03_performance_figures.py` | Performance table, observed vs predicted plots for each CV scheme, random vs spatial comparison, coverage of the QRF prediction intervals. |
| 4 | `pipeline/04_spatial_fold_maps.py` | Maps of the spatial blocks and folds that were used, and the distribution of test-to-training distances. |
| 5 | `pipeline/05_shap_pdp.py` | Out-of-fold SHAP values, importance, beeswarm, SHAP dependence with partial dependence overlaid, SHAP by land-cover and soil class. |
| 6 | `pipeline/06_gpu_maps.py` | Random Forest and XGBoost bootstrap ensembles predicted tile by tile on GPU: mean, standard deviation and coefficient of variation rasters, national stock totals, figures. |
| 7 | `pipeline/07_qrf_interval_maps.py` | QRF prediction with interval rasters: conditional mean, 5th, 50th and 95th percentiles, 90 % interval width and relative width, national totals, figure. |

The same steps are available as a single Kaggle notebook in
`notebooks/SOC_harmonized_GPU_pipeline.ipynb`.

---

## Repository layout

```
madagascar-soc-harmonized-mapping/
├── src/
│   ├── soc_config.py        # all paths and settings (the only file you should need to edit)
│   ├── soc_harmonize.py     # Jobbágy depth model, calibration, harmonisation, validation
│   ├── soc_spatial.py       # hexagonal block grid and spatial fold assignment
│   ├── soc_ml.py            # data loading, encoders, model factory, fold-wise targets, metrics
│   ├── soc_raster.py        # aligned tiled reading of the covariate rasters, masks, pixel areas
│   ├── train_soc_harm.py    # nested CV for one target / model / CV scheme (CLI)
│   ├── map_soc_gpu.py       # RF / XGBoost bootstrap training + tiled GPU prediction (CLI)
│   ├── map_qrf_intervals.py # QRF prediction-interval rasters (CLI)
│   └── nature_style.py      # figure style (Arial, Nature-like sizes and palette)
├── pipeline/                # steps 01 to 07, run in order
├── notebooks/
│   └── SOC_harmonized_GPU_pipeline.ipynb
├── tests/
│   ├── test_core.py         # unit tests (depth model, shrinkage, spatial blocks)
│   ├── make_synthetic_data.py
│   └── smoke_test.sh        # whole pipeline on synthetic data, tiny settings
├── .github/workflows/tests.yml
├── data/README.md           # expected input format (data are not distributed here)
├── docs/outputs.md          # every output file, column by column
├── requirements.txt
├── requirements-gpu.txt
├── environment.yml
├── CITATION.cff
├── .zenodo.json
├── CHANGELOG.md
└── LICENSE
```

---

## Requirements

Python 3.10 or newer. The CPU code was tested with the versions below; older minor
versions of the same packages will most likely work but were not checked.

| Package | Tested version | Used for |
|---|---|---|
| numpy | 2.5.3 | arrays |
| pandas | 3.0.5 | tables |
| scipy | 1.18.1 | root finding, KD-trees, kernel density |
| scikit-learn | 1.9.1 | Random Forest (CPU / SHAP twin), K-fold, encoders |
| xgboost | 3.4.1 | gradient boosting, GPU TreeSHAP |
| optuna | 5.0.0 | hyper-parameter search (TPE + median pruning) |
| shap | 0.54.1 | TreeSHAP for Random Forest |
| statsmodels | 0.15.0 | LOWESS smoothing in dependence plots |
| matplotlib | 3.11.2 | figures |
| rasterio | 1.5.2 | reading covariates, writing GeoTIFFs |
| pyproj | 3.8.0 | lon/lat to UTM 38S for the spatial blocks |
| shapely | 2.2.0 | land polygon in the fold maps |
| joblib | 1.6.0 | model files, parallel SHAP folds |
| openpyxl | 3.1.5 | Excel export |
| quantile-forest | 1.4.2 | quantile regression forest |

GPU part (optional but strongly recommended for the real dataset):

| Package | Version | Notes |
|---|---|---|
| cuML (RAPIDS) | ≥ 24.06 | GPU Random Forest; the Kaggle GPU image ships it |
| CuPy | matching your CUDA | moving tiles to the GPU |
| CUDA | 12.x | |

The full run was designed for a Kaggle notebook with two NVIDIA T4 GPUs. Without a
GPU everything still runs (scikit-learn replaces cuML, XGBoost uses the CPU), only
much more slowly.

Fonts: figures use Arial. If Arial is not installed, `nature_style.py` tries the
Kaggle dataset `hammaadali/arial-font` and then falls back to Liberation Sans, which
has the same metrics.

---

## Installation

```bash
git clone https://github.com/<your-user>/madagascar-soc-harmonized-mapping.git
cd madagascar-soc-harmonized-mapping

# CPU only
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# or with conda, including RAPIDS cuML for the GPU
conda env create -f environment.yml
conda activate soc-harmonized
```

To check that everything works before touching the real data:

```bash
pytest -q tests/test_core.py      # a few seconds
bash tests/smoke_test.sh          # full pipeline on synthetic data
```

This builds a small synthetic dataset (900 points, fake rasters, an elliptical
"country") and runs all six steps with tiny settings. It takes 10 to 20 minutes on a
laptop. The numbers it produces are meaningless; it only checks the plumbing.

---

## Input data

The soil data are not included in this repository. `data/README.md` describes the
expected format in detail. In short:

**Point table** (`SOC30_predictors_table.csv`), one row per profile:

| Column | Description |
|---|---|
| `num` | unique profile id |
| `Longitude`, `Latitude` | WGS84 coordinates |
| `SC30` | SOC stock 0–30 cm (Mg C ha⁻¹), required |
| `SOC100` | SOC stock 0–100 cm (Mg C ha⁻¹), empty when not measured |
| `LU` | field land-use label (e.g. Forest, Grassland, Crop, Rice, Mangroves, undetermined) |
| `NDVI`, `NDWI`, `NIRI`, `elevation`, `slope`, `mat`, `map`, `treecover` | covariates extracted at the point (MAT in °C) |
| `lulc` | ESA WorldCover 2020 class name at the point |
| `soil_type` | soil class name at the point |

**Optional corrections file**: profiles where the reported SOC100 was lower than
SC30 were checked by hand; the corrected values are read from a CSV matched on `num`.

**Rasters**: one GeoTIFF per covariate, all on the same grid (same size, transform
and CRS). `lulc` and `soil_type` hold integer codes; the code-to-name tables are in
`soc_config.py`.

---

## How to run

### On Kaggle (recommended)

1. Create a notebook with the GPU T4 ×2 accelerator and attach the datasets holding
   the point table, the rasters and the country outline.
2. Upload `notebooks/SOC_harmonized_GPU_pipeline.ipynb` or copy its cells.
3. Check the paths in the `soc_config.py` cell, then run the cells in order.

### Locally

```bash
export SOC_WORK=/path/to/working_dir                 # outputs go here
export SOC_PRED_CSV=/path/to/SOC30_predictors_table.csv
export SOC_CORR_CSV=/path/to/corrections.csv         # or "none"
export SOC_RASTER_DIR=/path/to/rasters               # NDVI.tif, elevation.tif, ...
export SOC_GEOJSON=/path/to/madagascar.geojson

python pipeline/01_harmonize_depth.py
python pipeline/02_run_nested_cv.py
python pipeline/03_performance_figures.py
python pipeline/04_spatial_fold_maps.py
python pipeline/05_shap_pdp.py
python pipeline/06_gpu_maps.py
python pipeline/07_qrf_interval_maps.py
```

A single model can also be trained or mapped directly:

```bash
cd src
python train_soc_harm.py --target SOC100 --model XGBoost --cv spatial
python map_soc_gpu.py    --target SOC100 --model XGBoost --n-boot 100
python train_soc_harm.py --target SOC100 --model QRF --cv random
python map_qrf_intervals.py --target SOC100 --decim 1
```

`--target` is one of `SOC30`, `DeepSOC` (30–100 cm) or `SOC100`; `--model` is
`RandomForest`, `XGBoost` or `QRF`.

---

## Methods

### 7.1 Depth harmonisation (Jobbágy & Jackson 2000)

**Depth model.** Following Gale & Grigal (1987) and Jobbágy & Jackson (2000), the
cumulative fraction of SOC found between the surface and depth $`d`$ (cm) is

```math
Y(d) = 1 - \beta^{\,d}, \qquad 0 < \beta
```

Writing $`\beta = e^{-k}`$, where $`k`$ (cm⁻¹) is a decay rate, the share of the
0–100 cm stock held in the top $`d_t`$ centimetres is

```math
f_{d_t}(k) \;=\; \frac{Y(d_t)}{Y(100)} \;=\; \frac{1-\beta^{\,d_t}}{1-\beta^{\,100}} \;=\; \frac{\mathrm{expm1}(-k\,d_t)}{\mathrm{expm1}(-100\,k)}
```

For the 0–30 cm layer,

```math
f_{30}(k) = \frac{1-\beta^{30}}{1-\beta^{100}}
```

$`f_{30}`$ decreases monotonically with $`\beta`$. When $`k \to 0`$ the profile is
uniform and $`f_{30} \to 0.30`$. Values of $`\beta > 1`$ ($`k < 0`$) are allowed and
describe SOC that increases with depth ($`f_{30} < 0.30`$), which is what we observe
in mangroves and some hydromorphic soils. $`k`$ is bounded to $`[-0.05,\ 0.50]`$
cm⁻¹, i.e. $`\beta \in [0.61,\ 1.05]`$.

**Decay rate of a measured profile.** For each profile with both stocks measured,
the observed ratio

```math
r_i = \frac{\mathrm{SOC}_{0\text{-}30,\,i}}{\mathrm{SOC}_{0\text{-}100,\,i}}
```

is inverted numerically (Brent's method) to give the profile decay rate
$`k_i`$ such that $`f_{30}(k_i) = r_i`$. Profiles with $`r_i \notin [0.02,\ 0.98]`$
(including those where SOC100 ≤ SC30 even after correction) are kept as data but not
used for calibration. By default only profiles carrying a field land-use label are
used for calibration.

**Vegetation groups.** Field labels are merged into eight groups (Forest,
Agroforestry, Fallow, Shrubland, Grassland, Cropland, Rice/Wetland, Mangrove). When
the field label is missing ("undetermined"), the ESA WorldCover class at the point
is used instead. Groups belong to three parents: woody (Forest, Agroforestry,
Fallow, Shrubland), herbaceous (Grassland, Cropland, Rice/Wetland) and Mangrove.

**Priors from Jobbágy & Jackson (2000).** At the global scale, the share of the
first metre's SOC found in the top 20 cm was 50 % under forests, 42 % under
grasslands and 33 % under shrublands. Solving $`f_{20}(k) = 0.50,\ 0.42,\ 0.33`$ gives

| JJ2000 biome | $`f_{20}`$ | $`k`$ (cm⁻¹) | $`\beta`$ | implied $`f_{30}`$ |
|---|---|---|---|---|
| Forest | 0.50 | 0.0328 | 0.9677 | 0.651 |
| Grassland | 0.42 | 0.0241 | 0.9762 | 0.565 |
| Shrubland | 0.33 | 0.0146 | 0.9855 | 0.462 |

**Hierarchical estimate.** The group decay rate is the median of the profile
rates, shrunk towards the parent, and the parent towards its JJ2000 prior, with a
pseudo-count $`\kappa`$ (default 10):

```math
\hat{k}_{p} = \frac{n_p\,\tilde{k}_p + \kappa\,k_p^{\mathrm{JJ}}}{n_p + \kappa},
\qquad
\hat{k}_{g} = \frac{n_g\,\tilde{k}_g + \kappa\,\hat{k}_{p(g)}}{n_g + \kappa}
```

where $`\tilde{k}`$ is the median of $`k_i`$ in the group or parent and $`n`$ the
number of calibration profiles. A group with hundreds of profiles is essentially
estimated from its own data; a group with 13 profiles is pulled about 40 % of the way
towards its parent. The mangrove parent has no JJ2000 equivalent and uses its own
median.

**Filling the missing stocks.** For a profile $`i`$ in group $`g`$ without a 1 m
measurement,

```math
\widehat{\mathrm{SOC}}_{0\text{-}100,\,i} = \max\!\left(\frac{\mathrm{SOC}_{0\text{-}30,\,i}}{f_{30}(\hat{k}_g)},\ \mathrm{SOC}_{0\text{-}30,\,i}\right),
\qquad
\widehat{\mathrm{SOC}}_{30\text{-}100,\,i} = \widehat{\mathrm{SOC}}_{0\text{-}100,\,i} - \mathrm{SOC}_{0\text{-}30,\,i}
```

Measured values are always kept as they are.

**Uncertainty.** The calibration profiles are resampled with replacement
($`B = 500`$), the group rates re-estimated each time, and the 5th–95th percentiles
and standard deviation of $`\widehat{\mathrm{SOC}}_{0\text{-}100}`$ stored for every
harmonised profile.

**Validation.** The harmonisation is checked by repeated 10-fold cross-validation
(10 repeats) on the calibration profiles: rates are estimated on nine folds and
used to predict SOC 0–100 cm of the tenth from its SOC 0–30 cm. Three variants are
compared: the locally calibrated hierarchical model, the global JJ2000 values, and
a single rate for all profiles. Results are reported for all profiles and without
mangroves, since no mangrove profile needed harmonising.

### 7.2 Predictors and models

Eight continuous covariates (NDVI, NDWI, NIRI, elevation, slope, mean annual
temperature, mean annual precipitation, tree cover) and two categorical ones (ESA
WorldCover land cover, soil type, label-encoded) are used for all three targets.
Tree-based models are invariant to monotone rescaling, so no standardisation is
applied.

Three learners are compared: Random Forest (Breiman 2001; cuML on GPU,
scikit-learn on CPU), XGBoost (Chen & Guestrin 2016, `tree_method="hist"`,
`device="cuda"`) and a quantile regression forest (Meinshausen 2006; package
`quantile-forest`, Johnson 2024; see [7.7](#77-quantile-regression-forest-and-prediction-intervals)).
Hyper-parameters are tuned with Optuna (Akiba et al. 2019) using the TPE sampler and
median pruning over the inner folds, minimising the mean inner RMSE of the point
prediction. The QRF uses the Random Forest search space; because it runs on CPU it
gets 30 trials per search by default instead of 100.

| Model | Hyper-parameter | Search space |
|---|---|---|
| Random Forest | `n_estimators` | 100–1000 |
| | `max_depth` | 5–24 |
| | `min_samples_split` | 2–15 |
| | `min_samples_leaf` | 1–8 |
| | `max_features` | sqrt, log2, 1.0 |
| XGBoost | `n_estimators` | 100–1000 |
| | `max_depth` | 3–10 |
| | `learning_rate` | 0.01–0.30 (log) |
| | `subsample` | 0.6–1.0 |
| | `colsample_bytree` | 0.6–1.0 |
| | `reg_alpha` | 0–2 |
| | `reg_lambda` | 0.5–5 |

The three targets are $`\mathrm{SOC}_{0\text{-}30}`$ (measured everywhere),
$`\mathrm{SOC}_{30\text{-}100}`$ and $`\mathrm{SOC}_{0\text{-}100}`$ (measured or
harmonised). All 4,184 profiles are used for every target.

### 7.3 Nested cross-validation

Both schemes are nested: hyper-parameters are tuned on inner folds drawn only from
the outer training set, and the tuned model is then scored on the outer test fold.
After the CV, a final model is tuned on all data with the same inner design and
saved for SHAP and mapping.

**Random K-fold.** Outer: 10 folds × 10 repeats (100 outer fits). Inner: 4 folds,
100 Optuna trials.

**Spatial block CV.** Profiles are projected to UTM zone 38S (km) and grouped into
pointy-top hexagons on a fixed national grid with origin
$`(x_0, y_0) = (300, 7100)`$ km. For a hexagon width $`s`$ (flat to flat) and
radius $`\rho = s/\sqrt{3}`$, a point $`(x, y)`$ shifted by the grid offset gets the
fractional axial coordinates

```math
q = \frac{\tfrac{\sqrt{3}}{3}\,(x - x_0) - \tfrac{1}{3}\,(y - y_0)}{\rho},
\qquad
z = \frac{\tfrac{2}{3}\,(y - y_0)}{\rho}
```

which are rounded to the nearest hexagon with cube-coordinate rounding. Block sizes
come from a separate block-design analysis: $`s = 125`$ km for SOC 0–30 cm and
$`s = 150`$ km for SOC 30–100 and 0–100 cm. Whole blocks are assigned to 10 outer
folds, largest block first, always to the fold with the smallest current load. The
four repeats use the grid shifted by half a block: $`(0,0)`$, $`(s/2, 0)`$,
$`(0, s/2)`$, $`(s/2, s/2)`$. Inner folds (4) are built the same way from the outer
training plots only. If there are too few blocks to fill the folds, the block size
is reduced by 15 % until there are enough.

For SOC 30–100 and 0–100 cm the fold load counts measured 1 m profiles with weight
1 and harmonised profiles with weight 0.1, so that every test fold contains enough
independent validation data. For each test plot the distance to the nearest
training plot is stored.

**No leakage from the harmonisation.** For SOC 30–100 and 0–100 cm, the group decay
rates $`\hat{k}_g`$ are re-estimated inside each outer training fold, and only those
are used to build the training targets. Held-out profiles therefore never influence
their own harmonised value.

**Two validation sets.** For the deeper layers, metrics are reported (i) on the
held-out profiles that were actually measured to 1 m, which is the independent
validation, and (ii) on all held-out profiles, including harmonised ones.

### 7.4 Accuracy metrics

With observations $`y_i`$, predictions $`\hat{y}_i`$, $`i = 1 \dots n`$:

```math
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2},
\qquad
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i-\hat{y}_i\rvert,
\qquad
\mathrm{Bias} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
```

```math
R^2 = 1-\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2},
\qquad
\mathrm{RPD} = \frac{\sigma_y}{\mathrm{RMSE}},
\qquad
\mathrm{RPIQ} = \frac{Q_3(y)-Q_1(y)}{\mathrm{RMSE}}
```

Lin's concordance correlation coefficient (Lin 1989):

```math
\mathrm{CCC} = \frac{2\,\sigma_{y\hat{y}}}{\sigma_y^2+\sigma_{\hat{y}}^2+(\bar{y}-\bar{\hat{y}})^2}
```

Metrics are given as mean ± SD over the outer folds and pooled over all outer
predictions. The scatter plots show, for each profile, the mean of its out-of-fold
predictions across repeats.

### 7.5 Model interpretation: SHAP and partial dependence

**SHAP values.** For a model $`f`$, features $`F`$ and an observation $`x`$, the
SHAP value of feature $`j`$ is its Shapley value (Lundberg & Lee 2017):

```math
\phi_j(x) = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,\bigl(|F|-|S|-1\bigr)!}{|F|!}\,\Bigl[f_{S\cup\{j\}}(x)-f_{S}(x)\Bigr],
\qquad
f(x) = \phi_0 + \sum_{j\in F}\phi_j(x)
```

They are computed exactly for trees with TreeSHAP (Lundberg et al. 2020): on GPU for
XGBoost (`pred_contribs=True`) and with `shap.TreeExplainer` on a scikit-learn Random
Forest that uses the tuned hyper-parameters. Values are out of fold: 3 × 10-fold, so
every profile is explained three times by models that did not see it.

**Importance.** $`I_j = \frac{1}{N}\sum_i \lvert\phi_j(x_i)\rvert`$, shown as a share
of $`\sum_j I_j`$; the error bars are the SD across the three seeds.

**Dependence.** For the four most important continuous covariates of each depth,
$`\phi_j`$ is plotted against $`x_j`$ (1st–99th percentile) with a LOWESS smoother
and its 95 % bootstrap band (100 resamples).

**Partial dependence and ICE.** For the final model, the individual conditional
expectation of profile $`i`$ and the partial dependence are

```math
\hat{f}^{(i)}_j(v) = f\bigl(v,\ x_{i,-j}\bigr),
\qquad
\bar{f}_j(v) = \frac{1}{n}\sum_{i=1}^{n} f\bigl(v,\ x_{i,-j}\bigr)
```

evaluated on 40 grid values between the 2nd and 98th percentiles of $`x_j`$ with a
subsample of 600 profiles. To put it on the SHAP scale, the curve is centred so
that its average over the observed distribution of $`x_j`$ is zero:

```math
\bar{f}^{\,c}_j(v) = \bar{f}_j(v) - \frac{1}{N}\sum_{i=1}^{N}\bar{f}_j(x_{ij})
```

The shaded band is the 10th–90th percentile of the centred ICE curves. Where the
LOWESS of the SHAP values and the centred PDP agree, the effect of the covariate is
mostly additive; where they diverge, interactions or correlated covariates are at
play.

**Categorical covariates.** The distribution of $`\phi_j`$ is shown per land-cover
and per soil class.

### 7.6 Mapping and uncertainty

**Bootstrap ensemble.** For each target, Random Forest and XGBoost are refitted $`B = 100`$ times on
bootstrap resamples of the full table with the final hyper-parameters. For SOC
30–100 and 0–100 cm the group decay rates are re-estimated on every bootstrap
sample before the training targets are built, so the map uncertainty includes the
uncertainty of the depth harmonisation.

**Prediction.** Covariate rasters are read in 4096 × 4096 pixel tiles. Pixels with
NoData in any layer, MAT or MAP equal to 0, or permanent water are masked.
Categorical codes are translated as raster code → class name → encoder index; land
cover classes absent from the training data are masked and unseen soil classes get
the most frequent training class. Each tile is sent to the GPU and predicted by
every bootstrap model (cuML for Random Forest, `inplace_predict` for XGBoost).

**Running statistics.** Mean and variance are accumulated with Welford's algorithm
(Welford 1962), so no stack of predictions is kept in memory. For the $`b`$-th
prediction $`p_b`$ of a pixel:

```math
\delta_b = p_b - \mu_{b-1},\qquad
\mu_b = \mu_{b-1} + \frac{\delta_b}{b},\qquad
M_b = M_{b-1} + \delta_b\,(p_b - \mu_b)
```

```math
\mathrm{SD} = \sqrt{\frac{M_B}{B-1}},\qquad
\mathrm{CV} = 100\,\frac{\mathrm{SD}}{\mu_B}\ (\%)
```

**National stocks.** Pixel areas are computed on the sphere when the rasters are in
geographic coordinates ($`R = 6\,371\,008.8`$ m, latitudes $`\varphi_1, \varphi_2`$ of
the pixel edges, width $`\Delta\lambda`$ in radians):

```math
A = R^2\,\Delta\lambda\,\bigl\lvert\sin\varphi_1-\sin\varphi_2\bigr\rvert
```

and the total stock is $`\sum_{\text{pixels}} \mu_B \cdot A`$ (reported in Pg C).
The summed pixel SD is an upper bound of the total's uncertainty since it assumes
fully correlated errors. Additivity is checked by comparing
$`\mathrm{SOC}_{0\text{-}30} + \mathrm{SOC}_{30\text{-}100}`$ with
$`\mathrm{SOC}_{0\text{-}100}`$ at the national scale.

### 7.7 Quantile regression forest and prediction intervals

A random forest summarises the training responses that fall in the same leaves as a
new observation by their mean. A quantile regression forest (Meinshausen 2006) keeps
the whole set and estimates the full conditional distribution. With $`T`$ trees,
$`L_t(x)`$ the leaf of tree $`t`$ reached by $`x`$ and $`n`$ training profiles
$`(X_i, Y_i)`$, each profile receives the weight

```math
w_i(x) = \frac{1}{T}\sum_{t=1}^{T}\frac{\mathbb{1}\{X_i \in L_t(x)\}}{\bigl\lvert\{j : X_j \in L_t(x)\}\bigr\rvert},
\qquad \sum_{i=1}^{n} w_i(x) = 1
```

The conditional distribution function and its quantiles are then

```math
\hat{F}(y \mid x) = \sum_{i=1}^{n} w_i(x)\,\mathbb{1}\{Y_i \le y\},
\qquad
\hat{q}_{\alpha}(x) = \inf\bigl\{\,y : \hat{F}(y \mid x) \ge \alpha\,\bigr\}
```

and the conditional mean, used as the QRF point prediction in the CV and the maps,
is $`\hat{\mu}(x) = \sum_i w_i(x)\,Y_i`$. All training samples are kept in the
leaves (`max_samples_leaf=None`), so the quantiles are those of the original
method, not an approximation. Quantiles are made monotone per observation before
use.

**Prediction interval.** The central $`(1-2\alpha)`$ interval is
$`\bigl[\hat{q}_{\alpha}(x),\ \hat{q}_{1-\alpha}(x)\bigr]`$; the maps use
$`\alpha = 0.05`$, i.e. a 90 % interval, with width and relative width

```math
\mathrm{PIW}_{90}(x) = \hat{q}_{0.95}(x) - \hat{q}_{0.05}(x),
\qquad
\mathrm{rPIW}_{90}(x) = 100\,\frac{\mathrm{PIW}_{90}(x)}{\hat{q}_{0.50}(x)}\ (\%)
```

**Are the intervals honest?** In both CV schemes the outer QRF also predicts the
quantiles 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95 and 0.975 of every
held-out profile. For the 50, 80, 90 and 95 % central intervals we report the
prediction interval coverage probability and the mean interval width

```math
\mathrm{PICP} = \frac{100}{n}\sum_{i=1}^{n}\mathbb{1}\bigl\{\hat{q}_{\alpha}(x_i) \le y_i \le \hat{q}_{1-\alpha}(x_i)\bigr\},
\qquad
\mathrm{MPIW} = \frac{1}{n}\sum_{i=1}^{n}\bigl(\hat{q}_{1-\alpha}(x_i)-\hat{q}_{\alpha}(x_i)\bigr)
```

on the measured profiles and on all profiles. A well-calibrated model has PICP close
to the nominal level; a PICP below it means the intervals are too narrow. Fig. S7
plots PICP against the nominal level for both CV schemes, and the spatial CV shows
how the intervals hold up away from the training data.

**Interval rasters.** The final QRF (tuned with the scheme set by `FINAL_MODEL_CV`)
predicts the mean and the 5th, 50th and 95th percentiles for every valid pixel, with
the same masks and categorical handling as the bootstrap maps (`soc_raster.py`).
Pixels are processed in batches of 250,000 inside 2048-pixel tiles. Because the QRF
runs on CPU, `QRF_MAP_DECIM` (or `--decim`) can predict on a grid 2, 4, … times
coarser (continuous covariates averaged, categorical ones taken at the nearest
pixel). National totals are computed from the mean map; the sums of the pixel 5th and
95th percentiles are also reported, but they only bound the national total and are
not its quantiles, since pixel errors are not perfectly correlated.

The two uncertainty products answer different questions. The bootstrap SD of the
Random Forest and XGBoost ensembles measures how much the *fitted model* would change
with another sample of profiles (plus the depth harmonisation). The QRF interval
measures where a *new observation* at that pixel is likely to fall, so it also
contains the residual variability of SOC that the covariates do not explain, and it
is much wider.

---

## Outputs

All outputs are written under `SOC_WORK` (default `/kaggle/working`). Main folders:

| Folder | Content |
|---|---|
| `Harmonization/` | harmonised table, group parameters, validation tables, Fig. S1 |
| `ML_SOC_Harmonized/results_<target>/` | random CV: predictions, OOF means, fold metrics, tuned parameters, summary JSON, final models |
| `ML_SOC_Harmonized/results_<target>_spatialCV/` | same for spatial CV, plus `spatial_folds.csv` |
| `ML_SOC_Harmonized/SHAP/` | SHAP values (`.npz`) and importance tables |
| `ML_SOC_Harmonized/Figures/` | all figures, PNG (600 dpi) and PDF |
| `Maps_Harmonized/<target>_<model>/` | RF / XGBoost: `*_mean.tif`, `*_sd.tif`, `*_cv.tif`, bootstrap diagnostics, `map_stats.json` |
| `Maps_Harmonized/<target>_QRF/` | `*_mean.tif`, `*_q05.tif`, `*_q50.tif`, `*_q95.tif`, `*_piw90.tif`, `*_rpiw90.tif`, `map_stats.json` |

Figures:

| File | Content |
|---|---|
| `FigS1_depth_harmonization` | ratios by group, depth curves, out-of-fold validation, measured vs harmonised distributions |
| `Fig2_obs_vs_pred_measured_{randomCV,spatialCV}` | observed vs predicted, measured profiles |
| `FigS2_obs_vs_pred_all_points_{randomCV,spatialCV}` | same, harmonised points included |
| `Fig2c_random_vs_spatialCV` | R², RMSE, CCC under both schemes, three models |
| `FigS7_QRF_interval_coverage` | QRF interval coverage (PICP) vs nominal level, both schemes |
| `FigS5_spatialCV_blocks_used`, `FigS6_spatialCV_all_grid_positions` | spatial folds and test-to-training distances |
| `Fig3_SHAP_importance` | relative mean \|SHAP\|, three models |
| `Fig4_SHAP_beeswarm` | SHAP beeswarm, 3 models × 3 depths |
| `Fig5a/5b/5c_SHAP_PDP_dependence_<model>` | SHAP dependence with centred PDP |
| `FigS3a/3b/3c_SHAP_categorical_<model>` | SHAP by land-cover and soil class |
| `Fig7_maps_<model>` | bootstrap mean, SD and CV maps for the three depths (RF, XGBoost) |
| `Fig8_QRF_prediction_interval_maps` | QRF lower bound, median, upper bound and 90 % interval width |

A column-by-column description of the tables is in [`docs/outputs.md`](docs/outputs.md).

---

## Configuration

Everything is set in `src/soc_config.py`. Paths can also be given through
environment variables (`SOC_WORK`, `SOC_PRED_CSV`, `SOC_CORR_CSV`,
`SOC_RASTER_DIR`, `SOC_GEOJSON`). The settings most likely to be changed:

| Setting | Default | Meaning |
|---|---|---|
| `SHRINK_K0` | 10 | pseudo-count $`\kappa`$ of the hierarchical shrinkage |
| `CALIBRATE_ON` | `'LU'` | calibrate only on profiles with a field land-use label |
| `N_BOOT_HARM` | 500 | bootstrap replicates for harmonisation uncertainty |
| `N_OUTER_FOLDS`, `N_OUTER_REPEATS`, `N_INNER_FOLDS`, `OPTUNA_TRIALS` | 10, 10, 4, 100 | random nested CV |
| `SPATIAL_DESIGN` | hexagon 125 / 150 / 150 km | block shape and size per target |
| `N_SPATIAL_REPEATS` | 4 | grid positions used as repeats |
| `SPATIAL_HARMONIZED_WEIGHT` | 0.1 | weight of harmonised profiles when balancing spatial folds |
| `CV_SCHEMES` | `['random', 'spatial']` | schemes run by step 2 |
| `FINAL_MODEL_CV` | `'random'` | which tuned final models feed SHAP and mapping |
| `N_BOOT_MAP` | 100 | bootstrap models per map |
| `MODELS` | RF, XGBoost, QRF | learners run by step 2 and shown in the figures |
| `QRF_OPTUNA_TRIALS` | 30 | Optuna trials per QRF search |
| `QRF_QUANTILES` | 0.025 … 0.975 | quantiles stored for every held-out profile |
| `QRF_PI` | (0.05, 0.95) | bounds of the mapped prediction interval |
| `QRF_MAP_DECIM` | 1 | coarsening factor for the QRF interval maps |
| `UNSEEN_CLASS_POLICY` | lulc: mask, soil: mode | handling of raster classes absent from training |
| `MASK_LULC_CODES` | (80,) | land-cover codes always masked (permanent water) |

---

## Run time

With 4,184 profiles, each outer fit runs 100 Optuna trials with 4 inner folds.
Step 2 is by far the longest: two CV schemes × three targets per GPU will not fit
in one 12-hour Kaggle session. Results are written to disk as each target finishes,
so the simplest approach is to run one scheme, or one target, per session
(`CV_TO_RUN`, `TARGETS_TO_RUN` in `02_run_nested_cv.py`) or to reduce the effort with
`EXTRA_ARGS = ['--repeats', '5', '--trials', '50']`. The QRF runs on the CPU in
parallel with the two GPU queues; with 30 trials it is usually not the bottleneck.
Mapping takes from tens of minutes to a few hours depending on raster resolution
and the number of bootstrap models. QRF interval prediction costs roughly 30–70 µs
per pixel and core, so about 10 million pixels take 5–15 minutes per depth on a
4-core machine; use `--decim` for very fine grids.

---

## Limitations

- The exponential model assumes a smooth monotonic decrease (or increase) of SOC
  with depth. Buried horizons, peat layers or very shallow soils are not described
  well.
- Profiles without a field land-use label take their group from ESA WorldCover at
  the point. Misclassification there translates directly into the wrong decay rate.
  Points labelled "Mangroves" by ESA get the mangrove rate, which roughly triples
  SOC 0–30 cm.
- Harmonised values are model outputs. Their uncertainty is propagated into the
  maps through the bootstrap, but the deeper-layer models should be judged on the
  measured-profile validation.
- Under spatial CV, folds contain different numbers of measured 1 m profiles, so
  fold-level R² varies a lot; the pooled value is the more stable summary.
- The national stock SD reported in `map_stats.json` is an upper bound.
- QRF intervals are conditional on the covariates and on the harmonised targets; the
  uncertainty of the depth model itself is not added to them (it is in the
  bootstrap maps). Where covariates fall outside the training range, the QRF
  cannot widen its interval beyond the spread of the training responses.

---

## How to cite

If you use this code, please cite the archived release (see `CITATION.cff`; GitHub
shows a "Cite this repository" button):

> Randrianantenaina, A. S., & Ramifehiarivo, N. (2026). *Depth-harmonised soil organic carbon mapping for
> Madagascar* (Version 1.1.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.23857981

---

## References

- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: a
  next-generation hyperparameter optimization framework. *Proceedings of the 25th
  ACM SIGKDD Conference*, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.
  https://doi.org/10.1023/A:1010933404324
- Chen, T. & Guestrin, C. (2016). XGBoost: a scalable tree boosting system.
  *Proceedings of the 22nd ACM SIGKDD Conference*, 785–794.
  https://doi.org/10.1145/2939672.2939785
- Gale, M. R. & Grigal, D. F. (1987). Vertical root distributions of northern tree
  species in relation to successional status. *Canadian Journal of Forest Research*,
  17, 829–834. https://doi.org/10.1139/x87-131
- Jobbágy, E. G. & Jackson, R. B. (2000). The vertical distribution of soil organic
  carbon and its relation to climate and vegetation. *Ecological Applications*, 10,
  423–436. https://doi.org/10.1890/1051-0761(2000)010[0423:TVDOSO]2.0.CO;2
- Lin, L. I.-K. (1989). A concordance correlation coefficient to evaluate
  reproducibility. *Biometrics*, 45, 255–268. https://doi.org/10.2307/2532051
- Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.
- Lundberg, S. M. et al. (2020). From local explanations to global understanding
  with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
  https://doi.org/10.1038/s42256-019-0138-9
- Johnson, R. A. (2024). quantile-forest: a Python package for quantile regression
  forests. *Journal of Open Source Software*, 9(93), 5976.
  https://doi.org/10.21105/joss.05976
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning
  Research*, 7, 983–999.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal,
  spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929.
  https://doi.org/10.1111/ecog.02881
- Welford, B. P. (1962). Note on a method for calculating corrected sums of squares
  and products. *Technometrics*, 4, 419–420.
  https://doi.org/10.1080/00401706.1962.10490022
- Zanaga, D. et al. (2021). ESA WorldCover 10 m 2020 v100. Zenodo.
  https://doi.org/10.5281/zenodo.5571936

---

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Antsa Sarobidy
Randrianantenaina and Nandrianina Ramifehiarivo.
