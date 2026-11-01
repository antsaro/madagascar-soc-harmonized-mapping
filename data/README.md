# Input data

The soil profile data and covariate rasters are not distributed with this
repository. This page describes what the code expects so that the pipeline can be
run on the original data or on a comparable dataset. `tests/make_synthetic_data.py`
writes files with exactly this structure.

## 1. Point table

CSV, one row per profile, default path `$SOC_WORK/SOC30_predictors_table.csv`
(override with `SOC_PRED_CSV`).

| Column | Type | Required | Notes |
|---|---|---|---|
| `num` | int | yes | unique id, used to match the corrections file |
| `Longitude` | float | yes | WGS84, decimal degrees |
| `Latitude` | float | yes | WGS84, decimal degrees |
| `SC30` | float | yes | SOC stock 0–30 cm, Mg C ha⁻¹ |
| `SOC100` | float | no | SOC stock 0–100 cm, Mg C ha⁻¹; empty if not measured |
| `LU` | str | yes | field land-use label; see the mapping below |
| `NDVI`, `NDWI`, `NIRI` | float | yes | spectral indices at the point |
| `elevation` | float | yes | m |
| `slope` | float | yes | degrees |
| `mat` | float | yes | mean annual temperature, °C (already divided by 10) |
| `map` | float | yes | mean annual precipitation, mm |
| `treecover` | float | yes | % |
| `lulc` | str | yes | ESA WorldCover 2020 class name |
| `soil_type` | str | yes | soil class name |

Columns named `SOC0_30` / `SOC0_100` / `SOC100_ten` are renamed automatically.
Value `-9999` is read as missing. Missing numeric covariates are filled with the
column median.

### Field land-use labels → harmonisation groups

| `LU` values | Group | Parent |
|---|---|---|
| Forest, Closed Canopy, Pine, Reforestation | Forest | woody |
| AgroF, AGroF, Agroforesterie | Agroforestry | woody |
| Tree fallow, Shrub fallow, Jachere | Fallow | woody |
| Shrubland, Pseudo-steppe | Shrubland | woody |
| Grassland, Tany Maty, Bare | Grassland | herbaceous |
| Crop, Cropland, Culture | Cropland | herbaceous |
| Rice | Rice/Wetland | herbaceous |
| Mangroves | Mangrove | mangrove |
| undetermined / anything else | from ESA WorldCover (below) | |

| ESA WorldCover class | Group |
|---|---|
| Tree Cover | Forest |
| Shrubland | Shrubland |
| Grassland, Built-up, Bare / Sparse Vegetation, Moss and Lichen, Snow and Ice | Grassland |
| Cropland | Cropland |
| Herbaceous Wetland, Permanent Water Bodies | Rice/Wetland |
| Mangroves | Mangrove |

The tables are defined in `src/soc_harmonize.py` (`LU_TO_GROUP`, `ESA_TO_GROUP`)
and can be edited there.

## 2. Corrections file (optional)

CSV (comma or semicolon) with at least `num`, `SC30`, `SOC100`. Values for matching
`num` replace those of the point table. Decimal commas are accepted. Set
`SOC_CORR_CSV=none` to skip.

## 3. Covariate rasters

One single-band GeoTIFF per covariate, all on the same grid (identical width,
height, transform and CRS). File names are set in `RASTERS` in
`src/soc_config.py`, or taken as `<SOC_RASTER_DIR>/<name>.tif` when
`SOC_RASTER_DIR` is set, with names `NDVI`, `NDWI`, `NIRI`, `elevation`, `slope`,
`mat`, `map`, `treecover`, `lulc`, `soil_type`.

- `mat` is stored × 10 (°C × 10) and divided by 10 on reading (`MAT_DIVIDE_BY_10`).
- In `mat` and `map`, 0 is treated as NoData.
- `lulc` uses ESA WorldCover codes (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100).
- `soil_type` uses codes 1–11:

| Code | Soil type |
|---|---|
| 1 | Andosols |
| 2 | Bare Rock |
| 3 | Ferrallitic Soils |
| 4 | Ferruginous Soils |
| 5 | Fersiallitic Soils |
| 6 | Hydromorphic Soils |
| 7 | Lithic Raw Mineral Soils |
| 8 | Alluvial Weakly Developed Soils |
| 9 | Erosional Weakly Developed Soils / Lithosols |
| 10 | Podzolic Soils and Podzols |
| 11 | Vertisols |

## 4. Country outline

GeoJSON (WGS84) of the land area, used only for the spatial fold maps
(`SOC_GEOJSON`).
