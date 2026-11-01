"""Build a small, fully synthetic dataset that has the same structure as the real
inputs (point table, covariate rasters, country outline), so the whole pipeline can
be run end to end without the original data.

    python tests/make_synthetic_data.py /tmp/soc_test

None of the numbers produced here mean anything.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

out = sys.argv[1] if len(sys.argv) > 1 else 'soc_test'
os.makedirs(f'{out}/rasters', exist_ok=True)
rng = np.random.default_rng(0)

# ── raster grid (0.05°) over a Madagascar-shaped ellipse ─────────────────────
x0, y0, res = 43.0, -11.9, 0.05
W, H = int((50.5 - x0) / res), int((25.7 - 11.9) / res)
yy, xx = np.mgrid[0:H, 0:W]
lon, lat = x0 + xx * res, y0 - yy * res
outside = ((lon - 47) ** 2 / 3.0 ** 2 + (lat + 19) ** 2 / 6.8 ** 2) > 1

elev = np.clip(2000 * np.exp(-((lon - 47) ** 2) / 2) * (1 + 0.2 * np.sin(lat)), 0, None)
layers = {
    'elevation': elev,
    'slope': np.abs(rng.normal(8, 5, lon.shape)),
    'mat': (26 - 0.006 * elev) * 10,                       # stored ×10 like the real raster
    'map': 800 + 200 * (lon - 43) + rng.normal(0, 30, lon.shape),
    'NDVI': np.clip(0.3 + 0.08 * (lon - 45) + rng.normal(0, .05, lon.shape), -0.2, 0.95),
    'NDWI': rng.normal(0, 0.1, lon.shape),
    'NIRI': rng.normal(0.3, 0.05, lon.shape),
    'treecover': np.clip(10 * (lon - 44) + rng.normal(0, 5, lon.shape), 0, 100),
    'lulc': np.choose(rng.integers(0, 6, lon.shape), [10, 20, 30, 40, 90, 95]).astype(float),
    'soil_type': rng.integers(1, 12, lon.shape).astype(float),
}
for name, arr in layers.items():
    a = arr.astype(np.float32)
    a[outside] = -9999
    with rasterio.open(f'{out}/rasters/{name}.tif', 'w', driver='GTiff', height=H, width=W,
                       count=1, dtype='float32', crs='EPSG:4326',
                       transform=from_origin(x0, y0, res, res), nodata=-9999) as dst:
        dst.write(a, 1)

# ── country outline ───────────────────────────────────────────────────────────
t = np.linspace(0, 2 * np.pi, 200)
ring = np.c_[47 + 2.9 * np.cos(t), -19 + 6.7 * np.sin(t)].tolist()
json.dump({'type': 'FeatureCollection', 'features': [{
    'type': 'Feature', 'properties': {},
    'geometry': {'type': 'Polygon', 'coordinates': [ring + [ring[0]]]}}]},
    open(f'{out}/outline.geojson', 'w'))

# ── point table ───────────────────────────────────────────────────────────────
n = 900
inside = np.flatnonzero(~outside.ravel())
pix = rng.choice(inside, n, replace=False)
r, c = np.unravel_index(pix, lon.shape)
df = pd.DataFrame({'num': np.arange(1, n + 1),
                   'Longitude': lon[r, c] + rng.uniform(0, res, n),
                   'Latitude': lat[r, c] - rng.uniform(0, res, n)})
LU = ['Forest', 'Grassland', 'Shrubland', 'Crop', 'Rice', 'AgroF', 'Mangroves', 'undetermined']
df['LU'] = rng.choice(LU, n, p=[.15, .2, .08, .1, .08, .05, .09, .25])
LC = {10: 'Tree Cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
      90: 'Herbaceous Wetland', 95: 'Mangroves'}
SC = {1: 'Andosols', 2: 'Bare Rock', 3: 'Ferrallitic Soils', 4: 'Ferruginous Soils',
      5: 'Fersiallitic Soils', 6: 'Hydromorphic Soils', 7: 'Lithic Raw Mineral Soils',
      8: 'Alluvial Weakly Developed Soils',
      9: 'Erosional Weakly Developed Soils / Lithosols',
      10: 'Podzolic Soils and Podzols', 11: 'Vertisols'}
for k, v in layers.items():
    val = v[r, c]
    df[k] = val / 10 if k == 'mat' else val
df['lulc'] = df['lulc'].astype(int).map(LC)
df['soil_type'] = df['soil_type'].astype(int).map(SC)

df['SC30'] = np.exp(3.8 + 0.0004 * df['elevation'] + 0.6 * df['NDVI']
                    + rng.normal(0, 0.35, n))
f30 = np.where(df['LU'].eq('Mangroves'), 0.29, rng.normal(0.56, 0.08, n).clip(0.35, 0.9))
has100 = rng.random(n) < 0.4
df['SOC100'] = np.where(has100, df['SC30'] / f30, np.nan)
df.to_csv(f'{out}/SOC30_predictors_table.csv', index=False)
print(f'synthetic data written to {out}/  ({n} points, {has100.sum()} with 0–100 cm)')
