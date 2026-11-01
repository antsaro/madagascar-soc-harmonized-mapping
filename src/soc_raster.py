# ═══════════════════════════════════════════════════════════════════════════════
#  soc_raster.py — covariate raster stack: aligned tiled reading, categorical
#  look-up tables, masks and pixel areas (shared by the two mapping scripts)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window

import soc_config as C


def build_cat_lut(col, encoder, mode_idx=None):
    """raster integer code → training-encoder index; -1 means masked (NoData).
    Codes are translated through their class *name* (SOIL/LULC_CLASSES)."""
    cmap = C.RASTER_CLASS_MAPS[col]
    classes = list(encoder.classes_)
    lut = np.full(max(cmap) + 2, -1, dtype=np.int32)
    policy = C.UNSEEN_CLASS_POLICY.get(col, 'mask')
    unseen = []
    for code, name in cmap.items():
        if name in classes:
            lut[code] = classes.index(name)
        else:
            unseen.append(name)
            lut[code] = mode_idx if (policy == 'mode' and mode_idx is not None) else -1
    if col == 'lulc':
        for code in C.MASK_LULC_CODES:
            if code < len(lut):
                lut[code] = -1
    print(f'   LUT {col}: {len(cmap) - len(unseen)} classes matched; '
          f'unseen {unseen} → {policy}')
    return lut


def pixel_area_ha(transform, crs, rows):
    """Area (ha) of the pixels lying in the given rows of a grid."""
    if crs is not None and crs.is_geographic:
        R = 6371008.8
        lat_top = transform.f + transform.e * rows
        lat_bot = lat_top + transform.e
        dlon = np.deg2rad(abs(transform.a))
        a = R ** 2 * dlon * np.abs(np.sin(np.deg2rad(lat_top)) - np.sin(np.deg2rad(lat_bot)))
        return a / 1e4
    return np.full(len(rows), abs(transform.a * transform.e) / 1e4)


class RasterStack:
    """All covariates opened once. Optional decimation `decim` (>1 predicts on a
    grid `decim` times coarser: continuous layers averaged, categorical nearest)."""

    def __init__(self, encoders, modes, tile=C.TILE_SIZE, decim=1):
        self.h = {k: rasterio.open(v) for k, v in C.RASTERS.items()}
        ref = self.h['NDVI'] if 'NDVI' in self.h else next(iter(self.h.values()))
        for k, src in self.h.items():
            if (src.height, src.width) != (ref.height, ref.width) or src.transform != ref.transform:
                raise ValueError(f'Raster {k} is not aligned with the reference grid')
        self.d = int(max(1, decim))
        self.src_H, self.src_W = ref.height, ref.width
        self.H, self.W = ref.height // self.d, ref.width // self.d
        self.transform = ref.transform * Affine.scale(self.d)
        self.crs = ref.crs
        self.tile = tile
        self.luts = {c: build_cat_lut(c, encoders[c], modes.get(c))
                     for c in C.CATEGORICAL_PREDICTORS}
        prof = ref.profile.copy()
        prof.update(dtype='float32', count=1, nodata=C.NODATA_OUT, compress='lzw',
                    predictor=2, tiled=True, blockxsize=512, blockysize=512,
                    BIGTIFF='YES', width=self.W, height=self.H, transform=self.transform)
        self.profile = prof
        self.tiles = [(r, c) for r in range(0, self.H, tile) for c in range(0, self.W, tile)]

    def window(self, r0, c0):
        return Window(c0, r0, min(self.tile, self.W - c0), min(self.tile, self.H - r0))

    def read(self, r0, c0):
        """Feature matrix of the valid pixels of an output tile.
        Returns X (n, p) float32, flat indices into the tile, output window."""
        win = self.window(r0, c0)
        th, tw = int(win.height), int(win.width)
        src_win = Window(win.col_off * self.d, win.row_off * self.d, tw * self.d, th * self.d)
        valid = np.ones((th, tw), bool)
        cols = {}
        for f in C.NUMERIC_PREDICTORS + C.CATEGORICAL_PREDICTORS:
            src = self.h[f]
            is_cat = f in C.CATEGORICAL_PREDICTORS
            if self.d == 1:
                arr = src.read(1, window=src_win).astype(np.float32)
            else:
                arr = src.read(1, window=src_win, out_shape=(th, tw), masked=True,
                               resampling=Resampling.nearest if is_cat else Resampling.average)
                arr = np.ma.filled(arr.astype(np.float32), np.nan)
            bad = ~np.isfinite(arr)
            if src.nodata is not None:
                bad |= np.isclose(arr, src.nodata)
            if f in C.ZERO_IS_NODATA:
                bad |= arr == 0
            if is_cat:
                lut = self.luts[f]
                code = np.clip(np.where(bad, 0, arr).astype(np.int64), 0, len(lut) - 1)
                enc = lut[code]
                bad |= enc < 0
                arr = enc.astype(np.float32)
            elif f == 'mat' and C.MAT_DIVIDE_BY_10:
                arr = arr / 10.0
            valid &= ~bad
            cols[f] = arr
        flat = np.flatnonzero(valid)
        if flat.size == 0:
            return None, None, win
        order = C.NUMERIC_PREDICTORS + C.CATEGORICAL_PREDICTORS
        X = np.column_stack([cols[f].ravel()[flat] for f in order]).astype(np.float32)
        return X, flat, win

    def close(self):
        for src in self.h.values():
            src.close()
