# ═══════════════════════════════════════════════════════════════════════════════
#  soc_spatial.py — hexagonal spatial blocks for nested spatial CV
#  (same grid / fold logic as soc_spatial_cv.ipynb: UTM 38S, fixed origin,
#   4 repeats = grid shifted by half a block, greedy balanced fold assignment)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
from scipy.spatial import cKDTree

ORIGIN = np.array([300.0, 7100.0])          # grid origin, km, UTM 38S


def lonlat_to_km(lon, lat):
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 32738, always_xy=True)
    x, y = t.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.c_[x, y] / 1000.0


def grid_shifts(size):
    return [(0.0, 0.0), (size / 2, 0.0), (0.0, size / 2), (size / 2, size / 2)]


def block_keys(xy, shape_, size, shift=(0.0, 0.0)):
    p = xy - ORIGIN - np.asarray(shift)
    if shape_ == 'square':
        return np.floor(p / size).astype(int)
    # pointy-top hexagons, flat-to-flat width = size (cube-coordinate rounding)
    r = size / np.sqrt(3)
    q = (np.sqrt(3) / 3 * p[:, 0] - p[:, 1] / 3) / r
    z = (2 / 3 * p[:, 1]) / r
    y = -q - z
    rx, ry, rz = np.round(q), np.round(y), np.round(z)
    dx, dy, dz = abs(rx - q), abs(ry - y), abs(rz - z)
    fx = (dx > dy) & (dx > dz)
    fz = ~fx & ~((dy > dz) & ~fx)
    rx = np.where(fx, -ry - rz, rx)
    rz = np.where(fz, -rx - ry, rz)
    return np.c_[rx, rz].astype(int)


def assign_folds(keys, k, weights=None):
    """Largest block first, always into the fold with the lowest load.
    `weights` (per plot) lets the load count e.g. measured 1 m profiles only."""
    blocks, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    if len(blocks) < k:
        return None, len(blocks)
    w = np.ones(len(keys)) if weights is None else np.asarray(weights, float)
    load_b = np.bincount(inv, weights=w, minlength=len(blocks))
    cnt_b = np.bincount(inv, minlength=len(blocks))
    load = np.zeros(k)
    fold_of_block = np.empty(len(blocks), int)
    for j in sorted(range(len(blocks)), key=lambda j: (-load_b[j], -cnt_b[j], j)):
        f = int(np.argmin(load))
        fold_of_block[j] = f
        load[f] += load_b[j]
    return fold_of_block[inv] + 1, len(blocks)


def build_folds(xy, shape_, size, k, shift=(0.0, 0.0), weights=None):
    # shrink the blocks only if there are too few of them to fill k folds
    while size >= 5:
        keys = block_keys(xy, shape_, size, shift)
        ids, n_blocks = assign_folds(keys, k, weights)
        if ids is not None and len(np.unique(ids)) == k:
            return dict(ids=ids, keys=keys, size=size, n_blocks=n_blocks)
        size *= 0.85
    raise RuntimeError('cannot build spatial folds')


def spatial_splits(xy, shape_, size, k, shift=(0.0, 0.0), weights=None):
    folds = build_folds(xy, shape_, size, k, shift, weights)
    splits = [(np.flatnonzero(folds['ids'] != f), np.flatnonzero(folds['ids'] == f))
              for f in range(1, k + 1)]
    return splits, folds


def test_train_dist(xy, fold):
    d = np.empty(len(xy))
    for f in np.unique(fold):
        test = fold == f
        d[test] = cKDTree(xy[~test]).query(xy[test])[0]
    return d
