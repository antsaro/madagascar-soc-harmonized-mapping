# ═══════════════════════════════════════════════════════════════════════════════
#  nature_style.py — shared Nature-Reviews-style figure settings (Arial)
# ═══════════════════════════════════════════════════════════════════════════════
import glob
import os

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

MM = 1 / 25.4                      # mm → inch
SINGLE_COL = 89 * MM               # Nature single column
DOUBLE_COL = 183 * MM              # Nature double column

# Nature Publishing Group palette (ggsci "npg")
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
       '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
MODEL_COLOR = {'RandomForest': '#3C5488', 'XGBoost': '#E64B35', 'QRF': '#7E6148'}
MODEL_LABEL = {'RandomForest': 'Random forest', 'XGBoost': 'XGBoost', 'QRF': 'QRF'}
MODEL_MARKER = {'RandomForest': 'o', 'XGBoost': 's', 'QRF': '^'}
INK = '#222222'
GREY = '#8C8C8C'
LIGHT = '#D9D9D9'

UNIT = r'Mg C ha$^{-1}$'
DEPTH_ORDER = ['SOC30', 'DeepSOC', 'SOC100']
DEPTH_LABEL = {                       # \mathregular keeps the en dash in Arial
    'SOC30':   r'SOC$_{\mathregular{0–30}}$',
    'DeepSOC': r'SOC$_{\mathregular{30–100}}$',
    'SOC100':  r'SOC$_{\mathregular{0–100}}$',
}

FEATURE_LABEL = {
    'NDVI': 'NDVI', 'NDWI': 'NDWI', 'NIRI': 'NIRI',
    'elevation': 'Elevation (m)', 'slope': 'Slope (°)',
    'mat': 'MAT (°C)', 'map': 'MAP (mm)', 'treecover': 'Tree cover (%)',
    'lulc': 'Land cover', 'soil_type': 'Soil type',
    'lulc_enc': 'Land cover', 'soil_type_enc': 'Soil type',
}


def flabel(k):
    return FEATURE_LABEL.get(k, k)


def setup_arial(verbose=True):
    """Register Arial. Tries (1) system Arial, (2) kagglehub 'hammaadali/arial-font',
    (3) Liberation Sans (metric-compatible clone) as last resort."""
    names = {f.name for f in fm.fontManager.ttflist}
    family = None
    if 'Arial' in names:
        family = 'Arial'
    else:
        try:
            import kagglehub
            p = kagglehub.dataset_download('hammaadali/arial-font')
            ttfs = glob.glob(f'{p}/**/*.ttf', recursive=True)
            for t in ttfs:
                fm.fontManager.addfont(t)
            if ttfs:
                family = fm.FontProperties(fname=ttfs[0]).get_name()
        except Exception as e:                                   # noqa: BLE001
            if verbose:
                print(f'[font] kagglehub Arial unavailable ({e})')
    if family is None:
        family = 'Liberation Sans' if 'Liberation Sans' in names else 'DejaVu Sans'
        if verbose:
            print(f'[font] Arial not found — falling back to {family}')
    apply_style(family)
    if verbose:
        print(f'[font] Active font family → {family}')
    return family


def apply_style(family='Arial'):
    mpl.rcParams.update({
        'font.family': family,
        'font.sans-serif': [family, 'Arial', 'Helvetica', 'Liberation Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': family, 'mathtext.it': f'{family}:italic',
        'mathtext.bf': f'{family}:bold', 'mathtext.sf': family,
        'mathtext.cal': family, 'mathtext.tt': family,
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
        'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
        'legend.frameon': False, 'legend.handlelength': 1.2,
        'lines.linewidth': 1.0, 'patch.linewidth': 0.4,
        'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def panel_letter(ax, letter, x=-0.18, y=1.04, size=9):
    """Nature style: bold lower-case letter, no period, top-left."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight='bold', va='bottom', ha='left', color='black')


def save(fig, path_no_ext, formats=('png', 'pdf')):
    os.makedirs(os.path.dirname(path_no_ext) or '.', exist_ok=True)
    for ext in formats:
        fig.savefig(f'{path_no_ext}.{ext}')
    print(f'Saved → {path_no_ext}.[{",".join(formats)}]')
