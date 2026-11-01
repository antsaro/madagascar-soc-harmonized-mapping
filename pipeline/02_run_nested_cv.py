# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — LAUNCH NESTED CV  (RF → GPU 0, XGBoost → GPU 1, QRF → CPU, in parallel)
#  Each model runs, on its own GPU, every CV scheme × target sequentially:
#    random  : 10 × 10-fold KFold outer, 4-fold inner
#    spatial : 10 hexagon-block folds × 4 grid shifts outer, 4 block folds inner
#  Separate processes avoid the "cudaErrorInitializationError" seen with fork.
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import os
import subprocess
import sys
import time

import soc_config as C

TARGETS_TO_RUN = ['SOC30', 'DeepSOC', 'SOC100']     # split across sessions if needed
CV_TO_RUN = C.CV_SCHEMES                             # ['random', 'spatial']
QUEUES = {0: 'RandomForest', 1: 'XGBoost', 'cpu': 'QRF'}   # gpu id (or 'cpu') → model
QUEUES = {g: m for g, m in QUEUES.items() if m in C.MODELS}
EXTRA_ARGS = []        # e.g. ['--repeats', '5', '--trials', '50'] for a faster run

n_gpu = 0
try:
    import cupy as cp
    n_gpu = cp.cuda.runtime.getDeviceCount()
except Exception:                                                # noqa: BLE001
    pass
print(f'GPUs visible: {n_gpu}')

SRC = os.path.dirname(os.path.abspath(C.__file__))        # folder holding the modules
log_dir = os.path.join(C.RESULTS_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)
procs = {}
for gpu, model in QUEUES.items():
    cmds = ' ; '.join(           # ';' → a failed job does not stop the queue
        f'{sys.executable} -u {SRC}/train_soc_harm.py --target {t} --model {model} --cv {cv} '
        + ' '.join(EXTRA_ARGS) for cv in CV_TO_RUN for t in TARGETS_TO_RUN)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '' if gpu == 'cpu' else str(gpu if n_gpu > 1 else 0)
    env['PYTHONPATH'] = SRC + os.pathsep + env.get('PYTHONPATH', '')
    log = open(os.path.join(log_dir, f'log_{model}.txt'), 'w')
    procs[model] = (subprocess.Popen(cmds, shell=True, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, cwd=SRC), log)
    print(f'Launched {model} on {"CPU" if gpu == "cpu" else "GPU " + env["CUDA_VISIBLE_DEVICES"]} → {log.name}')

# ── live monitor (last line of each log every 60 s) ──────────────────────────
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(60)
    for model, (p, log) in procs.items():
        try:
            with open(log.name) as f:
                lines = [l.strip() for l in f if l.strip()]
            print(f'[{model:<12}] {"running" if p.poll() is None else "done"} | '
                  f'{lines[-1] if lines else ""}', flush=True)
        except FileNotFoundError:
            pass
for model, (p, log) in procs.items():
    log.close()
    print(f'{model}: exit code {p.returncode}')
