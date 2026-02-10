"""Quick test: verify per-scenario RNG seeding gives realistic cascade rates."""
import warnings; warnings.filterwarnings("ignore")
import os, logging; logging.disable(logging.WARNING)
import pandapower as pp
_orig = pp.runpp
def _q(net, **kw):
    kw.setdefault('numba', False)
    return _orig(net, **kw)
pp.runpp = _q

import yaml, numpy as np, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.grid_setup import load_base_case
from src.cascade_sim import simulate_cascade

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

net = load_base_case(cfg["paths"]["base_case"])
weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))

# Get test indices
N = cfg["weather"]["n_scenarios"]
rng_split = np.random.default_rng(cfg["dataset"]["random_seed"])
all_idx = rng_split.permutation(N)
n_tr = int(N * cfg["dataset"]["train_ratio"])
n_va = int(N * cfg["dataset"]["val_ratio"])
test_idx = all_idx[n_tr + n_va:].tolist()

# Load ground truth
cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
y_all = cascade_raw["severity"]
gt_test_sev = y_all[test_idx]

n_test = 50
print(f"Testing {n_test} scenarios with per-scenario fixed RNG (idx+1000)")
print(f"Ground truth cascade rate on full test set: {(gt_test_sev > 0).mean():.3f}")
print(f"Ground truth cascade rate on first {n_test}: {(gt_test_sev[:n_test] > 0).mean():.3f}")

t0 = time.time()
preds = []
sevs = []
for idx in test_idx[:n_test]:
    rng = np.random.default_rng(idx + 1000)
    r = simulate_cascade(net, weather, idx, 12, cfg, rng)
    preds.append(1 if r["severity"] > 0 else 0)
    sevs.append(r["severity"])

preds = np.array(preds)
sevs = np.array(sevs)
elapsed = time.time() - t0

print(f"\nPer-scenario RNG results (n={n_test}):")
print(f"  Cascade rate: {preds.mean():.3f}")
print(f"  Severity dist: {[int((sevs==s).sum()) for s in range(4)]}")
print(f"  Time: {elapsed:.1f}s ({elapsed/n_test*1000:.0f}ms/scenario)")

# Also test with old approach (single continuous RNG)
print(f"\nSingle-RNG results (n={n_test}):")
rng_single = np.random.default_rng(cfg["dataset"]["random_seed"])
preds2 = []
sevs2 = []
for idx in test_idx[:n_test]:
    r = simulate_cascade(net, weather, idx, 12, cfg, rng_single)
    preds2.append(1 if r["severity"] > 0 else 0)
    sevs2.append(r["severity"])

preds2 = np.array(preds2)
sevs2 = np.array(sevs2)
print(f"  Cascade rate: {preds2.mean():.3f}")
print(f"  Severity dist: {[int((sevs2==s).sum()) for s in range(4)]}")
print(f"\nGround truth: {[int((gt_test_sev[:n_test]==s).sum()) for s in range(4)]}")
