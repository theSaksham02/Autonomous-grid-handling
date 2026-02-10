"""Quick test of cascade evaluation pipeline."""
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp
_orig = pp.runpp
def _q(net, **kw):
    kw.setdefault("numba", False)
    return _orig(net, **kw)
pp.runpp = _q

import numpy as np, yaml, time
from src.grid_setup import load_base_case
from src.grid_env import GridCascadeEnv

cfg = yaml.safe_load(open("config.yaml"))
cfg["rl"]["episode_length"] = 8
net = load_base_case(cfg["paths"]["base_case"])
weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))

sys_path = __import__("sys").path
sys_path.insert(0, ".")
from run_stage6 import evaluate_with_cascade

splits_rng = np.random.default_rng(cfg["dataset"]["random_seed"])
all_idx = splits_rng.permutation(cfg["weather"]["n_scenarios"])
n_tr = int(len(all_idx) * cfg["dataset"]["train_ratio"])
n_va = int(len(all_idx) * cfg["dataset"]["val_ratio"])
test_idx = all_idx[n_tr + n_va:].tolist()

N = 10

# Zero-action agent (baseline — no intervention)
def zero_agent(obs, net):
    return np.array([0, 0, 0, 0], dtype=np.float32)

# Active agent: curtail renewables + reduce demand
def active_agent(obs, net):
    return np.array([0.5, 0.0, 0.3, 0.15], dtype=np.float32)

print(f"Testing on {N} scenarios:\n")

print("Zero-action agent (no intervention):")
t0 = time.time()
z_p, z_s, z_sh, _ = evaluate_with_cascade(
    zero_agent, net, weather, cfg, test_idx, n_test=N)
print(f"  cascade_rate={z_p.mean():.2f}, mean_shed={z_sh.mean():.3f}")
print(f"  sev dist: {[int((z_s==s).sum()) for s in range(4)]}, {time.time()-t0:.1f}s")

print("\nActive agent (fixed actions):")
t0 = time.time()
a_p, a_s, a_sh, _ = evaluate_with_cascade(
    active_agent, net, weather, cfg, test_idx, n_test=N)
print(f"  cascade_rate={a_p.mean():.2f}, mean_shed={a_sh.mean():.3f}")
print(f"  sev dist: {[int((a_s==s).sum()) for s in range(4)]}, {time.time()-t0:.1f}s")

# Check if active agent is better
improvement = z_sh.mean() - a_sh.mean()
print(f"\nLoad shed reduction: {improvement:.3f} ({improvement/max(z_sh.mean(),0.001)*100:.1f}%)")
