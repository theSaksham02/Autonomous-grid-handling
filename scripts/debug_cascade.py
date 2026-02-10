"""Quick debug script to check cascade conditions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle, numpy as np, pandapower as pp, copy, yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

with open("data/raw/base_case.pkl", "rb") as f:
    net = pickle.load(f)

weather = dict(np.load("data/raw/weather_scenarios.npz", allow_pickle=True))

from src.cascade_sim import _apply_scenario

net2 = copy.deepcopy(net)
_apply_scenario(net2, weather, 0, 12, cfg)
pp.runpp(net2, algorithm="nr", max_iteration=30, tolerance_mva=1e-4)

print(f"Converged: {net2.converged}")
print(f"Voltage range: {net2.res_bus.vm_pu.min():.4f} – {net2.res_bus.vm_pu.max():.4f}")
print(f"Line loading range: {net2.res_line.loading_percent.min():.1f}% – {net2.res_line.loading_percent.max():.1f}%")
print(f"Lines > 50%: {(net2.res_line.loading_percent > 50).sum()}")
print(f"Lines > 80%: {(net2.res_line.loading_percent > 80).sum()}")
print(f"Lines > 100%: {(net2.res_line.loading_percent > 100).sum()}")
print(f"Voltage < 0.90: {(net2.res_bus.vm_pu < 0.90).sum()}")
print(f"Max line max_i_ka: {net2.line.max_i_ka.max():.3f}")
print(f"Min line max_i_ka: {net2.line.max_i_ka.min():.3f}")

# Check what happens after tripping a key line
most_loaded_line = net2.res_line.loading_percent.idxmax()
print(f"\nMost loaded line: {most_loaded_line} at {net2.res_line.loading_percent[most_loaded_line]:.1f}%")

net3 = copy.deepcopy(net2)
net3.line.at[most_loaded_line, "in_service"] = False
try:
    pp.runpp(net3, algorithm="nr", max_iteration=30, tolerance_mva=1e-4)
    print(f"After tripping: converged={net3.converged}")
    if net3.converged:
        print(f"  New max loading: {net3.res_line.loading_percent.max():.1f}%")
        print(f"  Voltage < 0.90: {(net3.res_bus.vm_pu < 0.90).sum()}")
except Exception as e:
    print(f"After tripping: FAILED - {e}")
