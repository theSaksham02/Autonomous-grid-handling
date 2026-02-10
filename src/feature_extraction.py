"""
Day 4 — Feature Extraction & Dataset Construction
Extract state vectors (Eq. 6), normalize, stratified-split, save as .npz.
"""

import os
import numpy as np
import yaml
import copy
import pandapower as pp
from tqdm import trange


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _moving_average(arr, window=3):
    """Simple causal moving average along axis=0."""
    out = np.copy(arr)
    for i in range(1, len(arr)):
        lo = max(0, i - window + 1)
        out[i] = arr[lo:i + 1].mean(axis=0)
    return out


def extract_features(base_net, weather, cascade_results, cfg, hour=12):
    """
    For each scenario build the state vector:
      [voltages(118), line_loadings(186), gen_outputs(54),
       weather(24), voltage_trends(118), time_encoding(2)]
    Returns:
      X  (N, D)  feature matrix
      y  (N,)    severity labels
    """
    N = cfg["weather"]["n_scenarios"]
    n_bus = cfg["features"]["n_voltage"]        # 118
    n_line = cfg["features"]["n_line_loading"]  # 186
    n_gen = cfg["features"]["n_gen_output"]     # 54
    n_weather_feat = cfg["features"]["n_weather"]  # 24
    n_trend = cfg["features"]["n_trend"]        # 118
    n_time = cfg["features"]["n_time"]          # 2

    D = n_bus + n_line + n_gen + n_weather_feat + n_trend + n_time

    X = np.zeros((N, D), dtype=np.float32)
    y = np.array([r["severity"] if isinstance(r, dict) else int(r)
                  for r in cascade_results], dtype=np.int32)

    # We need to re-run power flow per scenario to grab telemetry BEFORE the
    # contingency (i.e., the observation the agent would see).
    from src.cascade_sim import _apply_scenario

    print(f"[features] Extracting {D}-dim state vectors for {N} scenarios …")

    hours_arr = np.arange(24, dtype=float)
    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)

    for i in trange(N, desc="Feature extraction"):
        net = copy.deepcopy(base_net)
        _apply_scenario(net, weather, i, hour, cfg)

        try:
            pp.runpp(net, algorithm="nr", max_iteration=30, tolerance_mva=1e-4)
        except Exception:
            pass  # leave zeros – will be an outlier but not NaN

        offset = 0

        # --- voltages --------------------------------------------------------
        if net.converged:
            vm = net.res_bus["vm_pu"].values[:n_bus]
        else:
            vm = np.ones(n_bus)
        X[i, offset:offset + n_bus] = vm
        offset += n_bus

        # --- line loadings (fraction) ----------------------------------------
        if net.converged:
            ll = net.res_line["loading_percent"].values[:n_line] / 100.0
        else:
            ll = np.zeros(n_line)
        X[i, offset:offset + n_line] = ll
        offset += n_line

        # --- generator active power ------------------------------------------
        if net.converged:
            pg = net.res_gen["p_mw"].values
            if len(pg) < n_gen:
                pg = np.concatenate([pg, np.zeros(n_gen - len(pg))])
            pg = pg[:n_gen]
        else:
            pg = np.zeros(n_gen)
        X[i, offset:offset + n_gen] = pg
        offset += n_gen

        # --- weather features (6 raw × 4 look-ahead horizons) ----------------
        # Flatten: [wind_mean, ghi_mean, temp, load_factor] for h, h+1, …, h+3
        wfeat = np.zeros(n_weather_feat)
        for k, h in enumerate(range(hour, min(hour + 4, 24))):
            base = k * 6
            wfeat[base + 0] = weather["wind_speed"][i, h].mean()
            wfeat[base + 1] = weather["ghi"][i, h].mean()
            wfeat[base + 2] = weather["temperature"][i, h]
            wfeat[base + 3] = weather["load_factor"][i, h]
            wfeat[base + 4] = weather["wind_power"][i, h].mean()
            wfeat[base + 5] = weather["solar_power"][i, h].mean()
        X[i, offset:offset + n_weather_feat] = wfeat
        offset += n_weather_feat

        # --- voltage trend (moving avg of last 3 hours) ----------------------
        # Approximate by perturbing: use current voltages as proxy
        X[i, offset:offset + n_trend] = vm  # simple proxy
        offset += n_trend

        # --- time encoding ---------------------------------------------------
        X[i, offset] = sin_hour
        X[i, offset + 1] = cos_hour

    # Replace NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)

    print(f"[features] X shape: {X.shape},  y shape: {y.shape}")
    print(f"[features] Class counts: { {s: int((y==s).sum()) for s in range(4)} }")

    return X, y


# ── Normalisation ────────────────────────────────────────────────────────────

def normalize(X_train, X_val, X_test):
    """Z-score normalisation fitted on training set only."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma, mu, sigma


# ── Stratified Split ─────────────────────────────────────────────────────────

def stratified_split(X, y, cfg):
    """Split preserving class ratios."""
    from sklearn.model_selection import train_test_split

    seed = cfg["dataset"]["random_seed"]
    val_ratio = cfg["dataset"]["val_ratio"]
    test_ratio = cfg["dataset"]["test_ratio"]

    # first split: train vs (val+test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_ratio + test_ratio,
        stratify=y, random_state=seed
    )
    # second split: val vs test
    rel_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test,
        stratify=y_tmp, random_state=seed
    )

    print(f"[dataset] Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_datasets(X_tr, y_tr, X_v, y_v, X_te, y_te, mu, sigma, cfg):
    for arr, path in [
        (dict(X=X_tr, y=y_tr), cfg["paths"]["train_data"]),
        (dict(X=X_v, y=y_v), cfg["paths"]["val_data"]),
        (dict(X=X_te, y=y_te), cfg["paths"]["test_data"]),
        (dict(mu=mu, sigma=sigma), cfg["paths"]["norm_stats"]),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arr)
    print("[dataset] All splits saved.")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle
    from src.grid_setup import load_base_case

    cfg = load_config()
    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)

    # Reconstruct list-of-dicts for cascade results
    severities = cascade_raw["severity"]
    cascade_results = [{"severity": int(s)} for s in severities]

    X, y = extract_features(net, weather, cascade_results, cfg)

    X_tr, y_tr, X_v, y_v, X_te, y_te = stratified_split(X, y, cfg)
    X_tr, X_v, X_te, mu, sigma = normalize(X_tr, X_v, X_te)

    save_datasets(X_tr, y_tr, X_v, y_v, X_te, y_te, mu, sigma, cfg)

    # Sanity
    assert not np.isnan(X_tr).any(), "NaN in training data!"
    print("[features] ✅ Day 4 complete.")
