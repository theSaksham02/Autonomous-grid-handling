"""
Day 1 — Grid Setup
Load IEEE 118-bus network, run base-case AC power flow, verify topology,
optionally scale loads to create stress, and persist the base case.
"""

import os, pickle, yaml
import numpy as np
import pandapower as pp
import pandapower.networks as pn


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_base_case(cfg: dict, verbose: bool = True) -> pp.pandapowerNet:
    """Return a solved IEEE 118-bus network with scaled loading."""
    net = pn.case118()

    # --- scale loads to introduce grid stress --------------------------------
    scale = cfg["grid"]["base_loading_scale"]
    net.load["p_mw"] *= scale
    net.load["q_mvar"] *= scale

    # --- reduce line thermal ratings to create realistic stress ---------------
    # IEEE 118-bus default ratings are extremely generous; scale them down
    # so that contingencies actually cause overloads.
    # Target: base case loading at 30-60% so N-1/N-2 push some lines over 100%.
    net.line["max_i_ka"] *= 0.04   # reduce to 4% of default ratings
    if verbose:
        print(f"[grid] Loads scaled to {scale*100:.0f}% of nominal.")
        print(f"[grid] Line ratings reduced to 4% of default.")

    # --- reduce generator headroom -------------------------------------------
    # Default gen max_p_mw are extremely large; reduce to 120% of dispatch
    # so that generators cannot easily absorb all redistributed power.
    # This must happen AFTER the initial power flow to know actual dispatch.
    try:
        pp.runpp(net, algorithm="nr", max_iteration=50, tolerance_mva=1e-6)
    except Exception:
        pp.runpp(net, algorithm="nr", max_iteration=100, tolerance_mva=1e-4,
                 enforce_q_lims=False)

    if net.converged:
        # Limit generator max_p_mw to 130% of current dispatch
        for idx in net.gen.index:
            dispatch = net.res_gen.at[idx, "p_mw"]
            if dispatch > 0:
                net.gen.at[idx, "max_p_mw"] = dispatch * 1.30
            else:
                net.gen.at[idx, "max_p_mw"] = max(5.0, abs(dispatch) * 1.30)
        if verbose:
            print(f"[grid] Generator max_p_mw capped at 130% of dispatch.")

    # --- run AC power flow ---------------------------------------------------
    try:
        pp.runpp(net, algorithm="nr", max_iteration=50, tolerance_mva=1e-6)
    except Exception:
        # try with relaxed settings
        pp.runpp(net, algorithm="nr", max_iteration=100, tolerance_mva=1e-4,
                 enforce_q_lims=False)
    assert net.converged, "Base-case power flow did NOT converge!"

    # --- print summary -------------------------------------------------------
    n_bus = len(net.bus)
    n_line = len(net.line)
    n_gen = len(net.gen) + len(net.ext_grid)   # ext_grid acts as slack gen
    if verbose:
        print(f"[grid] {n_bus} buses, {n_line} lines, {n_gen} generators")
        vm = net.res_bus["vm_pu"]
        print(f"[grid] Voltage range: {vm.min():.4f} – {vm.max():.4f} pu")
        ll = net.res_line["loading_percent"]
        print(f"[grid] Line loading range: {ll.min():.1f}% – {ll.max():.1f}%")
        overloads = (ll > 100).sum()
        print(f"[grid] Lines > 100%: {overloads}")

    return net


def save_base_case(net: pp.pandapowerNet, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(net, f)
    print(f"[grid] Base case saved → {path}")


def load_base_case(path: str) -> pp.pandapowerNet:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    net = setup_base_case(cfg)
    save_base_case(net, cfg["paths"]["base_case"])
    print("[grid] ✅ Day 1 complete.")
