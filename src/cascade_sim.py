"""
Day 3 — Cascade Simulation Engine  (Algorithm 1 from paper)

Key mechanisms for realistic cascade generation:
  1. Inject weather-modified generation/load into the base case.
  2. Sample N-k contingency (line/gen outage), biased toward stressed elements.
  3. Remove element(s), re-run power flow.
  4. **Deterministic** trip: lines >100% loading, voltage violations.
  5. **Probabilistic** trip: lines >60% loading have p(trip) ∝ (loading-60)²,
     modelling hidden failures / protection mis-coordination.
  6. **Topology isolation**: any load bus disconnected from all generators
     immediately sheds its entire load.
  7. Repeat until stable or diverged.
  8. Record propagation depth, lines tripped, load shed.
  9. Label with severity σ ∈ {0, 1, 2, 3} (Eq. 5).
"""

import copy, os, pickle
import numpy as np
import pandapower as pp
import pandapower.topology as top
import networkx as nx
import yaml
from tqdm import trange


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Scenario injection ───────────────────────────────────────────────────────

def _apply_scenario(net: pp.pandapowerNet, weather: dict,
                    scen_idx: int, hour: int, cfg: dict) -> None:
    """Modify *net* in-place with renewable injections + load factor."""

    # --- load scaling --------------------------------------------------------
    lf = float(weather["load_factor"][scen_idx, hour])
    net.load["p_mw"]  = net.load["p_mw"]  * lf
    net.load["q_mvar"] = net.load["q_mvar"] * lf

    # --- wind injection ------------------------------------------------------
    wind_buses = weather["wind_buses"]
    for i, bus in enumerate(wind_buses):
        mw = float(weather["wind_power"][scen_idx, hour, i])
        mask = net.sgen["bus"] == bus
        if mask.any():
            net.sgen.loc[mask, "p_mw"] = mw
        else:
            pp.create_sgen(net, bus=bus, p_mw=mw, q_mvar=0.0,
                           name=f"wind_{bus}")

    # --- solar injection -----------------------------------------------------
    solar_buses = weather["solar_buses"]
    for i, bus in enumerate(solar_buses):
        mw = float(weather["solar_power"][scen_idx, hour, i])
        mask = net.sgen["bus"] == bus
        if mask.any():
            net.sgen.loc[mask, "p_mw"] = mw
        else:
            pp.create_sgen(net, bus=bus, p_mw=mw, q_mvar=0.0,
                           name=f"solar_{bus}")


# ── Contingency sampling ─────────────────────────────────────────────────────

def _sample_contingency(net: pp.pandapowerNet, rng: np.random.Generator):
    """
    Return list of (element_type, element_index) contingencies.
    Always N-1; 50% chance of N-2; 15% chance of N-3.
    Biased toward heavily loaded lines.
    """
    in_service_lines = net.line.index[net.line["in_service"]].tolist()
    in_service_gens  = net.gen.index[net.gen["in_service"]].tolist()

    contingencies = []
    if len(in_service_lines) == 0:
        return contingencies

    # loading-based weights (cubed for heavy bias)
    ll = net.res_line["loading_percent"]
    loadings = np.array([ll.get(i, 0.0) for i in in_service_lines], dtype=float)
    weights = loadings ** 3 + 0.1
    weights /= weights.sum()

    def _pick_line(exclude_set):
        remaining = [l for l in in_service_lines if l not in exclude_set]
        if not remaining:
            return None
        r_load = np.array([ll.get(i, 0.0) for i in remaining], dtype=float)
        w = r_load ** 3 + 0.1
        w /= w.sum()
        return rng.choice(remaining, p=w)

    # First contingency: 75% line, 25% generator
    if rng.random() < 0.75:
        idx = rng.choice(in_service_lines, p=weights)
        contingencies.append(("line", idx))
    else:
        if len(in_service_gens) > 0:
            # Bias toward larger generators for more impact
            gen_caps = np.array([net.gen.at[g, "max_p_mw"] for g in in_service_gens], dtype=float)
            gw = gen_caps ** 2 + 1.0
            gw /= gw.sum()
            idx = rng.choice(in_service_gens, p=gw)
            contingencies.append(("gen", idx))
        else:
            idx = rng.choice(in_service_lines, p=weights)
            contingencies.append(("line", idx))

    # N-2: 60% chance
    tripped_set = {c[1] for c in contingencies if c[0] == "line"}
    if rng.random() < 0.60:
        l2 = _pick_line(tripped_set)
        if l2 is not None:
            contingencies.append(("line", l2))
            tripped_set.add(l2)

    # N-3: 25% chance
    if rng.random() < 0.25:
        l3 = _pick_line(tripped_set)
        if l3 is not None:
            contingencies.append(("line", l3))

    return contingencies


# ── Element tripping ─────────────────────────────────────────────────────────

def _trip_element(net: pp.pandapowerNet, etype: str, eidx: int) -> None:
    if etype == "line":
        net.line.at[eidx, "in_service"] = False
    elif etype == "gen":
        net.gen.at[eidx, "in_service"] = False


# ── Violation checking (deterministic + probabilistic) ────────────────────────

def _check_violations(net: pp.pandapowerNet, cfg: dict,
                      rng: np.random.Generator):
    """
    Return lists of line indices and gen indices to trip.

    Two mechanisms:
      (a) Deterministic: lines >100% loading, voltage outside [vmin, vmax].
      (b) Probabilistic: lines between 60-100% loading have a probability
          of tripping proportional to (loading - 60)^2 / (40^2).
          This models hidden-failure / protection mis-coordination.
    """
    vmin, vmax = cfg["grid"]["voltage_limits"]
    line_thr = cfg["grid"]["line_overload_threshold"] * 100  # 100%
    prob_threshold = 40.0   # loading % above which probabilistic trip starts
    prob_scale     = 0.45   # max probability at 100% loading

    violated_lines = []
    violated_gens  = []

    # --- line overloads (deterministic + probabilistic) ----------------------
    ll = net.res_line["loading_percent"]
    for idx in net.line.index[net.line["in_service"]]:
        if idx not in ll.index:
            continue
        loading = ll[idx]
        if loading > line_thr:
            # Deterministic trip
            violated_lines.append(idx)
        elif loading > prob_threshold:
            # Probabilistic trip: p ∝ ((loading - threshold) / range)^2
            frac = (loading - prob_threshold) / (line_thr - prob_threshold)
            p_trip = prob_scale * frac * frac
            if rng.random() < p_trip:
                violated_lines.append(idx)

    # --- voltage violations → trip generators at those buses -----------------
    vm = net.res_bus["vm_pu"]
    for idx in net.gen.index[net.gen["in_service"]]:
        bus = net.gen.at[idx, "bus"]
        if bus in vm.index:
            v = vm[bus]
            if v < vmin or v > vmax:
                violated_gens.append(idx)

    return violated_lines, violated_gens


# ── Topology-based load shedding ─────────────────────────────────────────────

def _shed_isolated_loads(net: pp.pandapowerNet) -> float:
    """
    Detect buses that are disconnected from ALL generators (including ext_grid).
    Set loads at those buses to zero (they are blacked out).
    Returns MW of load shed by isolation.
    """
    # Build graph of in-service topology
    try:
        mg = top.create_nxgraph(net, respect_switches=True)
    except Exception:
        return 0.0

    # Collect all generator buses (gen + ext_grid)
    gen_buses = set()
    for idx in net.gen.index[net.gen["in_service"]]:
        gen_buses.add(int(net.gen.at[idx, "bus"]))
    for idx in net.ext_grid.index[net.ext_grid["in_service"]]:
        gen_buses.add(int(net.ext_grid.at[idx, "bus"]))

    # Find all buses reachable from any generator bus
    powered_buses = set()
    for gb in gen_buses:
        if gb in mg:
            powered_buses.update(nx.node_connected_component(mg, gb))

    # Shed load at isolated buses
    shed_mw = 0.0
    for idx in net.load.index:
        bus = int(net.load.at[idx, "bus"])
        if bus not in powered_buses:
            shed_mw += net.load.at[idx, "p_mw"]
            net.load.at[idx, "p_mw"]   = 0.0
            net.load.at[idx, "q_mvar"] = 0.0

    # Also disable generators at isolated buses
    for idx in net.gen.index[net.gen["in_service"]]:
        bus = int(net.gen.at[idx, "bus"])
        if bus not in powered_buses:
            net.gen.at[idx, "in_service"] = False

    # Disable sgens at isolated buses
    for idx in net.sgen.index[net.sgen["in_service"]]:
        bus = int(net.sgen.at[idx, "bus"])
        if bus not in powered_buses:
            net.sgen.at[idx, "in_service"] = False

    return shed_mw


# ── Progressive load shedding on PF divergence ──────────────────────────────

def _run_pf_with_shedding(net: pp.pandapowerNet, max_shed_steps: int = 5) -> bool:
    """
    Run power flow; if it diverges, progressively shed 15% of remaining load
    at the most stressed buses until convergence or max_shed_steps.

    Returns True if PF eventually converges, False otherwise.
    """
    # First try normal PF
    try:
        pp.runpp(net, algorithm="nr", max_iteration=30, tolerance_mva=1e-4)
        if net.converged:
            return True
    except Exception:
        pass

    # Try backup algorithm
    try:
        pp.runpp(net, algorithm="bfsw", max_iteration=50)
        if net.converged:
            return True
    except Exception:
        pass

    # Progressive load shedding: shed load at random buses until convergence
    for step in range(max_shed_steps):
        # Shed 15% of remaining total load, distributed across all buses
        shed_frac = 0.15
        net.load["p_mw"]   *= (1.0 - shed_frac)
        net.load["q_mvar"] *= (1.0 - shed_frac)

        try:
            pp.runpp(net, algorithm="nr", max_iteration=30, tolerance_mva=1e-4)
            if net.converged:
                return True
        except Exception:
            pass

        try:
            pp.runpp(net, algorithm="bfsw", max_iteration=50)
            if net.converged:
                return True
        except Exception:
            continue

    return False  # still diverged after shedding


# ── Main cascade simulation ─────────────────────────────────────────────────

def simulate_cascade(base_net: pp.pandapowerNet, weather: dict,
                     scen_idx: int, hour: int, cfg: dict,
                     rng: np.random.Generator) -> dict:
    """
    Run one cascade simulation with probabilistic tripping and
    topology-based load shedding.

    Returns a dict with:
      converged       bool
      n_iterations    int       cascade depth
      lines_tripped   list[int]
      gens_tripped    list[int]
      load_shed_frac  float     fraction of total load lost
      severity        int       0–3
      initial_contingency  list[(etype, eidx)]
    """
    net = copy.deepcopy(base_net)
    _apply_scenario(net, weather, scen_idx, hour, cfg)

    total_load = max(net.load["p_mw"].sum(), 1.0)

    # Initial power flow before contingency
    if not _run_pf_with_shedding(net, max_shed_steps=2):
        return dict(converged=False, n_iterations=0, lines_tripped=[],
                    gens_tripped=[], load_shed_frac=1.0, severity=3,
                    initial_contingency=[(None, None)])

    # Sample and apply initial contingencies
    contingencies = _sample_contingency(net, rng)
    if not contingencies:
        return dict(converged=True, n_iterations=0, lines_tripped=[],
                    gens_tripped=[], load_shed_frac=0.0, severity=0,
                    initial_contingency=[(None, None)])

    lines_tripped = []
    gens_tripped  = []
    for etype, eidx in contingencies:
        _trip_element(net, etype, eidx)
        if etype == "line":
            lines_tripped.append(eidx)
        else:
            gens_tripped.append(eidx)

    max_iter = cfg["cascade"]["max_iterations"]
    total_isolated_shed = 0.0

    for iteration in range(1, max_iter + 1):
        # --- topology-based load shedding first ---
        iso_shed = _shed_isolated_loads(net)
        total_isolated_shed += iso_shed

        # --- power flow (with progressive load shedding on divergence) ---
        converged = _run_pf_with_shedding(net)

        if not converged:
            return dict(converged=False, n_iterations=iteration,
                        lines_tripped=lines_tripped,
                        gens_tripped=gens_tripped,
                        load_shed_frac=1.0, severity=3,
                        initial_contingency=contingencies)

        # --- check violations (deterministic + probabilistic) ---
        viol_lines, viol_gens = _check_violations(net, cfg, rng)
        if not viol_lines and not viol_gens:
            break  # stable

        for li in viol_lines:
            _trip_element(net, "line", li)
            lines_tripped.append(li)
        for gi in viol_gens:
            _trip_element(net, "gen", gi)
            gens_tripped.append(gi)

    # Final topology check
    iso_shed_final = _shed_isolated_loads(net)
    total_isolated_shed += iso_shed_final

    # Compute load shed
    remaining_load = 0.0
    if net.converged:
        try:
            remaining_load = net.res_load["p_mw"].sum()
        except Exception:
            remaining_load = 0.0

    # Total load shed = explicit PF reduction + isolated loads
    load_shed_frac = max(0.0, 1.0 - remaining_load / total_load)

    # Severity labelling (Eq. 5)
    s1 = cfg["cascade"]["severity_thresholds"]["sigma_1"]
    s2 = cfg["cascade"]["severity_thresholds"]["sigma_2"]
    if load_shed_frac < cfg["cascade"]["load_shed_eps"] / total_load:
        severity = 0
    elif load_shed_frac < s1:
        severity = 1
    elif load_shed_frac < s2:
        severity = 2
    else:
        severity = 3

    return dict(
        converged=net.converged,
        n_iterations=iteration if contingencies else 0,
        lines_tripped=lines_tripped,
        gens_tripped=gens_tripped,
        load_shed_frac=load_shed_frac,
        severity=severity,
        initial_contingency=contingencies,
    )


# ── Batch runner ─────────────────────────────────────────────────────────────

def run_all_cascades(base_net, weather, cfg, hour: int = 12):
    """
    Simulate cascades for ALL scenarios at a fixed hour.

    Returns:
      results   list[dict]   one per scenario
    """
    N = cfg["weather"]["n_scenarios"]
    rng = np.random.default_rng(cfg["dataset"]["random_seed"])
    results = []

    for i in trange(N, desc="Cascade simulation"):
        r = simulate_cascade(base_net, weather, i, hour, cfg, rng)
        results.append(r)

    # Summary
    severities = [r["severity"] for r in results]
    cascade_pos = sum(1 for s in severities if s > 0)
    rate = cascade_pos / N
    print(f"\n[cascade] {cascade_pos}/{N} cascade-positive  ({rate*100:.1f}%)")
    for s in range(4):
        cnt = severities.count(s)
        print(f"  σ={s}: {cnt}  ({cnt/N*100:.1f}%)")

    # Detailed stats
    load_sheds = [r["load_shed_frac"] for r in results if r["severity"] > 0]
    if load_sheds:
        print(f"  Load shed (cascade+): mean={np.mean(load_sheds):.3f}, "
              f"max={np.max(load_sheds):.3f}")

    return results


def save_cascade_results(results: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    severities = np.array([r["severity"] for r in results], dtype=np.int32)
    load_shed = np.array([r["load_shed_frac"] for r in results], dtype=np.float32)
    n_iters = np.array([r["n_iterations"] for r in results], dtype=np.int32)
    converged = np.array([r["converged"] for r in results], dtype=bool)

    max_lt = max(len(r["lines_tripped"]) for r in results) or 1
    max_gt = max(len(r["gens_tripped"]) for r in results) or 1
    lt = np.full((len(results), max_lt), -1, dtype=np.int32)
    gt = np.full((len(results), max_gt), -1, dtype=np.int32)
    for i, r in enumerate(results):
        lt[i, :len(r["lines_tripped"])] = r["lines_tripped"]
        gt[i, :len(r["gens_tripped"])] = r["gens_tripped"]

    np.savez_compressed(path,
                        severity=severities,
                        load_shed_frac=load_shed,
                        n_iterations=n_iters,
                        converged=converged,
                        lines_tripped=lt,
                        gens_tripped=gt)
    print(f"[cascade] Results saved → {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.grid_setup import load_base_case

    cfg = load_config()
    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))

    results = run_all_cascades(net, weather, cfg, hour=12)
    save_cascade_results(results, cfg["paths"]["cascade_results"])
    print("[cascade] ✅ Day 3 complete.")
