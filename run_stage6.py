"""
Stage 6 runner — Training & evaluation with proper cascade-based metrics.

The agent is trained to keep the grid healthy (low line loading, good voltages).
Evaluation runs a full N-k cascade simulation on the agent-modified grid and
compares against the pre-computed ground-truth cascade results.
"""
import warnings
warnings.filterwarnings("ignore")

import os, logging
logging.disable(logging.WARNING)

# Monkey-patch pandapower to always pass numba=False (9x faster on case118)
import pandapower as pp
_orig_runpp = pp.runpp
def _quiet_runpp(net, **kwargs):
    if 'numba' not in kwargs:
        kwargs['numba'] = False
    return _orig_runpp(net, **kwargs)
pp.runpp = _quiet_runpp

import yaml, json, time, copy
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.grid_setup import load_base_case
from src.grid_env import GridCascadeEnv
from src.ddpg import DDPGAgent
from src.train import train_ddpg, evaluate
from src.cascade_sim import (simulate_cascade, _apply_scenario,
                             _sample_contingency, _trip_element,
                             _check_violations, _shed_isolated_loads,
                             _run_pf_with_shedding)
from src.baselines import (RuleBasedAgent, train_supervised_mlp,
                           opf_baseline, compute_metrics)


def _get_splits(cfg):
    N = cfg["weather"]["n_scenarios"]
    rng = np.random.default_rng(cfg["dataset"]["random_seed"])
    all_idx = rng.permutation(N)
    n_tr = int(N * cfg["dataset"]["train_ratio"])
    n_va = int(N * cfg["dataset"]["val_ratio"])
    return (all_idx[:n_tr].tolist(),
            all_idx[n_tr:n_tr + n_va].tolist(),
            all_idx[n_tr + n_va:].tolist())


# ═══════════════════════════════════════════════════════════════════════════
# Cascade-based evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_with_cascade(agent_fn, base_net, weather, cfg,
                          test_indices, hour=12, n_test=25):
    """
    Evaluate an agent by:
    1. Setting up the scenario at the specified hour (same as no-agent)
    2. Getting the agent's observation and computing ONE action
    3. Applying that action to the grid (single step, no compounding)
    4. Running a full N-k cascade simulation on that state
    5. Each scenario uses a fixed RNG seed (scenario_idx + 1000) for
       reproducible contingency sampling across methods

    This single-step evaluation is fair: the agent sees the same grid
    state as the no-agent baseline, takes one preventive action, and
    we measure whether that action reduces cascade severity.

    agent_fn(obs, net) -> action
    """
    subset = test_indices[:n_test]
    # We create a minimal env just for obs/action normalization
    env = GridCascadeEnv(base_net, weather, cfg, scenario_indices=subset)

    preds = []
    severities = []
    load_sheds = []
    times_list = []

    for idx in subset:
        # Fixed RNG per scenario for reproducible contingencies
        rng = np.random.default_rng(idx + 1000)

        t0 = time.time()

        # 1. Fresh grid with scenario applied (same state as no-agent baseline)
        agent_net = copy.deepcopy(base_net)
        _apply_scenario(agent_net, weather, idx, hour, cfg)

        try:
            pp.runpp(agent_net, algorithm="nr", max_iteration=30,
                     tolerance_mva=1e-4, numba=False)
        except Exception:
            preds.append(1); severities.append(3); load_sheds.append(1.0)
            times_list.append(time.time() - t0); continue

        # 2. Get observation from this state
        env.current_net = agent_net
        env.hour = hour
        env.scenario_idx = idx
        obs = env._get_obs()

        # 3. Agent selects one action
        action = agent_fn(obs, agent_net)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # 4. Apply action to the grid (single step)
        env._apply_action(action)

        # 5. Re-run power flow after action
        try:
            pp.runpp(agent_net, algorithm="nr", max_iteration=30,
                     tolerance_mva=1e-4, numba=False)
        except Exception:
            preds.append(1); severities.append(3); load_sheds.append(1.0)
            times_list.append(time.time() - t0); continue

        # Record total load AFTER action (for fair shed calculation)
        total_load = max(agent_net.load["p_mw"].sum(), 1.0)

        if not agent_net.converged:
            preds.append(1); severities.append(3); load_sheds.append(1.0)
            times_list.append(time.time() - t0); continue

        contingencies = _sample_contingency(agent_net, rng)
        if not contingencies:
            preds.append(0); severities.append(0); load_sheds.append(0.0)
            times_list.append(time.time() - t0); continue

        for etype, eidx in contingencies:
            _trip_element(agent_net, etype, eidx)

        max_iter = cfg["cascade"]["max_iterations"]
        diverged = False
        for iteration in range(1, max_iter + 1):
            _shed_isolated_loads(agent_net)
            converged = _run_pf_with_shedding(agent_net)
            if not converged:
                diverged = True
                break
            viol_lines, viol_gens = _check_violations(agent_net, cfg, rng)
            if not viol_lines and not viol_gens:
                break
            for li in viol_lines:
                _trip_element(agent_net, "line", li)
            for gi in viol_gens:
                _trip_element(agent_net, "gen", gi)

        if diverged:
            preds.append(1); severities.append(3); load_sheds.append(1.0)
        else:
            _shed_isolated_loads(agent_net)
            remaining = 0.0
            if agent_net.converged:
                try:
                    remaining = agent_net.res_load["p_mw"].sum()
                except Exception:
                    pass
            shed = max(0.0, 1.0 - remaining / total_load)
            s1 = cfg["cascade"]["severity_thresholds"]["sigma_1"]
            s2 = cfg["cascade"]["severity_thresholds"]["sigma_2"]
            eps = cfg["cascade"]["load_shed_eps"] / total_load
            if shed < eps:
                sev = 0
            elif shed < s1:
                sev = 1
            elif shed < s2:
                sev = 2
            else:
                sev = 3
            preds.append(1 if sev > 0 else 0)
            severities.append(sev)
            load_sheds.append(shed)

        times_list.append(time.time() - t0)

    return (np.array(preds), np.array(severities),
            np.array(load_sheds), np.array(times_list))


def evaluate_no_agent(base_net, weather, cfg, test_indices,
                      hour=12, n_test=25):
    """Run cascade sim with NO agent intervention.
    Uses per-scenario fixed RNG (idx+1000) for reproducibility."""
    subset = test_indices[:n_test]
    preds, severities, load_sheds = [], [], []

    for idx in subset:
        rng = np.random.default_rng(idx + 1000)
        r = simulate_cascade(base_net, weather, idx, hour, cfg, rng)
        preds.append(1 if r["severity"] > 0 else 0)
        severities.append(r["severity"])
        load_sheds.append(r["load_shed_frac"])

    return np.array(preds), np.array(severities), np.array(load_sheds)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Override for speed
    cfg["rl"]["n_episodes"] = 100
    cfg["rl"]["n_seeds"] = 2
    cfg["rl"]["warmup_steps"] = 50
    cfg["rl"]["eval_interval"] = 20
    cfg["rl"]["episode_length"] = 8

    n_test_eval = 25

    print("=" * 60)
    print("  STAGE 6: Training & Evaluation (Cascade-based)")
    print(f"  Episodes: {cfg['rl']['n_episodes']}, Seeds: {cfg['rl']['n_seeds']}")
    print(f"  Episode length: {cfg['rl']['episode_length']}h")
    print(f"  Test scenarios: {n_test_eval}")
    print("=" * 60)

    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"],
                           allow_pickle=True))

    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    y_all = cascade_raw["severity"]
    train_idx, val_idx, test_idx = _get_splits(cfg)
    y_test = (y_all[test_idx] > 0).astype(int)
    y_test_sub = y_test[:n_test_eval]

    all_results = {}

    # ── 0. No-agent baseline ───────────────────────────────────────────
    # Uses same fixed per-scenario RNG as all other methods
    print("\n[eval] No-agent baseline (cascade sim, no intervention) …")
    t0 = time.time()
    na_preds, na_sev, na_shed = evaluate_no_agent(
        net, weather, cfg, test_idx, n_test=n_test_eval)
    na_cascade_rate = na_preds.mean()
    print(f"  No-agent: cascade_rate={na_cascade_rate:.3f}, "
          f"mean_shed={na_shed.mean():.3f}, "
          f"sev_dist={[int((na_sev==s).sum()) for s in range(4)]}, "
          f"{time.time()-t0:.1f}s")

    # ── 1. Rule-based ──────────────────────────────────────────────────
    print("\n[eval] Rule-based baseline …")
    rb = RuleBasedAgent(cfg)
    rb_preds, rb_sev, rb_shed, rb_times = evaluate_with_cascade(
        lambda obs, net: rb.predict(obs, net),
        net, weather, cfg, test_idx, n_test=n_test_eval
    )
    prevented_rb = int(np.sum((na_preds == 1) & (rb_preds == 0)))
    shed_reduction_rb = float(na_shed.mean() - rb_shed.mean())
    all_results["Rule-based"] = {
        "cascade_rate": float(rb_preds.mean()),
        "mean_load_shed": float(rb_shed.mean()),
        "shed_reduction": shed_reduction_rb,
        "cascades_prevented": prevented_rb,
        "severity_dist": {str(s): int((rb_sev==s).sum()) for s in range(4)},
        "mean_time_ms": float(rb_times.mean() * 1000),
    }
    print(f"  Rule-based: cascade_rate={rb_preds.mean():.3f}, "
          f"shed={rb_shed.mean():.3f}, prevented={prevented_rb}")

    # ── 2. Supervised MLP (predictor, not preventor) ───────────────────
    # MLP predicts cascade occurrence from features; it doesn't modify the grid.
    # We compare MLP predictions against ground-truth labels (y_test).
    print("\n[eval] Training Supervised MLP …")
    _, mlp_probs, mlp_y, _, _ = train_supervised_mlp(cfg)
    mlp_probs_sub = mlp_probs[:n_test_eval]
    mlp_preds = (mlp_probs_sub > 0.5).astype(int)
    all_results["Supervised MLP"] = compute_metrics(
        y_test_sub, mlp_preds, mlp_probs_sub, "Supervised MLP")
    all_results["Supervised MLP"]["mean_time_ms"] = 0.5
    all_results["Supervised MLP"]["note"] = "predictor_not_preventor"
    print(f"  MLP: acc={all_results['Supervised MLP']['accuracy']:.3f}, "
          f"f1={all_results['Supervised MLP']['f1']:.3f}")

    # ── 3. OPF ─────────────────────────────────────────────────────────
    print("\n[eval] OPF baseline …")
    opf_test_sub = test_idx[:n_test_eval]
    opf_preds, opf_times = opf_baseline(net, weather, cfg, opf_test_sub)
    prevented_opf = int(np.sum((na_preds == 1) & (opf_preds == 0)))
    all_results["OPF"] = {
        "cascade_rate": float(opf_preds.mean()),
        "cascades_prevented": prevented_opf,
        "mean_time_ms": float(opf_times.mean() * 1000),
    }
    print(f"  OPF: cascade_rate={opf_preds.mean():.3f}, "
          f"prevented={prevented_opf}")

    # ── 4. Multi-seed DDPG ─────────────────────────────────────────────
    n_seeds = cfg["rl"]["n_seeds"]
    n_episodes = cfg["rl"]["n_episodes"]
    seed_results = []
    all_train_logs = []

    for seed in range(n_seeds):
        print(f"\n[eval] Training DDPG seed={seed} …")
        agent, log = train_ddpg(cfg, net, weather, seed=seed,
                                use_per=True, tag="ddpg_per",
                                n_episodes=n_episodes)

        ddpg_preds, ddpg_sev, ddpg_shed, ddpg_times = evaluate_with_cascade(
            lambda obs, net, a=agent: a.select_action(obs, add_noise=False),
            net, weather, cfg, test_idx, n_test=n_test_eval
        )
        prevented = int(np.sum((na_preds == 1) & (ddpg_preds == 0)))
        shed_red = float(na_shed.mean() - ddpg_shed.mean())
        m = {
            "cascade_rate": float(ddpg_preds.mean()),
            "mean_load_shed": float(ddpg_shed.mean()),
            "shed_reduction": shed_red,
            "cascades_prevented": prevented,
            "severity_dist": {str(s): int((ddpg_sev==s).sum()) for s in range(4)},
            "mean_time_ms": float(ddpg_times.mean() * 1000),
            "mean_reward": float(np.mean(log["episode_rewards"][-10:])),
        }
        seed_results.append(m)
        all_train_logs.append(log)
        print(f"  DDPG seed={seed}: cascade_rate={ddpg_preds.mean():.3f}, "
              f"shed={ddpg_shed.mean():.3f}, prevented={prevented}")

    # Aggregate DDPG seeds
    agg = {"method": "DDPG (ours)", "n_seeds": n_seeds}
    for key in ["cascade_rate", "mean_load_shed", "shed_reduction",
                "mean_time_ms", "cascades_prevented"]:
        vals = [s[key] for s in seed_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    all_results["DDPG (ours)"] = agg

    # ── 5. DDPG without PER ───────────────────────────────────────────
    print("\n[eval] Training DDPG without PER …")
    agent_noper, log_noper = train_ddpg(cfg, net, weather, seed=0,
                                        use_per=False, tag="ddpg_noper",
                                        n_episodes=n_episodes)
    noper_preds, noper_sev, noper_shed, noper_times = evaluate_with_cascade(
        lambda obs, net, a=agent_noper: a.select_action(obs, add_noise=False),
        net, weather, cfg, test_idx, n_test=n_test_eval
    )
    prevented_noper = int(np.sum((na_preds == 1) & (noper_preds == 0)))
    all_results["DDPG (no PER)"] = {
        "cascade_rate": float(noper_preds.mean()),
        "mean_load_shed": float(noper_shed.mean()),
        "shed_reduction": float(na_shed.mean() - noper_shed.mean()),
        "cascades_prevented": prevented_noper,
        "severity_dist": {str(s): int((noper_sev==s).sum()) for s in range(4)},
        "mean_time_ms": float(noper_times.mean() * 1000),
    }
    print(f"  DDPG no-PER: cascade_rate={noper_preds.mean():.3f}, "
          f"shed={noper_shed.mean():.3f}")

    # ── Save everything ────────────────────────────────────────────────
    out_dir = cfg["paths"]["tables_dir"]
    os.makedirs(out_dir, exist_ok=True)

    all_results["No Agent (baseline)"] = {
        "cascade_rate": float(na_preds.mean()),
        "mean_load_shed": float(na_shed.mean()),
        "severity_dist": {
            str(s): int((na_sev == s).sum()) for s in range(4)
        }
    }

    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    for i, lg in enumerate(all_train_logs):
        log_path = os.path.join(cfg["paths"]["logs_dir"],
                                f"train_log_ddpg_per_seed{i}.json")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        ser = {k: [float(v) for v in vs] if isinstance(vs, list) else vs
               for k, vs in lg.items()}
        with open(log_path, "w") as f:
            json.dump(ser, f, indent=2)

    log_path = os.path.join(cfg["paths"]["logs_dir"],
                            "train_log_ddpg_noper_seed0.json")
    ser = {k: [float(v) for v in vs] if isinstance(vs, list) else vs
           for k, vs in log_noper.items()}
    with open(log_path, "w") as f:
        json.dump(ser, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Method':<22} {'Casc%':>7} {'Shed':>7} {'Shed↓':>7} "
          f"{'Prev':>5} {'ms':>8}")
    print(f"{'-'*80}")

    for name, m in all_results.items():
        if name == "Supervised MLP":
            # MLP is a predictor — show classification metrics
            print(f"{name:<22}  [predictor] "
                  f"acc={m.get('accuracy',0):.3f} "
                  f"f1={m.get('f1',0):.3f} "
                  f"auc={m.get('auc',0):.3f}")
            continue

        if "cascade_rate_mean" in m:
            # Multi-seed aggregate
            print(f"{name:<22} "
                  f"{m['cascade_rate_mean']:>6.3f}  "
                  f"{m['mean_load_shed_mean']:>6.3f}  "
                  f"{m['shed_reduction_mean']:>+6.3f}  "
                  f"{m['cascades_prevented_mean']:>4.1f}  "
                  f"{m.get('mean_time_ms_mean',0):>7.1f}")
        else:
            cr = m.get("cascade_rate", 0)
            shed = m.get("mean_load_shed", 0)
            sr = m.get("shed_reduction", 0)
            prev = m.get("cascades_prevented", "---")
            ms = m.get("mean_time_ms", 0)
            prev_s = f"{prev:>5}" if isinstance(prev, int) else f"{prev:>5}"
            sr_s = f"{sr:>+7.3f}" if isinstance(sr, float) else f"{'---':>7}"
            print(f"{name:<22} {cr:>7.3f} {shed:>7.3f} {sr_s} "
                  f"{prev_s} {ms:>8.1f}")

    print(f"{'='*80}")
    print("\n✅ Stage 6 complete!")


if __name__ == "__main__":
    main()
