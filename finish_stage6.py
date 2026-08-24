"""
Finish Stage 6 — re-evaluate all trained models and save correct results.
The main run was interrupted after no-PER training completed.
All models are trained and saved; we just need cascade evaluation + summary.

Usage:
  python finish_stage6.py --n-test 150
  python finish_stage6.py --n-test 150 --skip-opf
"""
import warnings
warnings.filterwarnings("ignore")

import os, logging, argparse
logging.disable(logging.WARNING)

# Monkey-patch pandapower
import pandapower as pp
_orig_runpp = pp.runpp
def _quiet_runpp(net, **kwargs):
    if 'numba' not in kwargs:
        kwargs['numba'] = False
    return _orig_runpp(net, **kwargs)
pp.runpp = _quiet_runpp

import yaml, json, time, copy
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.grid_setup import load_base_case
from src.grid_env import GridCascadeEnv
from src.ddpg import DDPGAgent
from src.cascade_sim import (simulate_cascade, _apply_scenario,
                             _sample_contingency, _trip_element,
                             _check_violations, _shed_isolated_loads,
                             _run_pf_with_shedding)
from src.baselines import (RuleBasedAgent, train_supervised_mlp,
                           opf_baseline, compute_metrics)

# Import evaluation functions from run_stage6
from run_stage6 import evaluate_with_cascade, evaluate_no_agent, _get_splits


def _paired_stats(na_preds, na_shed, other_preds, other_shed):
    """McNemar on cascade bits + Wilcoxon on per-scenario load shed."""
    from scipy.stats import wilcoxon, binomtest

    na_c = na_preds.astype(int)
    ot_c = other_preds.astype(int)
    b = int(np.sum((na_c == 1) & (ot_c == 0)))  # prevented
    c = int(np.sum((na_c == 0) & (ot_c == 1)))  # induced
    n_disc = b + c
    mcnemar_p = float(binomtest(b, n=n_disc, p=0.5).pvalue) if n_disc > 0 else 1.0
    try:
        w = wilcoxon(na_shed - other_shed, zero_method="wilcox", alternative="greater")
        wilcox_p = float(w.pvalue)
        wilcox_stat = float(w.statistic)
    except ValueError:
        wilcox_p, wilcox_stat = 1.0, 0.0
    return {
        "cascades_prevented": b,
        "cascades_induced": c,
        "mcnemar_n_discordant": n_disc,
        "mcnemar_p": mcnemar_p,
        "wilcoxon_stat": wilcox_stat,
        "wilcoxon_p_shed_reduction": wilcox_p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=150,
                        help="Number of test scenarios (full split is 150)")
    parser.add_argument("--skip-opf", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["rl"]["n_episodes"] = 100
    cfg["rl"]["n_seeds"] = 2
    cfg["rl"]["warmup_steps"] = 50
    cfg["rl"]["eval_interval"] = 20
    cfg["rl"]["episode_length"] = 8

    n_test_eval = args.n_test

    print("=" * 60)
    print("  Finishing Stage 6: Re-evaluating all trained models")
    print(f"  n_test={n_test_eval}  skip_opf={args.skip_opf}  skip_mlp={args.skip_mlp}")
    print("=" * 60)

    per_scenario = {}

    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"],
                           allow_pickle=True))

    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    y_all = cascade_raw["severity"]
    train_idx, val_idx, test_idx = _get_splits(cfg)
    y_test = (y_all[test_idx] > 0).astype(int)
    y_test_sub = y_test[:n_test_eval]

    all_results = {}

    # Create a temporary env to get obs/act dimensions
    tmp_env = GridCascadeEnv(net, weather, cfg,
                             scenario_indices=test_idx[:n_test_eval])
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    # ── 0. No-agent baseline ──────────────────────────────────────────
    print("\n[eval] No-agent baseline …")
    t0 = time.time()
    na_preds, na_sev, na_shed = evaluate_no_agent(
        net, weather, cfg, test_idx, n_test=n_test_eval)
    na_cascade_rate = na_preds.mean()
    print(f"  No-agent: cascade_rate={na_cascade_rate:.3f}, "
          f"mean_shed={na_shed.mean():.3f}, "
          f"sev_dist={[int((na_sev==s).sum()) for s in range(4)]}, "
          f"{time.time()-t0:.1f}s")
    per_scenario["no_agent_preds"] = na_preds
    per_scenario["no_agent_sev"] = na_sev
    per_scenario["no_agent_shed"] = na_shed
    per_scenario["test_idx"] = np.array(test_idx[:n_test_eval])

    # ── 1. Rule-based ─────────────────────────────────────────────────
    print("\n[eval] Rule-based baseline …")
    rb = RuleBasedAgent(cfg)
    rb_preds, rb_sev, rb_shed, rb_times = evaluate_with_cascade(
        lambda obs, net: rb.predict(obs, net),
        net, weather, cfg, test_idx, n_test=n_test_eval
    )
    prevented_rb = int(np.sum((na_preds == 1) & (rb_preds == 0)))
    all_results["Rule-based"] = {
        "cascade_rate": float(rb_preds.mean()),
        "mean_load_shed": float(rb_shed.mean()),
        "shed_reduction": float(na_shed.mean() - rb_shed.mean()),
        "cascades_prevented": prevented_rb,
        "severity_dist": {str(s): int((rb_sev==s).sum()) for s in range(4)},
        "mean_time_ms": float(rb_times.mean() * 1000),
    }
    print(f"  Rule-based: cascade_rate={rb_preds.mean():.3f}, "
          f"shed={rb_shed.mean():.3f}, prevented={prevented_rb}")
    all_results["Rule-based"].update(_paired_stats(na_preds, na_shed, rb_preds, rb_shed))
    per_scenario["rule_preds"] = rb_preds
    per_scenario["rule_shed"] = rb_shed

    # ── 2. Supervised MLP ──────────────────────────────────────────────
    if not args.skip_mlp:
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
    if not args.skip_opf:
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
        per_scenario["opf_preds"] = opf_preds

    # ── 4. DDPG-PER (load all available saved seed models) ────────────
    import glob
    seed_files = sorted(glob.glob("models/trained_weights/best_ddpg_ddpg_per_seed*.pt"))
    n_seeds = len(seed_files) if seed_files else 2
    seed_results = []
    print(f"\nFound {n_seeds} DDPG-PER seed model checkpoints to evaluate.")

    for seed in range(n_seeds):
        print(f"\n[eval] Loading DDPG-PER seed={seed} …")
        model_path = f"models/trained_weights/best_ddpg_ddpg_per_seed{seed}.pt"
        agent = DDPGAgent(obs_dim, act_dim, cfg)
        ckpt = torch.load(model_path, map_location="cpu")
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.actor.eval()

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
        }
        m.update(_paired_stats(na_preds, na_shed, ddpg_preds, ddpg_shed))
        seed_results.append(m)
        per_scenario[f"ddpg_per_seed{seed}_preds"] = ddpg_preds
        per_scenario[f"ddpg_per_seed{seed}_shed"] = ddpg_shed
        print(f"  DDPG-PER seed={seed}: cascade_rate={ddpg_preds.mean():.3f}, "
              f"shed={ddpg_shed.mean():.3f}, prevented={prevented}")

    # Aggregate DDPG seeds
    agg = {"method": "DDPG (ours)", "n_seeds": n_seeds}
    for key in ["cascade_rate", "mean_load_shed", "shed_reduction",
                "mean_time_ms", "cascades_prevented"]:
        vals = [s[key] for s in seed_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    all_results["DDPG (ours)"] = agg

    # ── 5. DDPG without PER (load saved model) ────────────────────────
    print("\n[eval] Loading DDPG no-PER seed=0 …")
    model_path = "models/trained_weights/best_ddpg_ddpg_noper_seed0.pt"
    agent_noper = DDPGAgent(obs_dim, act_dim, cfg)
    ckpt = torch.load(model_path, map_location="cpu")
    agent_noper.actor.load_state_dict(ckpt["actor"])
    agent_noper.critic.load_state_dict(ckpt["critic"])
    agent_noper.actor.eval()

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
          f"shed={noper_shed.mean():.3f}, prevented={prevented_noper}")
    all_results["DDPG (no PER)"].update(
        _paired_stats(na_preds, na_shed, noper_preds, noper_shed))
    per_scenario["ddpg_noper_preds"] = noper_preds
    per_scenario["ddpg_noper_shed"] = noper_shed

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

    results_name = f"all_results_n{n_test_eval}.json"
    with open(os.path.join(out_dir, results_name), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    np.savez(os.path.join(out_dir, f"per_scenario_n{n_test_eval}.npz"), **per_scenario)
    print(f"\nResults saved to {out_dir}/{results_name}")

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Method':<22} {'Casc%':>7} {'Shed':>7} {'Shed↓':>7} "
          f"{'Prev':>5} {'ms':>8}")
    print(f"{'-'*80}")

    for name, m in all_results.items():
        if name == "Supervised MLP":
            print(f"{name:<22}  [predictor] "
                  f"acc={m.get('accuracy',0):.3f} "
                  f"f1={m.get('f1',0):.3f} "
                  f"auc={m.get('auc',0):.3f}")
            continue

        if "cascade_rate_mean" in m:
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
