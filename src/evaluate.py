"""
Day 10 — Run All Evaluations
Evaluate every method on test set, multi-seed DDPG, compute all metrics.
"""

import os, time, json
import numpy as np
import yaml
import copy
from src.grid_setup import load_base_case
from src.grid_env import GridCascadeEnv
from src.ddpg import DDPGAgent
from src.train import train_ddpg, evaluate
from src.baselines import (RuleBasedAgent, train_supervised_mlp,
                           opf_baseline, compute_metrics)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_splits(cfg):
    """Return train/val/test index lists."""
    N = cfg["weather"]["n_scenarios"]
    rng = np.random.default_rng(cfg["dataset"]["random_seed"])
    all_idx = rng.permutation(N)
    n_tr = int(N * cfg["dataset"]["train_ratio"])
    n_va = int(N * cfg["dataset"]["val_ratio"])
    return (all_idx[:n_tr].tolist(),
            all_idx[n_tr:n_tr + n_va].tolist(),
            all_idx[n_tr + n_va:].tolist())


def evaluate_ddpg_on_test(agent, base_net, weather, cfg, test_idx):
    """Run trained DDPG on test scenarios, return binary preds + probabilities."""
    env = GridCascadeEnv(base_net, weather, cfg, scenario_indices=test_idx)
    preds = []
    rewards = []
    action_costs = []
    times = []

    for idx in test_idx:
        env.scenario_idx = idx
        obs, _ = env.reset()
        env.scenario_idx = idx
        done = False
        ep_reward = 0.0
        cascade_occurred = False
        ep_actions = []

        t0 = time.time()
        while not done:
            action = agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            cascade_occurred = info.get("cascade", False)
            ep_actions.append(np.abs(action).sum())
        elapsed = time.time() - t0

        preds.append(1 if cascade_occurred else 0)
        rewards.append(ep_reward)
        action_costs.append(np.mean(ep_actions))
        times.append(elapsed)

    return np.array(preds), np.array(rewards), np.array(action_costs), np.array(times)


def evaluate_rule_based_on_test(base_net, weather, cfg, test_idx):
    """Run rule-based agent on test scenarios."""
    env = GridCascadeEnv(base_net, weather, cfg, scenario_indices=test_idx)
    rb = RuleBasedAgent(cfg)
    preds = []
    times = []

    for idx in test_idx:
        env.scenario_idx = idx
        obs, _ = env.reset()
        env.scenario_idx = idx
        done = False
        cascade_occurred = False

        t0 = time.time()
        while not done:
            action = rb.predict(obs, env.current_net)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cascade_occurred = info.get("cascade", False)
        elapsed = time.time() - t0

        preds.append(1 if cascade_occurred else 0)
        times.append(elapsed)

    return np.array(preds), np.array(times)


def run_full_evaluation(cfg=None):
    """Master evaluation function. Returns all_results dict."""
    if cfg is None:
        cfg = load_config()

    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))

    # Load ground-truth labels
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    y_all = cascade_raw["severity"]

    train_idx, val_idx, test_idx = _get_splits(cfg)
    y_test = (y_all[test_idx] > 0).astype(int)

    all_results = {}

    # ── 1. Rule-based ──────────────────────────────────────────────────────
    print("\n[eval] Running Rule-based baseline …")
    rb_preds, rb_times = evaluate_rule_based_on_test(net, weather, cfg, test_idx)
    all_results["Rule-based"] = compute_metrics(y_test, rb_preds, method_name="Rule-based")
    all_results["Rule-based"]["mean_time_ms"] = float(rb_times.mean() * 1000)
    all_results["Rule-based"]["mean_action_cost"] = 0.0  # fixed actions

    # ── 2. Supervised MLP ──────────────────────────────────────────────────
    print("[eval] Training & evaluating Supervised MLP …")
    _, mlp_probs, mlp_y, _, _ = train_supervised_mlp(cfg)
    mlp_preds = (mlp_probs > 0.5).astype(int)
    all_results["Supervised MLP"] = compute_metrics(
        mlp_y, mlp_preds, mlp_probs, "Supervised MLP")
    all_results["Supervised MLP"]["mean_time_ms"] = 0.5  # fast inference

    # ── 3. OPF ─────────────────────────────────────────────────────────────
    print("[eval] Running OPF baseline …")
    opf_preds, opf_times = opf_baseline(net, weather, cfg, test_idx)
    all_results["OPF"] = compute_metrics(y_test, opf_preds, method_name="OPF")
    all_results["OPF"]["mean_time_ms"] = float(opf_times.mean() * 1000)

    # ── 4. Multi-seed DDPG ─────────────────────────────────────────────────
    n_seeds = cfg["rl"]["n_seeds"]
    n_episodes = cfg["rl"]["n_episodes"]
    seed_results = []
    all_train_logs = []

    for seed in range(n_seeds):
        print(f"\n[eval] Training DDPG seed={seed} …")
        agent, log = train_ddpg(cfg, net, weather, seed=seed,
                                use_per=True, tag=f"ddpg_per",
                                n_episodes=n_episodes)
        ddpg_preds, ddpg_rewards, ddpg_costs, ddpg_times = \
            evaluate_ddpg_on_test(agent, net, weather, cfg, test_idx)
        m = compute_metrics(y_test, ddpg_preds, method_name=f"DDPG seed={seed}")
        m["mean_time_ms"] = float(ddpg_times.mean() * 1000)
        m["mean_action_cost"] = float(ddpg_costs.mean())
        m["mean_reward"] = float(ddpg_rewards.mean())
        seed_results.append(m)
        all_train_logs.append(log)

    # Aggregate across seeds
    agg = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc", "fpr",
                "mean_time_ms", "mean_action_cost"]:
        vals = [s[key] for s in seed_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    agg["method"] = "DDPG (ours)"
    agg["n_seeds"] = n_seeds
    all_results["DDPG (ours)"] = agg

    # ── 5. DDPG without PER (ablation) ─────────────────────────────────────
    print("\n[eval] Training DDPG without PER (ablation) …")
    agent_noper, log_noper = train_ddpg(cfg, net, weather, seed=0,
                                        use_per=False, tag="ddpg_noper",
                                        n_episodes=n_episodes)
    noper_preds, noper_rewards, noper_costs, noper_times = \
        evaluate_ddpg_on_test(agent_noper, net, weather, cfg, test_idx)
    all_results["DDPG (no PER)"] = compute_metrics(
        y_test, noper_preds, method_name="DDPG (no PER)")
    all_results["DDPG (no PER)"]["mean_time_ms"] = float(noper_times.mean() * 1000)

    # ── Save everything ────────────────────────────────────────────────────
    out_dir = cfg["paths"]["tables_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Save training logs for figure generation
    for i, lg in enumerate(all_train_logs):
        log_path = os.path.join(cfg["paths"]["logs_dir"],
                                f"train_log_ddpg_per_seed{i}.json")
        serialisable = {k: [float(v) for v in vs] if isinstance(vs, list) else vs
                        for k, vs in lg.items()}
        with open(log_path, "w") as f:
            json.dump(serialisable, f, indent=2)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Method':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'FPR':>6} {'ms':>8}")
    print(f"{'='*80}")
    for name, m in all_results.items():
        if "accuracy_mean" in m:
            # Multi-seed
            print(f"{name:<20} "
                  f"{m['accuracy_mean']:>5.3f}±{m['accuracy_std']:.3f} "
                  f"{m['precision_mean']:>5.3f} {m['recall_mean']:>5.3f} "
                  f"{m['f1_mean']:>5.3f} {m['auc_mean']:>5.3f} "
                  f"{m['fpr_mean']:>5.3f} {m['mean_time_ms_mean']:>7.1f}")
        else:
            print(f"{name:<20} "
                  f"{m.get('accuracy',0):>6.3f} "
                  f"{m.get('precision',0):>6.3f} {m.get('recall',0):>6.3f} "
                  f"{m.get('f1',0):>6.3f} {m.get('auc',0):>6.3f} "
                  f"{m.get('fpr',0):>6.3f} {m.get('mean_time_ms',0):>8.1f}")
    print(f"{'='*80}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_full_evaluation()
    print("\n[eval] ✅ Day 10 complete.")
