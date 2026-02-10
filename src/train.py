"""
Day 8 — Training Loop
Train DDPG with episodic logging, periodic validation, and checkpointing.
"""

import os, time, json
import numpy as np
import yaml
from tqdm import trange

from src.grid_env import GridCascadeEnv
from src.ddpg import DDPGAgent


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(agent, env, scenario_indices, n_eval=None):
    """
    Run greedy policy on the given scenarios.
    Returns dict with mean reward, max loading stats.
    """
    if n_eval is None:
        n_eval = len(scenario_indices)
    indices = scenario_indices[:n_eval]

    total_reward = 0.0
    max_loadings = []
    action_costs = []

    for idx in indices:
        env.scenario_idx = idx
        obs, _ = env.reset()
        env.scenario_idx = idx  # force this scenario
        done = False
        ep_reward = 0.0
        ep_actions = []
        ep_max_loading = 0.0

        while not done:
            action = agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_actions.append(np.abs(action).sum())
            ep_max_loading = max(ep_max_loading, info.get("max_loading", 0))

        total_reward += ep_reward
        max_loadings.append(ep_max_loading)
        action_costs.append(np.mean(ep_actions) if ep_actions else 0)

    n = len(indices)
    return {
        "mean_reward": total_reward / max(n, 1),
        "mean_max_loading": float(np.mean(max_loadings)),
        "mean_action_cost": float(np.mean(action_costs)),
        "n_eval": n,
    }


def train_ddpg(cfg, base_net, weather, seed=42, use_per=False,
               tag="ddpg", n_episodes=None):
    """
    Full training run.
    Returns: agent, log (dict with episode-level records).
    """
    if n_episodes is None:
        n_episodes = cfg["rl"]["n_episodes"]

    # --- build scenario index splits ----------------------------------------
    N = cfg["weather"]["n_scenarios"]
    rng = np.random.default_rng(cfg["dataset"]["random_seed"])
    all_idx = rng.permutation(N)
    n_tr = int(N * cfg["dataset"]["train_ratio"])
    n_va = int(N * cfg["dataset"]["val_ratio"])
    train_idx = all_idx[:n_tr].tolist()
    val_idx = all_idx[n_tr:n_tr + n_va].tolist()

    # --- env & agent --------------------------------------------------------
    env = GridCascadeEnv(base_net, weather, cfg,
                         scenario_indices=train_idx)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = DDPGAgent(obs_dim, act_dim, cfg, use_per=use_per, seed=seed)

    eval_env = GridCascadeEnv(base_net, weather, cfg,
                              scenario_indices=val_idx)

    # --- logging -------------------------------------------------------------
    log = {
        "episode_rewards": [],
        "cascade_prevented": [],
        "critic_losses": [],
        "actor_losses": [],
        "val_accuracy": [],
        "val_reward": [],
        "sigma_history": [],
    }

    best_val_acc = -1.0
    eval_interval = cfg["rl"]["eval_interval"]

    print(f"\n{'='*60}")
    print(f"  Training {tag}  |  {n_episodes} episodes  |  seed={seed}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  PER={use_per}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for ep in trange(n_episodes, desc=f"Train {tag}"):
        obs, _ = env.reset(seed=seed + ep)
        agent.noise.reset()
        done = False
        ep_reward = 0.0
        ep_cascade = False
        c_losses, a_losses = [], []

        while not done:
            action = agent.select_action(obs, add_noise=True)
            obs2, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_cascade = info.get("cascade", False)

            agent.buffer.store(obs, action, reward, obs2, done)
            obs = obs2
            ep_reward += reward
            agent.total_steps += 1

            # Learn
            cl, al = agent.update()
            if cl is not None:
                c_losses.append(cl)
                a_losses.append(al)

        agent.noise.anneal()

        # Log
        log["episode_rewards"].append(ep_reward)
        log["cascade_prevented"].append(int(not ep_cascade))
        log["critic_losses"].append(np.mean(c_losses) if c_losses else 0)
        log["actor_losses"].append(np.mean(a_losses) if a_losses else 0)
        log["sigma_history"].append(agent.noise.sigma)

        # Periodic validation
        if (ep + 1) % eval_interval == 0:
            val_metrics = evaluate(agent, eval_env, val_idx, n_eval=min(50, len(val_idx)))
            log["val_accuracy"].append(val_metrics["mean_reward"])
            log["val_reward"].append(val_metrics["mean_reward"])

            if val_metrics["mean_reward"] > best_val_acc:
                best_val_acc = val_metrics["mean_reward"]
                model_path = cfg["paths"]["best_model"].replace(
                    ".pt", f"_{tag}_seed{seed}.pt")
                agent.save(model_path)

            if (ep + 1) % (eval_interval * 2) == 0:
                elapsed = time.time() - t0
                tqdm_msg = (f"ep={ep+1}  R={ep_reward:.1f}  "
                            f"val_R={val_metrics['mean_reward']:.1f}  "
                            f"max_ll={val_metrics['mean_max_loading']:.2f}  "
                            f"σ_noise={agent.noise.sigma:.4f}  "
                            f"{elapsed:.0f}s")
                print(f"  [{tag}] {tqdm_msg}")

    total_time = time.time() - t0
    print(f"\n[train] {tag} finished in {total_time:.1f}s.  Best val acc: {best_val_acc:.3f}")

    # Save training log
    log_path = os.path.join(cfg["paths"]["logs_dir"], f"train_log_{tag}_seed{seed}.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Convert numpy types for JSON
    serialisable_log = {k: [float(v) for v in vs] if isinstance(vs, list) else vs
                        for k, vs in log.items()}
    with open(log_path, "w") as f:
        json.dump(serialisable_log, f, indent=2)

    return agent, log


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.grid_setup import load_base_case

    cfg = load_config()
    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))

    agent, log = train_ddpg(cfg, net, weather, seed=42, use_per=False,
                            tag="ddpg", n_episodes=500)
    print("[train] ✅ Day 8 complete.")
