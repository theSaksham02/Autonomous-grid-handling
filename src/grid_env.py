"""
Day 6 — Gymnasium-compatible RL Environment for Grid Cascade Prevention

Observation: state vector (voltages, line loadings, gen outputs, weather, trends, time)
Action:      Box([-1,-1,0,0], [1,1,1,1]) — 4 continuous dims
             [storage charge/discharge, reactive compensation,
              renewable curtailment, demand reduction]
Reward:      Voltage stability + line loading reduction + action cost.
             The agent learns to keep the grid in a healthy state so that
             when N-k contingencies occur, cascades are less severe.
Episode:     T timesteps (one per hour)
Done:        episode_length reached.
"""

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandapower as pp
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class GridCascadeEnv(gym.Env):
    """Grid cascade prevention environment.

    The agent takes preventive actions each hour.  After each action we run
    power flow and reward the agent for:
      • Keeping bus voltages within [0.95, 1.05] pu
      • Reducing maximum line loading (lines close to capacity cascade easily)
      • Minimising action cost

    Evaluation is done *outside* the env by running a full N-k cascade
    simulation on the agent-modified grid state (see evaluate_with_cascade).
    """

    metadata = {"render_modes": []}

    def __init__(self, base_net, weather_data, cfg,
                 scenario_indices=None, deterministic=False):
        super().__init__()

        self.base_net = base_net
        self.weather = weather_data
        self.cfg = cfg
        self.deterministic = deterministic
        self.scenario_indices = scenario_indices
        self.rng = np.random.default_rng(cfg["dataset"]["random_seed"])

        # Dimensions
        n_bus = cfg["features"]["n_voltage"]
        n_line = cfg["features"]["n_line_loading"]
        n_gen = cfg["features"]["n_gen_output"]
        n_weather = cfg["features"]["n_weather"]
        n_trend = cfg["features"]["n_trend"]
        n_time = cfg["features"]["n_time"]
        self.obs_dim = n_bus + n_line + n_gen + n_weather + n_trend + n_time
        self.act_dim = cfg["rl"]["action_dim"]

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Normalisation stats
        norm = np.load(cfg["paths"]["norm_stats"])
        self.mu = norm["mu"]
        self.sigma = norm["sigma"]

        # Reward weights
        self.alpha = cfg["reward"]["alpha"]
        self.beta = cfg["reward"]["beta"]
        self.lam = cfg["reward"]["lambda_cost"]
        self.cascade_pen = cfg["reward"]["cascade_penalty"]
        self.stab_bonus = cfg["reward"]["stability_bonus"]

        # Episode state
        self.current_net = None
        self.hour = 0
        self.scenario_idx = 0
        self.episode_reward = 0.0

        # Grid topology helpers
        self.n_bus = n_bus
        self.n_line = n_line
        self.n_gen = n_gen

        # Storage / DR buses
        self.storage_buses = cfg["renewables"]["wind_buses"][:4]
        self.dr_buses = list(range(0, 20))

    # ── helpers ──────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        net = self.current_net
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        offset = 0

        # voltages
        if net.converged:
            vm = net.res_bus["vm_pu"].values[:self.n_bus]
        else:
            vm = np.ones(self.n_bus, dtype=np.float32)
        obs[offset:offset + self.n_bus] = vm
        offset += self.n_bus

        # line loadings
        if net.converged:
            ll = net.res_line["loading_percent"].values[:self.n_line] / 100.0
        else:
            ll = np.zeros(self.n_line, dtype=np.float32)
        obs[offset:offset + self.n_line] = ll
        offset += self.n_line

        # gen output
        if net.converged:
            pg = net.res_gen["p_mw"].values
            if len(pg) < self.n_gen:
                pg = np.concatenate([pg, np.zeros(self.n_gen - len(pg))])
            pg = pg[:self.n_gen]
        else:
            pg = np.zeros(self.n_gen, dtype=np.float32)
        obs[offset:offset + self.n_gen] = pg
        offset += self.n_gen

        # weather features
        n_wf = self.cfg["features"]["n_weather"]
        wfeat = np.zeros(n_wf, dtype=np.float32)
        h = self.hour
        si = self.scenario_idx
        for k, hh in enumerate(range(h, min(h + 4, 24))):
            base = k * 6
            wfeat[base + 0] = self.weather["wind_speed"][si, hh].mean()
            wfeat[base + 1] = self.weather["ghi"][si, hh].mean()
            wfeat[base + 2] = self.weather["temperature"][si, hh]
            wfeat[base + 3] = self.weather["load_factor"][si, hh]
            wfeat[base + 4] = self.weather["wind_power"][si, hh].mean()
            wfeat[base + 5] = self.weather["solar_power"][si, hh].mean()
        obs[offset:offset + n_wf] = wfeat
        offset += n_wf

        # voltage trend (proxy: same as current voltage)
        n_trend = self.cfg["features"]["n_trend"]
        obs[offset:offset + n_trend] = vm[:n_trend]
        offset += n_trend

        # time encoding
        obs[offset] = np.sin(2 * np.pi * h / 24.0)
        obs[offset + 1] = np.cos(2 * np.pi * h / 24.0)

        # normalise
        obs = (obs - self.mu) / self.sigma
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        return obs.astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        """
        action[0]: storage charge/discharge  ∈ [-1, 1]
        action[1]: reactive compensation     ∈ [-1, 1]
        action[2]: renewable curtailment     ∈ [0, 1]
        action[3]: demand reduction          ∈ [0, 1]
        """
        net = self.current_net
        storage_mw = action[0] * 15.0  # ±15 MW (conservative)
        reactive_mvar = action[1] * 10.0  # ±10 MVAr
        curtail_frac = float(action[2]) * 0.20  # max 20% curtailment
        dr_frac = float(action[3]) * 0.15  # max 15% demand reduction

        # Storage: add/remove MW at storage buses
        for bus in self.storage_buses:
            mask = net.sgen["bus"] == bus
            if mask.any():
                net.sgen.loc[mask, "p_mw"] += storage_mw / len(self.storage_buses)

        # Reactive: adjust shunt compensation
        if len(net.shunt) > 0:
            net.shunt["q_mvar"] += reactive_mvar / max(len(net.shunt), 1)

        # Curtailment: scale down renewable sgens (capped)
        if curtail_frac > 0.005:
            ren_mask = net.sgen["name"].str.contains("wind|solar", na=False)
            net.sgen.loc[ren_mask, "p_mw"] *= (1.0 - curtail_frac)

        # Demand reduction (gentle, spread across DR buses)
        if dr_frac > 0.005:
            for bus in self.dr_buses:
                mask = net.load["bus"] == bus
                net.load.loc[mask, "p_mw"] *= (1.0 - dr_frac)
                net.load.loc[mask, "q_mvar"] *= (1.0 - dr_frac)

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Reward function (Eq. 7-8).

        Four components:
          1. Voltage stability: penalty for deviations from [0.95, 1.05]
          2. Line loading: penalty for lines approaching capacity
             (high-loaded lines are the ones that trip in cascades)
          3. Action cost: penalty for large control actions
          4. Generation adequacy: bonus for maintaining generation margin
        """
        net = self.current_net

        # Voltage stability term
        if net.converged:
            vm = net.res_bus["vm_pu"].values
            vmin, vmax = self.cfg["grid"]["voltage_normal"]
            v_dev = np.maximum(0, vmin - vm) + np.maximum(0, vm - vmax)
            r_voltage = -self.alpha * np.sum(v_dev)
        else:
            r_voltage = -self.alpha * 10.0

        # Line loading term — penalise lines above 60% (cascade-prone region)
        if net.converged:
            ll = net.res_line["loading_percent"].values / 100.0
            in_svc = net.line["in_service"].values
            ll_active = ll[in_svc]
            # Quadratic penalty above 60% threshold
            excess = np.maximum(0, ll_active - 0.60)
            r_loading = -self.beta * np.sum(excess ** 2)
            # Bonus for keeping max loading low
            max_loading = ll_active.max() if len(ll_active) > 0 else 0
            r_loading += self.stab_bonus * max(0, 1.0 - max_loading)
        else:
            r_loading = self.cascade_pen * 0.5

        # Action cost — stronger penalty to encourage minimal intervention
        action_cost = self.lam * 1.5 * np.sum(np.abs(action))

        # Generation adequacy: penalize excessive curtailment/demand reduction
        # Agent should keep power balance healthy, not strip the grid
        r_adequacy = 0.0
        if net.converged:
            try:
                total_gen = net.res_gen["p_mw"].sum() + net.res_ext_grid["p_mw"].sum()
                total_load = net.res_load["p_mw"].sum()
                # Reward for maintaining good gen/load ratio (≈1.0 is ideal)
                if total_load > 0:
                    ratio = total_gen / total_load
                    # Penalize if ratio deviates too far from 1.0
                    r_adequacy = -0.5 * max(0, abs(ratio - 1.0) - 0.1)
            except Exception:
                pass

        return float(r_voltage + r_loading - action_cost + r_adequacy)

    # ── Gym interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Pick scenario
        if self.scenario_indices is not None:
            self.scenario_idx = int(self.rng.choice(self.scenario_indices))
        else:
            self.scenario_idx = int(self.rng.integers(0, self.cfg["weather"]["n_scenarios"]))

        self.hour = 0
        self.episode_reward = 0.0

        # Deep-copy base network ONCE per episode
        self.current_net = copy.deepcopy(self.base_net)

        # Store original load/sgen values for fast restoration each hour
        self._orig_load_p = self.current_net.load["p_mw"].values.copy()
        self._orig_load_q = self.current_net.load["q_mvar"].values.copy()
        self._orig_sgen_p = self.current_net.sgen["p_mw"].values.copy() if len(self.current_net.sgen) > 0 else np.array([])
        self._orig_line_in_svc = self.current_net.line["in_service"].values.copy()

        from src.cascade_sim import _apply_scenario
        _apply_scenario(self.current_net, self.weather, self.scenario_idx,
                        self.hour, self.cfg)

        try:
            pp.runpp(self.current_net, algorithm="nr", max_iteration=30,
                     tolerance_mva=1e-4, numba=False)
        except Exception:
            pass

        obs = self._get_obs()
        return obs, {}

    def _restore_base_state(self):
        """Restore load/sgen/line to original values (fast, no deepcopy)."""
        self.current_net.load["p_mw"] = self._orig_load_p.copy()
        self.current_net.load["q_mvar"] = self._orig_load_q.copy()
        if len(self._orig_sgen_p) > 0:
            self.current_net.sgen["p_mw"] = self._orig_sgen_p.copy()
        self.current_net.line["in_service"] = self._orig_line_in_svc.copy()

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply agent's action
        self._apply_action(action)

        # Re-run power flow
        try:
            pp.runpp(self.current_net, algorithm="nr", max_iteration=30,
                     tolerance_mva=1e-4, numba=False)
        except Exception:
            self.current_net.converged = False

        reward = self._compute_reward(action)
        self.episode_reward += reward

        self.hour += 1
        terminated = False  # episodes always run to completion
        truncated = self.hour >= self.cfg["rl"]["episode_length"]

        # Advance to next hour if not done
        if not terminated and not truncated:
            self._restore_base_state()
            from src.cascade_sim import _apply_scenario
            _apply_scenario(self.current_net, self.weather,
                            self.scenario_idx, self.hour, self.cfg)
            try:
                pp.runpp(self.current_net, algorithm="nr", max_iteration=30,
                         tolerance_mva=1e-4, numba=False)
            except Exception:
                pass

        obs = self._get_obs()

        # Compute max loading for info (used by training logger)
        if self.current_net.converged:
            ll = self.current_net.res_line["loading_percent"].values
            in_svc = self.current_net.line["in_service"].values
            max_loading = ll[in_svc].max() / 100.0 if in_svc.any() else 0
        else:
            max_loading = 2.0  # diverged → very bad

        info = {
            "episode_reward": self.episode_reward,
            "max_loading": float(max_loading),
        }
        return obs, reward, terminated, truncated, info


# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle
    from src.grid_setup import load_base_case

    cfg = load_config()
    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))

    env = GridCascadeEnv(net, weather, cfg)

    n_ok = 0
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        done = False
        steps = 0
        while not done:
            act = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            steps += 1
        n_ok += 1

    print(f"[env] {n_ok}/100 episodes completed without crash. ✅ Day 6 complete.")
