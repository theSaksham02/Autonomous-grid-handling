"""
OpenAI Gym Environment for Grid Cascading Failure Mitigation
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from typing import Dict
from grid_simulator import GridSimulator
from weather_injector import WeatherInjector


class GridEnv(gym.Env):
    """Custom Environment for Cascading Failure Prediction"""

    metadata = {'render.modes': ['human']}

    def __init__(self, config_path: str = "config.yaml"):
        super(GridEnv, self).__init__()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.grid_sim = GridSimulator(config_path)
        self.weather = WeatherInjector(config_path)

        # Action space: 7 continuous mitigation actions
        # [load_shed, battery, solar_curtail, wind_curtail, reactive_comp, freq_reg, delay]
        self.action_space = spaces.Box(
            low=np.array([0, -0.3, 0, 0, -0.2, -0.1, 0.5]),
            high=np.array([0.15, 0.3, 0.25, 0.20, 0.2, 0.1, 24]),
            dtype=np.float32
        )

        # State space: 247-dimensional
        # [voltages(118), angles(118), P_gen(54), Q_gen(54), P_load(91), weather(12), history(6), time(3)]
        state_dim = self.config['ddpg']['state_dim']
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Reward function weights
        self.lambda_cascade = self.config['reward']['lambda_cascade']
        self.lambda_cost = self.config['reward']['lambda_cost']
        self.lambda_overshed = self.config['reward']['lambda_overshed']
        self.costs = self.config['reward']['costs']

        # Episode state
        self.current_step = 0
        self.max_steps = self.config['ddpg']['training']['timesteps_per_episode']
        self.weather_forecast = None
        self.history_buffer = []

        print("ðŸŽ® Grid Environment initialized")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset grid
        self.grid_sim.reset()

        # Get new weather forecast
        self.weather_forecast = self.weather.get_renewable_forecast()

        # Apply random initial weather condition
        time_idx = np.random.randint(0, len(self.weather_forecast['timestamps']))
        solar = self.weather_forecast['solar_output'][time_idx]
        wind = self.weather_forecast['wind_output'][time_idx]
        self.grid_sim.update_renewable_output(np.array([solar]), np.array([wind]))

        # Run initial power flow
        self.grid_sim.run_power_flow()

        # Reset episode state
        self.current_step = 0
        self.history_buffer = []

        state = self._get_state(time_idx)
        return state, {}

    def step(self, action):
        """Execute one time step"""
        # Apply mitigation actions
        self._apply_mitigation(action)

        # Random contingency (20% probability)
        if np.random.rand() < 0.2:
            contingency_types = list(self.config['simulation']['contingency_types'].keys())
            cont_type = np.random.choice(contingency_types)
            self.grid_sim.apply_contingency(cont_type)

        # Update renewable output (next time step)
        time_idx = min(self.current_step + 1, len(self.weather_forecast['timestamps']) - 1)
        solar = self.weather_forecast['solar_output'][time_idx]
        wind = self.weather_forecast['wind_output'][time_idx]
        self.grid_sim.update_renewable_output(np.array([solar]), np.array([wind]))

        # Run power flow
        converged = self.grid_sim.run_power_flow()

        # Check violations
        violations = self.grid_sim.check_violations()

        # Compute reward
        reward = self._compute_reward(action, violations, converged)

        # Check termination
        self.current_step += 1
        terminated = (not converged) or (violations['total'] >= 5)
        truncated = self.current_step >= self.max_steps

        # Get next state
        next_state = self._get_state(time_idx)

        info = {
            'converged': converged,
            'violations': violations,
            'cascading_prob': self._estimate_cascade_prob(violations)
        }

        return next_state, reward, terminated, truncated, info

    def _get_state(self, time_idx: int) -> np.ndarray:
        """Extract current state vector"""
        # Grid state (415 dimensions)
        grid_state = self.grid_sim.get_grid_state()

        # Weather forecast (12 dimensions: 6 variables Ã— 2 horizons)
        weather_df = self.weather_forecast['weather_raw']
        current_weather = weather_df.iloc[time_idx]
        future_weather = weather_df.iloc[min(time_idx + 4, len(weather_df) - 1)]  # +12 hours

        weather_state = np.array([
            current_weather['temperature'],
            current_weather['wind_speed'],
            current_weather['wind_direction'],
            current_weather['cloud_cover'],
            current_weather['humidity'],
            current_weather['pressure'],
            future_weather['temperature'],
            future_weather['wind_speed'],
            future_weather['wind_direction'],
            future_weather['cloud_cover'],
            future_weather['humidity'],
            future_weather['pressure']
        ])

        # Historical trends (6 dimensions)
        if len(self.history_buffer) >= 6:
            recent_voltages = [h['voltage_min'] for h in self.history_buffer[-6:]]
            recent_loadings = [h['line_max_loading'] for h in self.history_buffer[-6:]]
            history_state = np.array([
                np.mean(recent_voltages),
                np.std(recent_voltages),
                np.mean(recent_loadings),
                np.std(recent_loadings),
                self.history_buffer[-1]['voltage_min'],
                self.history_buffer[-1]['line_max_loading']
            ])
        else:
            history_state = np.zeros(6)

        # Update history
        self.history_buffer.append({
            'voltage_min': self.grid_sim.net.res_bus.vm_pu.min(),
            'line_max_loading': self.grid_sim.net.res_line.loading_percent.max() if hasattr(self.grid_sim.net, 'res_line') else 0
        })

        # Temporal features (3 dimensions)
        timestamp = self.weather_forecast['timestamps'][time_idx]
        temporal_state = np.array([
            timestamp.hour / 24.0,
            timestamp.dayofweek / 7.0,
            (timestamp.month - 1) / 12.0
        ])

        # Combine all features
        state = np.concatenate([
            grid_state,      # 415
            weather_state,   # 12
            history_state,   # 6
            temporal_state   # 3
        ])

        # Pad to 247 dimensions (take first 247 for now)
        state = state[:247]

        # Normalize
        state = np.clip(state, -10, 10)  # Prevent extreme values

        return state.astype(np.float32)

    def _apply_mitigation(self, action: np.ndarray):
        """Apply mitigation actions to grid"""
        # action = [load_shed, battery, solar_curtail, wind_curtail, reactive_comp, freq_reg, delay]

        # Load shedding
        if action[0] > 0.01:
            shed_percentage = action[0]
            for idx in self.grid_sim.net.load.index[:int(len(self.grid_sim.net.load) * shed_percentage)]:
                self.grid_sim.net.load.at[idx, 'p_mw'] *= (1 - shed_percentage)

        # Battery dispatch
        if abs(action[1]) > 0.01:
            battery_power = action[1]  # MW
            for idx in self.grid_sim.net.storage.index:
                self.grid_sim.net.storage.at[idx, 'p_mw'] = battery_power

        # Solar curtailment
        if action[2] > 0.01:
            curtail_pct = action[2]
            solar_gens = self.grid_sim.net.sgen[self.grid_sim.net.sgen.type == "PV"]
            for idx in solar_gens.index:
                self.grid_sim.net.sgen.at[idx, 'p_mw'] *= (1 - curtail_pct)

        # Wind curtailment
        if action[3] > 0.01:
            curtail_pct = action[3]
            wind_gens = self.grid_sim.net.sgen[self.grid_sim.net.sgen.type == "WP"]
            for idx in wind_gens.index:
                self.grid_sim.net.sgen.at[idx, 'p_mw'] *= (1 - curtail_pct)

    def _compute_reward(self, action: np.ndarray, violations: Dict, converged: bool) -> float:
        """Compute multi-objective reward"""
        # 1. Cascading failure probability term
        if not converged:
            cascade_prob = 1.0
        else:
            cascade_prob = self._estimate_cascade_prob(violations)

        # 2. Economic cost term
        mitigation_cost = (
            self.costs['load_shed'] * action[0] +
            self.costs['battery_cycle'] * abs(action[1]) +
            self.costs['renewable_curtail'] * (action[2] + action[3]) +
            self.costs['frequency_reg'] * abs(action[5])
        )
        mitigation_cost /= 10000  # Normalize

        # 3. Over-shedding penalty
        if cascade_prob < 0.3 and action[0] > 0.05:
            overshed_penalty = self.config['reward']['overshed_penalty'] * (action[0] ** 2)
        else:
            overshed_penalty = 0

        # Combined reward
        reward = -(
            self.lambda_cascade * cascade_prob +
            self.lambda_cost * mitigation_cost +
            self.lambda_overshed * overshed_penalty
        )

        return float(reward)

    def _estimate_cascade_prob(self, violations: Dict) -> float:
        """Estimate cascading failure probability from violations"""
        if violations['total'] == 0:
            return 0.0

        # Sigmoid function
        x = (violations['voltage_critical'] * 2 + 
             violations['voltage_emergency'] * 5 +
             violations['line_overload'] * 3)

        prob = 1 / (1 + np.exp(-0.1 * (x - 10)))
        return float(prob)

    def render(self, mode='human'):
        """Render environment (optional)"""
        if mode == 'human':
            violations = self.grid_sim.check_violations()
            print(f"Step: {self.current_step}, Violations: {violations['total']}")


if __name__ == "__main__":
    # Test environment
    env = GridEnv()

    print("\n=== Testing Environment ===")
    state, _ = env.reset()
    print(f"State shape: {state.shape}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")

    # Test random action
    action = env.action_space.sample()
    print(f"\nRandom action: {action}")

    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.4f}")
    print(f"Info: {info}")