"""
Cascading Failure Simulation Engine --> Google collab
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from grid_simulator import GridSimulator
from weather_injector import WeatherInjector
import yaml
from tqdm import tqdm
import pickle


class CascadingFailureEngine:
    """Simulates multi-stage cascading failures in power grids"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.grid_sim = GridSimulator(config_path)
        self.weather = WeatherInjector(config_path)

    def simulate_single_contingency(
        self, 
        contingency_type: str,
        weather_data: Dict,
        time_idx: int = 0
    ) -> Dict:
        """Simulate single N-1 contingency with weather conditions"""

        # Reset grid
        self.grid_sim.reset()

        # Apply weather conditions
        if time_idx < len(weather_data['solar_output']):
            solar = weather_data['solar_output'][time_idx]
            wind = weather_data['wind_output'][time_idx]
            self.grid_sim.update_renewable_output(
                np.array([solar]), 
                np.array([wind])
            )

        # Initial power flow
        if not self.grid_sim.run_power_flow():
            return self._create_failure_record(True, 0, "initial_divergence")

        initial_violations = self.grid_sim.check_violations()

        # Apply contingency
        self.grid_sim.apply_contingency(contingency_type)

        # Simulate cascading propagation
        cascade_stages = []
        time_horizon = self.config['simulation']['time_horizon_seconds']
        time_step = self.config['simulation']['time_step_seconds']

        for t in range(0, time_horizon, int(time_step)):
            # Run power flow
            converged = self.grid_sim.run_power_flow()

            if not converged:
                cascade_stages.append({
                    'time': t,
                    'event': 'power_flow_divergence',
                    'cascading': True
                })
                break

            # Check for secondary failures
            violations = self.grid_sim.check_violations()

            if violations['total'] > 0:
                cascade_stages.append({
                    'time': t,
                    'violations': violations,
                    'voltage_min': self.grid_sim.net.res_bus.vm_pu.min(),
                    'line_max_loading': self.grid_sim.net.res_line.loading_percent.max() if hasattr(self.grid_sim.net, 'res_line') else 0
                })

                # Check cascading criteria: >= 3 secondary failures
                if len(cascade_stages) >= 3:
                    return self._create_failure_record(
                        True, 
                        len(cascade_stages), 
                        contingency_type,
                        cascade_stages
                    )

        # No cascading failure
        is_cascading = len(cascade_stages) >= 3
        return self._create_failure_record(
            is_cascading,
            len(cascade_stages),
            contingency_type,
            cascade_stages
        )

    def _create_failure_record(
        self, 
        is_cascading: bool, 
        num_stages: int,
        contingency_type: str,
        cascade_stages: List = None
    ) -> Dict:
        """Create standardized failure record"""

        state = self.grid_sim.get_grid_state()
        violations = self.grid_sim.check_violations()

        return {
            'is_cascading': is_cascading,
            'num_stages': num_stages,
            'contingency_type': contingency_type,
            'grid_state': state,
            'violations': violations,
            'cascade_stages': cascade_stages or [],
            'affected_buses': violations['voltage_critical'] + violations['voltage_emergency']
        }

    def generate_dataset(self, n_scenarios: int = 1000) -> pd.DataFrame:
        """Generate complete cascading failure dataset"""

        print(f"\nðŸ”„ Generating {n_scenarios} cascading failure scenarios...")

        # Get weather forecast
        weather_data = self.weather.get_renewable_forecast()

        # Contingency type distribution
        cfg = self.config['simulation']['contingency_types']
        contingency_types = []
        for cont_type, prob in cfg.items():
            contingency_types.extend([cont_type] * int(n_scenarios * prob))

        # Pad to exact size
        while len(contingency_types) < n_scenarios:
            contingency_types.append(np.random.choice(list(cfg.keys())))
        contingency_types = contingency_types[:n_scenarios]
        np.random.shuffle(contingency_types)

        # Run simulations
        dataset = []
        for i in tqdm(range(n_scenarios)):
            # Random weather condition
            time_idx = np.random.randint(0, len(weather_data['timestamps']))

            result = self.simulate_single_contingency(
                contingency_types[i],
                weather_data,
                time_idx
            )

            # Add weather features
            result['weather_temp'] = weather_data['weather_raw'].iloc[time_idx]['temperature']
            result['weather_wind'] = weather_data['weather_raw'].iloc[time_idx]['wind_speed']
            result['weather_cloud'] = weather_data['weather_raw'].iloc[time_idx]['cloud_cover']
            result['solar_output'] = weather_data['solar_output'][time_idx]
            result['wind_output'] = weather_data['wind_output'][time_idx]
            result['time_of_day'] = weather_data['timestamps'][time_idx].hour

            dataset.append(result)

        df = pd.DataFrame(dataset)

        # Statistics
        n_cascading = df['is_cascading'].sum()
        print(f"\nâœ… Dataset generated:")
        print(f"   Total scenarios: {len(df)}")
        print(f"   Cascading failures: {n_cascading} ({n_cascading/len(df)*100:.1f}%)")
        print(f"   Non-cascading: {len(df) - n_cascading} ({(len(df)-n_cascading)/len(df)*100:.1f}%)")
        print(f"   Avg cascade depth: {df[df['is_cascading']]['num_stages'].mean():.2f}")
        print(f"   Max cascade depth: {df['num_stages'].max()}")

        return df

    def save_dataset(self, df: pd.DataFrame, filename: str = "cascading_dataset.pkl"):
        """Save dataset to disk"""
        import os
        os.makedirs('data/processed', exist_ok=True)

        filepath = f"data/processed/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)

        print(f"\nðŸ’¾ Dataset saved to {filepath}")

    def load_dataset(self, filename: str = "cascading_dataset.pkl") -> pd.DataFrame:
        """Load dataset from disk"""
        filepath = f"data/processed/{filename}"

        with open(filepath, 'rb') as f:
            df = pickle.load(f)

        print(f"ðŸ“‚ Loaded dataset: {len(df)} scenarios")
        return df


if __name__ == "__main__":
    # Test cascading failure engine
    engine = CascadingFailureEngine()

    # Generate small test dataset
    dataset = engine.generate_dataset(n_scenarios=100)

    # Save it
    engine.save_dataset(dataset, "test_dataset.pkl")

    print("\n=== Sample Results ===")
    print(dataset[['is_cascading', 'num_stages', 'contingency_type', 'affected_buses']].head(10))