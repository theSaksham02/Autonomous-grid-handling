"""
Grid Simulation using PandaPower (IEEE 118-bus system)
"""
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import yaml


class GridSimulator:
    """IEEE 118-bus system simulator with renewable integration"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load IEEE 118-bus system
        self.net = pn.case118()
        self.base_net = self.net.deepcopy()  # Store original state

        # Add renewable generators
        self._add_renewable_generators()
        self._add_battery_storage()

        print(f"âœ… Loaded IEEE {self.config['grid']['buses']}-bus system")
        print(f"   Buses: {len(self.net.bus)}")
        print(f"   Lines: {len(self.net.line)}")
        print(f"   Generators: {len(self.net.gen)}")
        print(f"   Solar PV: {self.config['renewables']['solar']['count']}")
        print(f"   Wind: {self.config['renewables']['wind']['count']}")
        print(f"   Batteries: {self.config['renewables']['battery']['count']}")

    def _add_renewable_generators(self):
        """Add solar and wind generators to random buses"""
        np.random.seed(42)

        # Solar PV
        solar_cfg = self.config['renewables']['solar']
        solar_buses = np.random.choice(
            self.net.bus.index, 
            size=solar_cfg['count'], 
            replace=False
        )
        solar_capacity_each = solar_cfg['total_capacity_mw'] / solar_cfg['count']

        for bus_idx in solar_buses:
            pp.create_sgen(
                self.net,
                bus=bus_idx,
                p_mw=solar_capacity_each * 0.5,  # Initial output
                q_mvar=0,
                name=f"Solar_PV_{bus_idx}",
                type="PV"
            )

        # Wind
        wind_cfg = self.config['renewables']['wind']
        wind_buses = np.random.choice(
            [b for b in self.net.bus.index if b not in solar_buses],
            size=wind_cfg['count'],
            replace=False
        )
        wind_capacity_each = wind_cfg['total_capacity_mw'] / wind_cfg['count']

        for bus_idx in wind_buses:
            pp.create_sgen(
                self.net,
                bus=bus_idx,
                p_mw=wind_capacity_each * 0.5,  # Initial output
                q_mvar=0,
                name=f"Wind_{bus_idx}",
                type="WP"
            )

    def _add_battery_storage(self):
        """Add battery storage systems"""
        np.random.seed(43)
        battery_cfg = self.config['renewables']['battery']

        battery_buses = np.random.choice(
            self.net.bus.index,
            size=battery_cfg['count'],
            replace=False
        )

        for bus_idx in battery_buses:
            pp.create_storage(
                self.net,
                bus=bus_idx,
                p_mw=0,  # Initial: neither charging nor discharging
                max_e_mwh=battery_cfg['total_capacity_mwh'] / battery_cfg['count'],
                soc_percent=50.0,
                name=f"Battery_{bus_idx}"
            )

    def update_renewable_output(self, solar_output: np.ndarray, wind_output: np.ndarray):
        """Update renewable generator outputs based on weather forecast"""
        solar_cfg = self.config['renewables']['solar']
        wind_cfg = self.config['renewables']['wind']

        # Update solar
        solar_gens = self.net.sgen[self.net.sgen.type == "PV"]
        for idx, (gen_idx, gen) in enumerate(solar_gens.iterrows()):
            capacity = solar_cfg['total_capacity_mw'] / solar_cfg['count']
            self.net.sgen.at[gen_idx, 'p_mw'] = capacity * solar_output[min(idx, len(solar_output)-1)]

        # Update wind
        wind_gens = self.net.sgen[self.net.sgen.type == "WP"]
        for idx, (gen_idx, gen) in enumerate(wind_gens.iterrows()):
            capacity = wind_cfg['total_capacity_mw'] / wind_cfg['count']
            self.net.sgen.at[gen_idx, 'p_mw'] = capacity * wind_output[min(idx, len(wind_output)-1)]

    def run_power_flow(self) -> bool:
        """Run AC power flow analysis"""
        try:
            pp.runpp(self.net, algorithm='nr', calculate_voltage_angles=True)
            return self.net.converged
        except Exception as e:
            print(f"Power flow failed: {e}")
            return False

    def get_grid_state(self) -> np.ndarray:
        """Extract current grid state as feature vector"""
        state = []

        # Bus voltages (118)
        state.extend(self.net.res_bus.vm_pu.values)

        # Voltage angles (118)
        state.extend(self.net.res_bus.va_degree.values)

        # Generator active power (54)
        gen_p = np.zeros(self.config['grid']['generators'])
        gen_p[:len(self.net.res_gen)] = self.net.res_gen.p_mw.values
        state.extend(gen_p)

        # Generator reactive power (54)
        gen_q = np.zeros(self.config['grid']['generators'])
        gen_q[:len(self.net.res_gen)] = self.net.res_gen.q_mvar.values
        state.extend(gen_q)

        # Load active power (99 - IEEE 118-bus system has 99 loads)
        load_p = np.zeros(99)
        load_p[:len(self.net.res_load)] = self.net.res_load.p_mw.values
        state.extend(load_p)

        return np.array(state)

    def check_violations(self) -> Dict[str, int]:
        """Check for voltage and line loading violations"""
        cfg = self.config['simulation']

        violations = {
            'voltage_critical': 0,
            'voltage_emergency': 0,
            'line_overload': 0,
            'total': 0
        }

        # Voltage violations
        v_critical = (self.net.res_bus.vm_pu < cfg['voltage_critical']).sum()
        v_emergency = (self.net.res_bus.vm_pu < cfg['voltage_emergency']).sum()
        violations['voltage_critical'] = int(v_critical)
        violations['voltage_emergency'] = int(v_emergency)

        # Line loading violations
        if hasattr(self.net, 'res_line'):
            l_overload = (self.net.res_line.loading_percent > 100).sum()
            violations['line_overload'] = int(l_overload)

        violations['total'] = violations['voltage_critical'] + violations['line_overload']

        return violations

    def apply_contingency(self, contingency_type: str, element_idx: Optional[int] = None) -> bool:
        """Apply N-1 contingency (line trip, generator trip, etc.)"""
        if element_idx is None:
            element_idx = np.random.randint(0, 100)

        try:
            if contingency_type == 'line_outage' and len(self.net.line) > element_idx:
                self.net.line.at[element_idx, 'in_service'] = False
                return True

            elif contingency_type == 'generator_trip' and len(self.net.gen) > element_idx % len(self.net.gen):
                gen_idx = element_idx % len(self.net.gen)
                self.net.gen.at[gen_idx, 'in_service'] = False
                return True

            elif contingency_type == 'load_spike':
                # Increase random loads by 15-40%
                spike_factor = 1.15 + np.random.rand() * 0.25
                affected_loads = np.random.choice(self.net.load.index, size=10, replace=False)
                for load_idx in affected_loads:
                    self.net.load.at[load_idx, 'p_mw'] *= spike_factor
                return True

            elif contingency_type == 'transformer_failure' and len(self.net.trafo) > 0:
                trafo_idx = element_idx % len(self.net.trafo)
                self.net.trafo.at[trafo_idx, 'in_service'] = False
                return True

        except Exception as e:
            print(f"Contingency application failed: {e}")

        return False

    def reset(self):
        """Reset grid to initial state"""
        self.net = self.base_net.deepcopy()


if __name__ == "__main__":
    # Test grid simulator
    sim = GridSimulator()

    print("\n=== Running test power flow ===")
    converged = sim.run_power_flow()
    print(f"Power flow converged: {converged}")

    if converged:
        violations = sim.check_violations()
        print(f"Violations: {violations}")

    state = sim.get_grid_state()
    print(f"State vector dimension: {len(state)}")
    print(f"Voltage range: {state[:118].min():.3f} - {state[:118].max():.3f} p.u.")