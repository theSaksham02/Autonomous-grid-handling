"""
Day 2 — Weather & Renewable Pipeline
Generate synthetic weather scenarios and convert to renewable power injections
using the wind (Eq. 3) and solar (Eq. 4) models from the paper.
"""

import os
import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Wind power (Eq. 3) ──────────────────────────────────────────────────────
def wind_power(v_wind: np.ndarray, v_ci: float, v_r: float,
               v_co: float, p_rated: float) -> np.ndarray:
    """
    Piece-wise turbine curve (Eq. 3 in paper).
    Returns MW output array with same shape as v_wind.
    """
    p = np.zeros_like(v_wind)
    ramp = (v_wind >= v_ci) & (v_wind < v_r)
    p[ramp] = p_rated * ((v_wind[ramp] - v_ci) / (v_r - v_ci)) ** 3
    rated = (v_wind >= v_r) & (v_wind <= v_co)
    p[rated] = p_rated
    return p


# ── Solar power (Eq. 4) ─────────────────────────────────────────────────────
def solar_power(ghi: np.ndarray, p_rated: float,
                ghi_stc: float = 1000.0) -> np.ndarray:
    """
    Linear model: P_solar = P_rated * (GHI / GHI_STC).
    Clipped to [0, P_rated].
    """
    return np.clip(p_rated * ghi / ghi_stc, 0.0, p_rated)


# ── Scenario generation ─────────────────────────────────────────────────────
def generate_weather_scenarios(cfg: dict) -> dict:
    """
    Generate N synthetic 24-h weather scenarios.

    Returns dict with keys:
      wind_speed   (N, 24, n_wind_buses)     m/s
      ghi          (N, 24, n_solar_buses)     W/m²
      temperature  (N, 24)                    °C
      load_factor  (N, 24)                    fraction of peak
      wind_power   (N, 24, n_wind_buses)      MW
      solar_power  (N, 24, n_solar_buses)     MW
    """
    rng = np.random.default_rng(cfg["weather"]["seed"])
    N = cfg["weather"]["n_scenarios"]
    H = cfg["weather"]["n_hours"]

    n_wind = len(cfg["renewables"]["wind_buses"])
    n_solar = len(cfg["renewables"]["solar_buses"])

    hours = np.arange(H, dtype=float)

    # ── wind speed: Weibull draws with temporal correlation ──────────────
    shape = cfg["renewables"]["weibull_shape"]
    scale = cfg["renewables"]["weibull_scale"]
    ws = rng.weibull(shape, size=(N, H, n_wind)) * scale
    # add mild autocorrelation via exponential smoothing
    for t in range(1, H):
        ws[:, t, :] = 0.7 * ws[:, t, :] + 0.3 * ws[:, t - 1, :]

    # ── GHI: sinusoidal daily pattern + cloud noise ─────────────────────
    sunrise, sunset = 6.0, 18.0
    solar_envelope = np.maximum(
        np.sin(np.pi * (hours - sunrise) / (sunset - sunrise)), 0.0
    )  # shape (24,)
    ghi_base = 1000.0 * solar_envelope[None, :, None]  # (1,24,1)
    cloud = rng.uniform(0.5, 1.0, size=(N, H, n_solar))
    ghi = ghi_base * cloud  # (N, 24, n_solar)

    # ── temperature: seasonal sinusoid + daily swing + noise ────────────
    day_of_year = rng.integers(1, 366, size=N)
    seasonal = 15.0 + 15.0 * np.sin(
        2 * np.pi * (day_of_year[:, None] - 80) / 365
    )  # (N, 1)
    daily_swing = 5.0 * np.sin(
        2 * np.pi * (hours[None, :] - 14) / 24
    )  # (1, 24)
    temp = seasonal + daily_swing + rng.normal(0, 2, size=(N, H))

    # ── load factor: double-peak daily profile + noise ──────────────────
    morning = np.exp(-0.5 * ((hours - 8) / 2) ** 2)
    evening = np.exp(-0.5 * ((hours - 19) / 2) ** 2)
    base_profile = 0.5 + 0.3 * morning + 0.4 * evening  # (24,)
    load_noise = rng.normal(0, 0.03, size=(N, H))
    load_factor = np.clip(base_profile[None, :] + load_noise, 0.3, 1.2)

    # ── convert to MW ───────────────────────────────────────────────────
    wp = wind_power(ws,
                    cfg["renewables"]["cut_in"],
                    cfg["renewables"]["rated_speed"],
                    cfg["renewables"]["cut_out"],
                    cfg["renewables"]["wind_capacity_mw"])

    sp = solar_power(ghi, cfg["renewables"]["solar_capacity_mw"])

    return dict(
        wind_speed=ws.astype(np.float32),
        ghi=ghi.astype(np.float32),
        temperature=temp.astype(np.float32),
        load_factor=load_factor.astype(np.float32),
        wind_power=wp.astype(np.float32),
        solar_power=sp.astype(np.float32),
        wind_buses=np.array(cfg["renewables"]["wind_buses"]),
        solar_buses=np.array(cfg["renewables"]["solar_buses"]),
    )


def save_weather(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"[weather] Saved {len(data['wind_speed'])} scenarios → {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    data = generate_weather_scenarios(cfg)

    # quick sanity print
    print(f"  wind_speed  : {data['wind_speed'].shape}")
    print(f"  ghi         : {data['ghi'].shape}")
    print(f"  wind_power  : {data['wind_power'].shape}")
    print(f"  solar_power : {data['solar_power'].shape}")
    print(f"  load_factor : {data['load_factor'].shape}")
    print(f"  Night solar (h=0): max = {data['solar_power'][:, 0, :].max():.2f} MW")
    print(f"  Noon solar (h=12): mean = {data['solar_power'][:, 12, :].mean():.2f} MW")

    save_weather(data, cfg["paths"]["weather_scenarios"])
    print("[weather] ✅ Day 2 complete.")
