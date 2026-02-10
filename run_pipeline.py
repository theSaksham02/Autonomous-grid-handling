#!/usr/bin/env python3
"""
Master Pipeline — runs the entire project end-to-end.

Usage:
    python run_pipeline.py              # run everything
    python run_pipeline.py --stage 1    # run specific stage (1-7)
    python run_pipeline.py --from 3     # resume from stage 3

Stages:
    1  Grid setup (Day 1)
    2  Weather & renewables (Day 2)
    3  Cascade simulation (Day 3)
    4  Feature extraction & dataset (Day 4)
    5  Validation checks (Day 5)
    6  Train DDPG + baselines & full evaluation (Days 6-10)
    7  Figures, tables, ablation, sensitivity (Days 11-14)
"""

import argparse, sys, os, time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def stage_1(cfg):
    """Grid Setup"""
    from src.grid_setup import setup_base_case, save_base_case
    net = setup_base_case(cfg)
    save_base_case(net, cfg["paths"]["base_case"])
    return net


def stage_2(cfg):
    """Weather & Renewables"""
    from src.weather_renewables import generate_weather_scenarios, save_weather
    data = generate_weather_scenarios(cfg)
    save_weather(data, cfg["paths"]["weather_scenarios"])
    # Quick sanity
    print(f"  Night solar (h=0): max = {data['solar_power'][:, 0, :].max():.2f} MW")
    print(f"  Noon solar (h=12): mean = {data['solar_power'][:, 12, :].mean():.2f} MW")
    return data


def stage_3(cfg):
    """Cascade Simulation"""
    from src.grid_setup import load_base_case
    from src.cascade_sim import run_all_cascades, save_cascade_results
    import numpy as np

    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))
    results = run_all_cascades(net, weather, cfg, hour=12)
    save_cascade_results(results, cfg["paths"]["cascade_results"])
    return results


def stage_4(cfg):
    """Feature Extraction & Dataset"""
    import numpy as np
    from src.grid_setup import load_base_case
    from src.feature_extraction import (extract_features, stratified_split,
                                        normalize, save_datasets)

    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    severities = cascade_raw["severity"]
    cascade_results = [{"severity": int(s)} for s in severities]

    X, y = extract_features(net, weather, cascade_results, cfg)
    X_tr, y_tr, X_v, y_v, X_te, y_te = stratified_split(X, y, cfg)
    X_tr, X_v, X_te, mu, sigma = normalize(X_tr, X_v, X_te)
    save_datasets(X_tr, y_tr, X_v, y_v, X_te, y_te, mu, sigma, cfg)


def stage_5(cfg):
    """Validation Checks"""
    from src.validate_data import validate_dataset
    stats = validate_dataset(cfg)
    return stats


def stage_6(cfg):
    """Train & Evaluate Everything"""
    from src.evaluate import run_full_evaluation
    results = run_full_evaluation(cfg)
    return results


def stage_7(cfg):
    """Figures, Tables, Ablation, Sensitivity"""
    from src.figures import generate_all_figures
    from src.analysis import run_ablation_study, run_sensitivity_analysis, generate_latex_tables

    run_ablation_study(cfg)
    run_sensitivity_analysis(cfg)
    generate_all_figures(cfg)
    generate_latex_tables(cfg)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Grid Handling Pipeline")
    parser.add_argument("--stage", type=int, help="Run a specific stage (1-7)")
    parser.add_argument("--from-stage", type=int, default=1,
                        help="Resume from this stage (1-7)")
    parser.add_argument("--to-stage", type=int, default=7,
                        help="Stop after this stage (1-7)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    stages = {
        1: ("Grid Setup", stage_1),
        2: ("Weather & Renewables", stage_2),
        3: ("Cascade Simulation", stage_3),
        4: ("Feature Extraction", stage_4),
        5: ("Data Validation", stage_5),
        6: ("Train & Evaluate", stage_6),
        7: ("Figures & Analysis", stage_7),
    }

    if args.stage:
        stage_range = [args.stage]
    else:
        stage_range = range(args.from_stage, args.to_stage + 1)

    t_total = time.time()
    for s in stage_range:
        name, fn = stages[s]
        print(f"\n{'='*60}")
        print(f"  STAGE {s}: {name}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            fn(cfg)
        except Exception as e:
            print(f"\n  ❌ Stage {s} FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        elapsed = time.time() - t0
        print(f"\n  ✅ Stage {s} completed in {elapsed:.1f}s")

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — {total:.1f}s total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
