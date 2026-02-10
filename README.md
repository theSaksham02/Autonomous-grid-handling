# Autonomous Grid Handling

**Deep Reinforcement Learning for Cascading Failure Prevention in Power Systems**

This repository contains the complete codebase for the paper *"Autonomous Grid Handling: Deep Reinforcement Learning for Cascading Failure Prevention in Power Systems"*. The framework uses DDPG (Deep Deterministic Policy Gradient) to learn preventive control actions on the IEEE 118-bus test system, reducing cascade rates and load shedding under weather-driven stress conditions.

---

## Key Results

| Method | Cascade Rate | Load Shed | Shed Reduction | Latency (ms) |
|--------|:-----------:|:---------:|:--------------:|:------------:|
| No Agent (baseline) | 0.280 | 0.118 | — | — |
| Rule-based | 0.240 | 0.101 | +0.017 | 157 |
| OPF | 0.240 | 0.000 | +0.118 | 330 |
| **DDPG (ours, PER)** | **0.240** | **0.090** | **+0.028** | **164** |
| DDPG (no PER) | 0.240 | 0.100 | +0.019 | 194 |

DDPG matches OPF cascade prevention at **2× faster inference**.

---

## Architecture

```
                    ┌─────────────────────┐
                    │  Weather Scenarios   │
                    │  (1000 × 24h)        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  IEEE 118-bus Grid   │
                    │  (180% loading)      │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │        DDPG Agent               │
              │  obs(489) → Actor → action(4)   │
              │  ┌─────────────────────────┐    │
              │  │ Storage    ±15 MW       │    │
              │  │ Reactive   ±10 MVAr     │    │
              │  │ Curtailment 0–20%       │    │
              │  │ Demand Resp 0–15%       │    │
              │  └─────────────────────────┘    │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  N-k Cascade Sim    │
                    │  (evaluation only)  │
                    └─────────────────────┘
```

---

## Project Structure

```
├── config.yaml              # Master configuration
├── run_pipeline.py          # Full 7-stage pipeline runner
├── run_stage6.py            # Stage 6: Training & cascade-based evaluation
├── run_stage7.py            # Stage 7: Ablation, sensitivity, figures, tables
├── finish_stage6.py         # Re-evaluation utility (loads saved models)
├── paper.tex                # LaTeX paper (IEEE format)
│
├── src/
│   ├── grid_setup.py        # pandapower case118 setup
│   ├── weather_gen.py       # Stochastic weather scenario generation
│   ├── cascade_sim.py       # Physics-based N-k cascade simulation
│   ├── feature_eng.py       # 489-dim feature extraction
│   ├── grid_env.py          # Gymnasium RL environment
│   ├── ddpg.py              # DDPG agent (Actor-Critic, PER, OU noise)
│   ├── train.py             # Training loop
│   ├── baselines.py         # Rule-based, MLP, OPF baselines
│   ├── evaluate.py          # Legacy evaluation (classification-based)
│   ├── analysis.py          # Legacy analysis module
│   ├── figures.py           # Legacy figure generation
│   └── validation.py        # Data sanity checks
│
├── data/
│   ├── raw/                 # Base case, weather, cascade results
│   └── processed/           # Train/val/test splits, normalization stats
│
├── models/trained_weights/  # Saved DDPG checkpoints
├── results/
│   ├── tables/              # JSON results + LaTeX tables
│   └── figures/             # Publication-quality PNG figures
└── logs/                    # Training logs (JSON)
```

---

## Quick Start

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline (Stages 1–7)

```bash
python run_pipeline.py --start 1 --end 7
```

### Individual Stages

```bash
# Stage 6: Train DDPG + baselines, cascade evaluation
python run_stage6.py

# Stage 7: Ablation, sensitivity, figures, LaTeX tables  
python run_stage7.py
```

---

## Methodology

### Training
- **Proxy reward**: Voltage stability + line loading penalty + generation adequacy + action cost
- **8-hour episodes** with base state restoration between hours
- **100 episodes**, 2 seeds, PER with proportional prioritization

### Evaluation (Single-Step Paradigm)
1. Apply weather scenario to fresh grid at peak hour (h=12)
2. Agent observes grid state → takes **one** preventive action
3. Full N-k cascade simulation on the modified grid
4. Per-scenario deterministic RNG (seed = index + 1000) ensures fair comparison

---

## Citation

```bibtex
@article{mishra2026autonomous,
  title={Autonomous Grid Handling: Deep Reinforcement Learning for 
         Cascading Failure Prevention in Power Systems},
  author={Mishra, Saksham},
  year={2026}
}
```

---

## License

This project is for academic research purposes.
