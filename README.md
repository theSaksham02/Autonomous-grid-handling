# Autonomous Grid Handling

Short, reproducible study: can a **small learning agent** take **one gentle preventive action** so a weather-stressed IEEE 118-bus grid cascades less often?

**Submit this:** EASRP 2026 anonymous pack in [`easrp2026/`](easrp2026/) (`main.tex` + `easrp2026.sty` + `references.bib` + `figures/`). Checklist: [`easrp2026/WHAT_TO_CHECK.txt`](easrp2026/WHAT_TO_CHECK.txt).

**UG 0-to-tech guide (for you):** [`docs/ug_guide/README.md`](docs/ug_guide/README.md) — cascade analogy, folder map, code walkthrough, libraries, viva Q&A, diagrams.

Older IEEE draft (kept for reference): [`paper.tex`](paper.tex)

## Elevator pitch (say this out loud)

The grid is like a road network. If one line trips, power spills onto the next line and you can get a chain-reaction blackout (a **cascade**).

We do **not** try to drive the grid in real time during the crash. We allow **one small move before** the accident: nudge a battery, a bit of reactive power, slightly cut wind/solar, or slightly cut demand. Then we replay the **same accident** for four methods: do nothing, a simple if-then rule, a slow optimiser (OPF), and a DDPG agent.

**Result on 150 test days:** doing nothing cascades 34% of the time; the rule and DDPG both land near 30%. DDPG is about **1.9× faster** than OPF. The improvement is **modest** (not statistically significant at this size). That honesty is the point of the paper.

## Key results (`n = 150`)

| Method | Cascade rate | Load shed | Days prevented | Time (ms) |
|--------|:------------:|:---------:|:--------------:|:---------:|
| Do nothing | 0.340 | 0.141 | — | — |
| If-then rule | 0.300 | 0.121 | 11 | 293 |
| OPF | 0.320 | — | 10 | 568 |
| **DDPG (2-seed mean)** | **0.303** | **0.127** | **8.5** | **299** |
| DDPG (no PER) | 0.313 | 0.136 | 6 | 309 |

Better DDPG seed: cascade 0.293, shed 0.119, 11 days prevented.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --start 1 --end 5
python run_stage6.py
python finish_stage6.py --n-test 150
```

Knobs: `config.yaml`. Accidents for scenario `i` use seed `i + 1000` so every method sees the same trips.

Optional GPU (not required; power flow is CPU):

```bash
colab new -s grid-eval --gpu T4
colab install -s grid-eval -r requirements.txt
```

`colab sessions` is empty until you run `colab new`.

## Paper figures and tables

Figures in `results/figures/` (built by `python scripts/make_paper_figures.py`):

| File | What it shows |
|------|----------------|
| `fig_pipeline.png` | One test-day scoring loop |
| `fig_summary_three.png` | Cascade rate, load shed, speed |
| `fig_severity_none.png` | Do-nothing severity counts |
| `fig_prevented_induced.png` | Days saved vs days made worse |
| `fig_shed_box.png` | Per-day load-shed boxplots |
| `fig_shed_scatter.png` | DDPG seed 1 vs do-nothing, paired |
| `fig_training.png` | Training reward curves |
| `fig_main_n150.png` | Earlier two-panel summary |

Tables: `results/tables/latex_tables.tex`, numbers in `all_results_n150.json` and `extra_stats_n150.json`.

## Layout

```
config.yaml          # all experiment knobs
run_pipeline.py      # stages 1–7
run_stage6.py        # train + eval
finish_stage6.py     # reload checkpoints, full n=150 table
paper.tex            # short IEEE draft
src/                 # grid, weather, cascade, DDPG, baselines
results/tables/      # all_results_n150.json
results/figures/     # fig_main_n150.png
```
