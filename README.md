# Autonomous Grid Healing - DDPG Agent for Cascading Failure Mitigation

## 📌 Quick Links

- **Local Development**: See [GITHUB_WORKFLOW.md](GITHUB_WORKFLOW.md)
- **Colab GPU Training**: See [COLAB_SETUP.md](COLAB_SETUP.md)
- **Project Status**: See [PROJECT_STATUS.txt](PROJECT_STATUS.txt)
- **Paper Roadmap**: See [roadmap.md](roadmap.md)

---

## 🎯 Project Overview

This project implements a **Deep Deterministic Policy Gradient (DDPG)** reinforcement learning agent to prevent cascading failures in power grids with renewable energy integration.

### Key Components

1. **IEEE 118-Bus Power Grid Simulator** (`grid_simulator.py`)
   - Realistic grid topology using PandaPower
   - 53 generators, 99 loads, 173 transmission lines
   - Integration of 35 solar + 18 wind + 12 battery units

2. **DDPG Reinforcement Learning Agent** (`ddpg_Agent.py`)
   - State space: 462 dimensions (voltages, angles, generation, loads, weather)
   - Action space: 7 continuous mitigation actions
   - Actor-Critic network architecture

3. **Grid Environment** (`grid_env.py`)
   - OpenAI Gymnasium environment
   - 96 time steps per episode (24 hours at 15-min intervals)
   - Weather-driven renewable variability

4. **Baseline Comparisons** (`baseline_methods.py`)
   - Optimal Power Flow (OPF)
   - Rule-based heuristics
   - Simple feedforward neural network

---

## 🚀 Quick Start

### Option 1: Test Locally (2-3 minutes)

```bash
cd /Users/sakshammishra/Autonomous-grid-handling
python quick_test.py
```

**Output**: Verifies all components work (5 episodes)

### Option 2: Train Locally (30-45 minutes)

```bash
python train_with_monitoring.py
```

**Output**: 100-episode training with real-time monitoring
- Saves: `models/trained_weights/ddpg_final.pth`
- Plots: `results/training_history.png`

### Option 3: Full Training on Colab (4-8 hours with GPU)

1. Push code to GitHub:
   ```bash
   git add .
   git commit -m "DDPG training setup"
   git push origin main
   ```

2. Open Colab notebook: [train_ddpg_colab.ipynb](train_ddpg_colab.ipynb)

3. Run all cells (notebook handles setup, training, and results)

---

## 📊 Workflow: Local Development + Colab Training

```
┌─────────────────────────┐
│   Local (VS Code)       │
│  ─────────────────────  │
│  • Write code           │
│  • Test (100 episodes)  │
│  • Git push to GitHub   │
└──────────────┬──────────┘
               │
               ▼
┌─────────────────────────┐
│  GitHub (Repository)    │
│  ─────────────────────  │
│  • Central code hub      │
│  • Version control      │
│  • CI/CD ready          │
└──────────────┬──────────┘
               │
               ▼
┌─────────────────────────┐
│   Colab (GPU Training)  │
│  ─────────────────────  │
│  • Clone from GitHub    │
│  • Install dependencies │
│  • Train 5000 episodes  │
│  • Save to Drive/GitHub │
└──────────────┬──────────┘
               │
               ▼
┌─────────────────────────┐
│   Local Analysis        │
│  ─────────────────────  │
│  • Download results     │
│  • Evaluate vs baselines│
│  • Generate figures     │
└─────────────────────────┘
```

---

## 📁 Project Structure

```
Autonomous-grid-handling/
├── 📄 README.md                    ← You are here
├── 📄 COLAB_SETUP.md              ← Colab instructions
├── 📄 GITHUB_WORKFLOW.md           ← Git workflow
├── 📄 PROJECT_STATUS.txt           ← Status summary
├── 📄 roadmap.md                   ← Technical details
│
├── ⚙️ Configuration
│   ├── config.yaml                 ← All hyperparameters
│   └── requirements.txt            ← Dependencies
│
├── 🔴 Core Training
│   ├── train_ddpg_colab.ipynb      ← RUN ON COLAB (5000 ep)
│   ├── train_with_monitoring.py    ← RUN LOCAL (100 ep test)
│   ├── train_ddpg.py               ← Original script
│   └── quick_test.py               ← Verification (5 ep)
│
├── 🧠 Agent & Environment
│   ├── ddpg_Agent.py               ← DDPG implementation
│   ├── grid_env.py                 ← Gym environment
│   └── baseline_methods.py         ← OPF, rules, FNN baselines
│
├── ⚡ Grid Simulation
│   ├── grid_simulator.py           ← IEEE 118-bus system
│   ├── weather_injector.py         ← Renewable forecasts
│   └── failure_engine.py           ← Cascade failures
│
├── 📊 Evaluation & Analysis
│   ├── evaluation.py               ← Compare DDPG vs baselines
│   └── results/                    ← Training outputs
│       ├── training_history.pkl    ← Metrics
│       └── training_history.png    ← Plots
│
├── 🤖 Models (Git-ignored, save to Drive)
│   └── models/trained_weights/
│       ├── ddpg_final.pth          ← Final trained agent
│       └── ddpg_ep*.pth            ← Checkpoints
│
└── 📦 Virtual Environment (Git-ignored)
    └── .venv/                      ← Python environment
```

---

## 🔧 Setup Instructions

### Local Environment (First Time)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python quick_test.py
```

### Dependencies

- **RL**: PyTorch 2.0+, Gymnasium, Stable-Baselines3
- **Power Systems**: PandaPower, PyPSA
- **Data**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly

---

## 📈 Training & Results

### Local Test Run (100 episodes, ~45 minutes)

```bash
python train_with_monitoring.py
```

**Outputs:**
```
models/trained_weights/
├── ddpg_ep20.pth
├── ddpg_ep40.pth
├── ...
└── ddpg_final.pth

results/
├── training_history.pkl    (metrics: rewards, losses)
└── training_history.png    (plots)
```

### Full Training on Colab (5000 episodes, ~6-8 hours on GPU)

1. Open notebook: `train_ddpg_colab.ipynb`
2. Select GPU runtime (T4 or A100)
3. Run all cells
4. Results saved to Google Drive

**Expected Results:**
- Average episode reward: 20-50+
- Final cascade prevention: 60-80%
- Outperforms OPF/rule-based baselines by 15-25%

### Evaluation

After training:

```bash
python evaluation.py
```

**Compares:**
- DDPG Agent (trained)
- OPF Baseline (theoretical optimal)
- Rule-Based Baseline (expert heuristics)
- Feedforward NN Baseline (simple neural network)

**Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Cascade prevention rate
- Cost analysis

---

## 🛠️ Development Workflow

### Day 1: Local Development

```bash
# Test your changes
python quick_test.py                    # 2 min
python train_with_monitoring.py         # 45 min (100 episodes)

# Commit and push
git add modified_files.py
git commit -m "Improved reward function"
git push origin main
```

### Day 2: Colab Training

```
1. Open train_ddpg_colab.ipynb
2. Notebook auto-pulls latest from GitHub
3. Runs 5000-episode training on GPU
4. Saves models to Google Drive
```

### Day 3: Analysis & Evaluation

```bash
# Download trained model from Drive (or git pull)
python evaluation.py        # Compare against baselines
# Generate plots and analysis
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
python quick_test.py
```

### GPU Not Available in Colab
- Runtime → Change Runtime Type → Select GPU

### Training Too Slow
- Check GPU: `!nvidia-smi` (Colab)
- Reduce batch size in `config.yaml`
- Use Colab Pro for A100 GPU

### Out of Memory
- Reduce `timesteps_per_episode` from 96 to 48
- Reduce `batch_size` from 128 to 64

---

## 📚 References

### Papers & Theory
- DDPG: [Continuous Control with Deep RL](https://arxiv.org/abs/1509.02971)
- Power Grid: IEEE 118-bus benchmark system
- Cascading Failures: Complex network theory

### Code References
- Environment: OpenAI Gymnasium
- Grid Simulator: PandaPower
- Deep Learning: PyTorch

---

## 🤝 Contributing

To extend this project:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes locally and test
3. Push: `git push origin feature/your-feature`
4. Create Pull Request on GitHub

---

## 📝 License

MIT License - See repository for details

---

## ❓ Quick FAQ

**Q: Can I run this on my laptop?**
A: Yes! Use `python train_with_monitoring.py` with 100 episodes (~45 min). Full training (5000 ep) recommended on Colab GPU.

**Q: How long does full training take?**
A: GPU (Colab T4): 6-8 hours | CPU (local): 24-48 hours

**Q: Should I commit model files to GitHub?**
A: No! Save to Google Drive instead (files are too large). Use Git only for code, config, and documentation.

**Q: Can I use TensorFlow instead of PyTorch?**
A: The current implementation uses PyTorch. TensorFlow support would require adapter code.

**Q: How do I get better results?**
A: 1) Increase episodes (more training), 2) Tune hyperparameters in `config.yaml`, 3) Improve reward function in `grid_env.py`

---

## 📞 Support

- **Local Issues**: Check `PROJECT_STATUS.txt`
- **Colab Issues**: See `COLAB_SETUP.md`
- **Git Issues**: See `GITHUB_WORKFLOW.md`
- **Code Issues**: Check docstrings in source files

---

**Last Updated**: January 24, 2026
**Status**: ✅ Ready for Training
