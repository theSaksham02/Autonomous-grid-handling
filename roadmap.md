autonomous-grid-healing/
├── data/
│   ├── raw/                    # IEEE 118-bus data
│   ├── weather/                # Weather API responses
│   └── processed/              # Generated datasets
├── models/
│   ├── ddpg_agent.py          # DDPG implementation
│   ├── grid_env.py            # OpenAI Gym environment
│   └── trained_weights/        # Saved models
├── simulation/
│   ├── grid_simulator.py      # PyPSA/PandaPower interface
│   ├── failure_engine.py      # Cascading failure logic
│   └── weather_injector.py    # Weather data integration
├── baselines/
│   ├── opf_baseline.py        # Optimal Power Flow
│   ├── rule_based.py          # Heuristic rules
│   └── feedforward_nn.py      # Simple NN baseline
├── evaluation/
│   ├── metrics.py             # Accuracy, ROC-AUC, etc.
│   └── visualizations.py      # Charts and plots
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── config.yaml
└── README.md
