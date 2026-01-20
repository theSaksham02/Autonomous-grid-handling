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



Required Files: 
	1.	requirements.txt - All dependencies
	2.	config.yaml - Complete configuration
	3.	simulation/weather_injector.py - Real-time weather data integration
	4.	simulation/grid_simulator.py - IEEE 118-bus system
	5.	simulation/failure_engine.py - Cascading failure generation
	6.	models/ddpg_agent.py - DDPG implementation
	7.	models/grid_env.py - OpenAI Gym environment
    8. Training Script
    9. Baseline Methods
    10. Evaluations Script

## Setup
# Create project directory
mkdir autonomous-grid-healing && cd autonomous-grid-healing

# Create folder structure
mkdir -p data/{raw,weather,processed} models/{trained_weights} simulation baselines evaluation results logs

# Copy all 10 files into respective folders

# Install dependencies
pip install -r requirements.txt

# Get OpenWeatherMap API key (free): https://openweathermap.org/api
# Add to .env file: OPENWEATHER_API_KEY=your_key_here

---- 

## Generated Dateset 
python simulation/failure_engine.py
# This creates cascading_dataset.pkl with 1000 scenarios

----

## Train DDPG
python train_ddpg.py
# Trains for 5000 episodes (takes ~12-24 hours on GPU, 2-3 days on CPU)
# Start with 100 episodes for testing

---- 

## Evaluate 
python evaluate.py
# Compares DDPG vs OPF vs Rules vs FNN

---- 

## Quick Start
# 1. Test weather injector
python simulation/weather_injector.py

# 2. Test grid simulator
python simulation/grid_simulator.py

# 3. Generate 10 test scenarios
python -c "from failure_engine import CascadingFailureEngine; engine = CascadingFailureEngine(); engine.generate_dataset(10)"

# 4. Test DDPG agent
python models/ddpg_agent.py

# 5. Test environment
python models/grid_env.py
