# Adaptive Radiative Cooling Technology (ARCT)

Reinforcement Learning for Datacenter Radiative Cooling Control

## Overview

This project implements a DDPG (Deep Deterministic Policy Gradient) agent to optimize radiative cooling systems for datacenters. The agent learns to control the split between radiative panels and cooling towers to minimize electricity consumption and water usage.

## Key Innovation

**Problem**: Current radiative cooling systems use fixed-flow control and shut down during marginal conditions (cloudy, humid weather), wasting potential cooling hours.

**Solution**: Adaptive flow control using reinforcement learning that:
- Continues operation during marginal conditions by reducing flow rate
- Learns to anticipate weather changes using forecasts
- Optimizes the trade-off between electricity and water savings

## Results

| Controller | Electricity Savings | Water Savings |
|------------|-------------------|---------------|
| Tower Only | -4.6% | 0% |
| Fixed 50% | 1.4% | 25.1% |
| Fixed 100% | 1.5% | 26.5% |
| **DDPG (Ours)** | **5.8%** | **65.2%** |

## Project Structure

```
radiative-cooling-control/
├── config/                 # Configuration files
├── data/weather/          # TMY weather data
├── src/
│   ├── physics/           # Physical models (atmosphere, radiation, chiller)
│   ├── environment/       # RL environments (datacenter cooling)
│   └── agents/            # DDPG agents (standard and forecast-aware)
├── scripts/               # Training and evaluation scripts
├── results/               # Trained models and results
└── figures/               # Generated figures
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download weather data
python scripts/download_weather.py

# Train the agent
python scripts/train_correct.py

# Compare controllers
python scripts/compare_controllers.py

# Generate figures
python scripts/generate_figures.py
```

## Physics Models

- **Atmosphere**: Berdahl-Martin sky temperature model
- **Radiation**: Two-band selective emitter model with atmospheric transmittance
- **Chiller**: Carnot-based COP model with part-load correction
- **Cooling Tower**: Wet-bulb approach model with water consumption

## References

- Raman et al. (2014) Nature 515, 540-544
- Aili et al. (2024) Nature Energy
- ASHRAE Handbook: Fundamentals

## Author

Iko Imamatdin - Gap Year Research Project
