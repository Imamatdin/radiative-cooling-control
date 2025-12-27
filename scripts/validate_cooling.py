"""Validate radiative cooling model against SkyCool benchmark."""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.physics.atmosphere import calculate_sky_conditions
from src.physics.radiation import selective_emitter_cooling_realistic


def validate_against_skycool():
    """Compare model output against SkyCool EPC-18-006 data."""
    print("Running Radiative Cooling Validation...")
    
    ambient_temps = np.linspace(10, 35, 20)
    scenarios = [
        {"rh": 0.20, "cloud": 0.0, "label": "Dry (20% RH)", "color": "blue"},
        {"rh": 0.60, "cloud": 0.0, "label": "Humid (60% RH)", "color": "green"},
    ]
    
    plt.figure(figsize=(10, 6))
    plt.axhspan(70, 100, color='gray', alpha=0.2, label='SkyCool Range')
    
    for sc in scenarios:
        cooling_powers = []
        for t_air_c in ambient_temps:
            t_air_k = t_air_c + 273.15
            sky_cond = calculate_sky_conditions(t_air_c, sc['rh'], sc['cloud'])
            res = selective_emitter_cooling_realistic(
                T_surface_K=t_air_k, 
                T_sky_K=sky_cond['T_sky_K'], 
                T_air_K=t_air_k,
                relative_humidity=sc['rh'], 
                cloud_fraction=sc['cloud']
            )
            cooling_powers.append(res['q_total'])
        plt.plot(ambient_temps, cooling_powers, label=sc['label'], color=sc['color'], linewidth=2)

    plt.title("Model Validation: Net Cooling Power vs. Ambient Temperature")
    plt.xlabel("Ambient Temperature (°C)")
    plt.ylabel("Net Cooling Power (W/m²)")
    plt.ylim(0, 140)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    os.makedirs('figures', exist_ok=True)
    output_path = "figures/validation_cooling_skycool.png"
    plt.savefig(output_path)
    print(f"✅ Validation plot saved to {output_path}")


if __name__ == "__main__":
    validate_against_skycool()
