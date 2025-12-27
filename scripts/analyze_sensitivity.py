"""
Sensitivity Analysis for Radiative Cooling System.

Analyzes how system performance varies with:
- Humidity
- Cloud cover
- Panel area
- Tank size
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.atmosphere import calculate_sky_conditions
from physics.radiation import selective_emitter_cooling_realistic
from physics.chiller import RadiativeHeatRejection


def analyze_humidity_sensitivity():
    """Analyze cooling capacity vs relative humidity."""
    print("\n--- Humidity Sensitivity ---")
    
    humidity_range = np.linspace(0.1, 0.9, 17)
    T_air_C = 25.0
    T_air_K = T_air_C + 273.15
    
    results = []
    for rh in humidity_range:
        sky = calculate_sky_conditions(T_air_C, rh, cloud_fraction=0.0)
        cooling = selective_emitter_cooling_realistic(
            T_surface_K=T_air_K,
            T_sky_K=sky['T_sky_K'],
            T_air_K=T_air_K,
            relative_humidity=rh,
            cloud_fraction=0.0,
        )
        results.append({
            'rh': rh * 100,
            'T_sky_C': sky['T_sky_C'],
            'q_net': cooling['q_total'],
        })
    
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(df['rh'], df['T_sky_C'], 'b-', linewidth=2)
    ax1.set_xlabel('Relative Humidity (%)')
    ax1.set_ylabel('Sky Temperature (°C)')
    ax1.set_title('Sky Temperature vs Humidity')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['rh'], df['q_net'], 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(df['rh'], df['q_net'], 0, where=df['q_net'] > 0, alpha=0.3, color='blue', label='Net cooling')
    ax2.fill_between(df['rh'], df['q_net'], 0, where=df['q_net'] < 0, alpha=0.3, color='red', label='Net heating')
    ax2.set_xlabel('Relative Humidity (%)')
    ax2.set_ylabel('Net Cooling Power (W/m²)')
    ax2.set_title('Net Cooling vs Humidity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/sensitivity_humidity.png', dpi=150)
    plt.close()
    
    return df


def analyze_cloud_sensitivity():
    """Analyze cooling capacity vs cloud cover."""
    print("\n--- Cloud Cover Sensitivity ---")
    
    cloud_range = np.linspace(0.0, 1.0, 21)
    T_air_C = 25.0
    T_air_K = T_air_C + 273.15
    rh = 0.40
    
    results = []
    for cloud in cloud_range:
        sky = calculate_sky_conditions(T_air_C, rh, cloud_fraction=cloud)
        cooling = selective_emitter_cooling_realistic(
            T_surface_K=T_air_K,
            T_sky_K=sky['T_sky_K'],
            T_air_K=T_air_K,
            relative_humidity=rh,
            cloud_fraction=cloud,
        )
        results.append({
            'cloud': cloud * 100,
            'T_sky_C': sky['T_sky_C'],
            'q_net': cooling['q_total'],
        })
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['cloud'], df['q_net'], 'b-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(df['cloud'], df['q_net'], 0, where=df['q_net'] > 0, alpha=0.3, color='blue')
    ax.set_xlabel('Cloud Cover (%)')
    ax.set_ylabel('Net Cooling Power (W/m²)')
    ax.set_title('Radiative Cooling vs Cloud Cover (RH=40%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/sensitivity_clouds.png', dpi=150)
    plt.close()
    
    return df


def analyze_panel_area_scaling():
    """Analyze system economics vs panel area."""
    print("\n--- Panel Area Scaling ---")
    
    panel_areas = np.array([500, 1000, 1500, 2000, 2500, 3000, 4000, 5000])
    datacenter_load = 340  # kW typical
    
    results = []
    for area in panel_areas:
        panels = RadiativeHeatRejection(panel_area_m2=area)
        
        # Average capacity estimate (clear night conditions)
        avg_capacity = panels.calculate_capacity(T_air_C=25, rh=0.3, cloud_fraction=0.1, T_fluid_in_C=35)
        
        # Cost estimates (simplified)
        panel_cost = area * 150  # $150/m² installed
        annual_savings = avg_capacity * 0.5 * 8760 * 0.10 / 1000  # 50% utilization, $0.10/kWh
        
        results.append({
            'area_m2': area,
            'avg_capacity_kW': avg_capacity,
            'capacity_fraction': avg_capacity / (datacenter_load * 1.3) * 100,
            'panel_cost_k': panel_cost / 1000,
            'annual_savings_k': annual_savings / 1000,
            'simple_payback_yr': panel_cost / max(annual_savings, 1),
        })
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].plot(df['area_m2'], df['avg_capacity_kW'], 'b-', linewidth=2, marker='s')
    axes[0].axhline(y=datacenter_load * 1.3, color='r', linestyle='--', label=f'Rejection load ({datacenter_load*1.3:.0f} kW)')
    axes[0].set_xlabel('Panel Area (m²)')
    axes[0].set_ylabel('Radiative Capacity (kW)')
    axes[0].set_title('Cooling Capacity vs Panel Area')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(df['area_m2'].astype(str), df['capacity_fraction'], color='steelblue')
    axes[1].axhline(y=100, color='r', linestyle='--', label='100% of load')
    axes[1].set_xlabel('Panel Area (m²)')
    axes[1].set_ylabel('% of Heat Rejection Load')
    axes[1].set_title('Load Coverage')
    axes[1].legend()
    
    axes[2].plot(df['area_m2'], df['simple_payback_yr'], 'g-', linewidth=2, marker='o')
    axes[2].axhline(y=5, color='orange', linestyle='--', label='5-year target')
    axes[2].set_xlabel('Panel Area (m²)')
    axes[2].set_ylabel('Simple Payback (years)')
    axes[2].set_title('Economic Payback Period')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/sensitivity_panel_area.png', dpi=150)
    plt.close()
    
    return df


def comprehensive_sensitivity_analysis():
    """Generate comprehensive sensitivity heatmap."""
    print("\n--- Comprehensive Sensitivity Heatmap ---")
    
    humidity_range = np.linspace(0.2, 0.8, 7)
    temp_range = np.linspace(15, 40, 6)
    
    cooling_matrix = np.zeros((len(humidity_range), len(temp_range)))
    
    for i, rh in enumerate(humidity_range):
        for j, T_air_C in enumerate(temp_range):
            T_air_K = T_air_C + 273.15
            sky = calculate_sky_conditions(T_air_C, rh, cloud_fraction=0.0)
            cooling = selective_emitter_cooling_realistic(
                T_surface_K=T_air_K,
                T_sky_K=sky['T_sky_K'],
                T_air_K=T_air_K,
                relative_humidity=rh,
                cloud_fraction=0.0,
            )
            cooling_matrix[i, j] = cooling['q_total']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(cooling_matrix, cmap='RdYlBu', aspect='auto', origin='lower',
                   extent=[temp_range[0], temp_range[-1], humidity_range[0]*100, humidity_range[-1]*100])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Net Cooling Power (W/m²)')
    
    contour = ax.contour(temp_range, humidity_range*100, cooling_matrix, 
                         levels=[0, 25, 50, 75, 100], colors='black', linewidths=1)
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.0f W/m²')
    
    ax.set_xlabel('Ambient Temperature (°C)')
    ax.set_ylabel('Relative Humidity (%)')
    ax.set_title('Radiative Cooling Capacity: Temperature × Humidity (Clear Sky)')
    
    plt.tight_layout()
    plt.savefig('figures/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive sensitivity analysis saved to figures/sensitivity_analysis.png")


def main():
    print("=" * 60)
    print("SENSITIVITY ANALYSIS - RADIATIVE COOLING SYSTEM")
    print("=" * 60)
    
    os.makedirs('figures', exist_ok=True)
    
    df_humidity = analyze_humidity_sensitivity()
    print(f"  Humidity range analyzed: {df_humidity['rh'].min():.0f}% - {df_humidity['rh'].max():.0f}%")
    
    df_clouds = analyze_cloud_sensitivity()
    print(f"  Cloud range analyzed: {df_clouds['cloud'].min():.0f}% - {df_clouds['cloud'].max():.0f}%")
    
    df_area = analyze_panel_area_scaling()
    print(f"  Panel areas analyzed: {df_area['area_m2'].min():.0f} - {df_area['area_m2'].max():.0f} m²")
    
    comprehensive_sensitivity_analysis()
    
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
