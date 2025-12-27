"""
Annual Simulation for Datacenter Radiative Cooling.

Simulates full year with hourly resolution to calculate:
- Total energy savings
- Water savings
- Operating hours
- Economic benefits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.chiller import Chiller, CoolingTower, RadiativeHeatRejection
from environment.datacenter_env import load_weather


class AnnualSimulator:
    """Simulate datacenter cooling for a full year."""
    
    def __init__(self, weather_df: pd.DataFrame, panel_area_m2: float = 2000):
        self.weather = weather_df
        self.hours = len(weather_df)
        
        self.chiller = Chiller(capacity_kW=400, rated_COP=5.5)
        self.tower = CoolingTower(capacity_kW=600)
        self.panels = RadiativeHeatRejection(panel_area_m2=panel_area_m2)
        
        self.base_load = 340  # kW
        
        # TOU pricing
        self.prices = {
            'off_peak': 0.06,
            'mid_peak': 0.12,
            'on_peak': 0.25,
        }
        self.water_price = 0.005  # $/L
        
    def get_load(self, hour_of_day: int) -> float:
        """Get cooling load based on time of day."""
        if 0 <= hour_of_day < 6:
            factor = 0.75
        elif 6 <= hour_of_day < 9:
            factor = 0.75 + 0.20 * (hour_of_day - 6) / 3
        elif 9 <= hour_of_day < 18:
            factor = 0.95 + 0.05 * np.sin((hour_of_day - 9) / 9 * np.pi)
        elif 18 <= hour_of_day < 22:
            factor = 1.0
        else:
            factor = 0.85
        return self.base_load * factor
    
    def get_price(self, hour_of_day: int) -> float:
        """Get electricity price based on TOU."""
        if 22 <= hour_of_day or hour_of_day < 6:
            return self.prices['off_peak']
        elif 14 <= hour_of_day < 19:
            return self.prices['on_peak']
        return self.prices['mid_peak']
    
    def simulate_baseline(self) -> dict:
        """Simulate tower-only baseline."""
        results = {
            'electricity_kWh': 0,
            'water_L': 0,
            'electricity_cost': 0,
            'water_cost': 0,
        }
        
        for i, row in self.weather.iterrows():
            hour_of_day = i % 24
            load = self.get_load(hour_of_day)
            price = self.get_price(hour_of_day)
            
            T_air = row['T_air_C']
            rh = row['rh']
            
            tower = self.tower.operate(load * 1.3, T_air, rh)
            chiller = self.chiller.operate(load, tower['T_water_out_C'])
            
            elec = chiller['W_electric_kW'] + tower['W_fan_kW']
            water = tower['water_consumption_L']
            
            results['electricity_kWh'] += elec
            results['water_L'] += water
            results['electricity_cost'] += elec * price
            results['water_cost'] += water * self.water_price
        
        results['total_cost'] = results['electricity_cost'] + results['water_cost']
        return results
    
    def simulate_hybrid(self, strategy: str = 'smart') -> dict:
        """Simulate hybrid radiative + tower system."""
        results = {
            'electricity_kWh': 0,
            'water_L': 0,
            'electricity_cost': 0,
            'water_cost': 0,
            'radiative_hours': 0,
            'radiative_kWh': 0,
        }
        
        hourly_data = []
        
        for i, row in self.weather.iterrows():
            hour_of_day = i % 24
            load = self.get_load(hour_of_day)
            price = self.get_price(hour_of_day)
            
            T_air = row['T_air_C']
            rh = row['rh']
            cloud = row.get('cloud', 0.3)
            
            # Calculate radiative capacity
            rad_capacity = self.panels.calculate_capacity(T_air, rh, cloud, T_fluid_in_C=35)
            
            # Strategy-based radiative fraction
            if strategy == 'always_on':
                rad_fraction = 1.0
            elif strategy == 'night_only':
                rad_fraction = 1.0 if (hour_of_day < 6 or hour_of_day >= 20) else 0.0
            elif strategy == 'smart':
                if rad_capacity > 200:
                    rad_fraction = 1.0
                elif rad_capacity > 100:
                    rad_fraction = 0.7
                elif rad_capacity > 50:
                    rad_fraction = 0.4
                else:
                    rad_fraction = 0.1
            else:
                rad_fraction = 0.5
            
            Q_reject = load * 1.3
            Q_to_rad = Q_reject * rad_fraction
            
            rad_result = self.panels.operate(Q_to_rad, T_air, rh, cloud, T_fluid_in_C=35)
            Q_to_tower = Q_reject - rad_result['Q_rejected_kW']
            
            if Q_to_tower > 0:
                tower = self.tower.operate(Q_to_tower, T_air, rh)
            else:
                tower = {'T_water_out_C': 25, 'W_fan_kW': 0, 'water_consumption_L': 0}
            
            # Condenser temperature (weighted average)
            if rad_result['Q_rejected_kW'] > 0 and Q_to_tower > 0:
                T_cond = (rad_result['Q_rejected_kW'] * rad_result['T_fluid_out_C'] + 
                          Q_to_tower * tower['T_water_out_C']) / Q_reject
            elif rad_result['Q_rejected_kW'] > 0:
                T_cond = rad_result['T_fluid_out_C']
            else:
                T_cond = tower['T_water_out_C']
            
            chiller = self.chiller.operate(load, T_cond)
            
            elec = chiller['W_electric_kW'] + tower['W_fan_kW'] + rad_result['W_pump_kW']
            water = tower['water_consumption_L']
            
            results['electricity_kWh'] += elec
            results['water_L'] += water
            results['electricity_cost'] += elec * price
            results['water_cost'] += water * self.water_price
            
            if rad_result['Q_rejected_kW'] > 10:
                results['radiative_hours'] += 1
                results['radiative_kWh'] += rad_result['Q_rejected_kW']
            
            hourly_data.append({
                'hour': i,
                'hour_of_day': hour_of_day,
                'T_air_C': T_air,
                'rad_capacity_kW': rad_capacity,
                'rad_used_kW': rad_result['Q_rejected_kW'],
                'COP': chiller['COP'],
                'electricity_kW': elec,
                'water_L': water,
            })
        
        results['total_cost'] = results['electricity_cost'] + results['water_cost']
        results['hourly_data'] = pd.DataFrame(hourly_data)
        return results


def run_annual_comparison(weather_path: str):
    """Run annual comparison between strategies."""
    print("=" * 70)
    print("ANNUAL SIMULATION - DATACENTER RADIATIVE COOLING")
    print("=" * 70)
    
    weather_df = load_weather(weather_path)
    print(f"Loaded {len(weather_df)} hours of weather data")
    
    sim = AnnualSimulator(weather_df, panel_area_m2=2000)
    
    # Baseline
    print("\nSimulating baseline (tower only)...")
    baseline = sim.simulate_baseline()
    
    # Strategies
    strategies = ['night_only', 'always_on', 'smart']
    results = {'Baseline': baseline}
    
    for strat in strategies:
        print(f"Simulating {strat}...")
        results[strat] = sim.simulate_hybrid(strategy=strat)
    
    # Print comparison table
    print("\n" + "-" * 70)
    print(f"{'Strategy':<15} {'Elec (MWh)':<12} {'Water (kL)':<12} {'Cost ($)':<12} {'Savings':<10}")
    print("-" * 70)
    
    for name, r in results.items():
        elec_mwh = r['electricity_kWh'] / 1000
        water_kl = r['water_L'] / 1000
        cost = r['total_cost']
        
        if name == 'Baseline':
            savings = '-'
        else:
            savings_pct = (baseline['total_cost'] - cost) / baseline['total_cost'] * 100
            savings = f"{savings_pct:.1f}%"
        
        print(f"{name:<15} {elec_mwh:<12.1f} {water_kl:<12.1f} ${cost:<11,.0f} {savings:<10}")
    
    print("-" * 70)
    
    # Generate figures
    generate_annual_figures(results, baseline)
    
    return results


def generate_annual_figures(results: dict, baseline: dict):
    """Generate annual performance figures."""
    os.makedirs('figures', exist_ok=True)
    
    # Figure: Annual comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    strategies = list(results.keys())
    x = np.arange(len(strategies))
    
    # Electricity
    elec_values = [results[s]['electricity_kWh']/1000 for s in strategies]
    axes[0].bar(x, elec_values, color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, rotation=45, ha='right')
    axes[0].set_ylabel('Electricity (MWh/year)')
    axes[0].set_title('Annual Electricity Consumption')
    
    # Water
    water_values = [results[s]['water_L']/1000 for s in strategies]
    axes[1].bar(x, water_values, color='teal')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, rotation=45, ha='right')
    axes[1].set_ylabel('Water (kL/year)')
    axes[1].set_title('Annual Water Consumption')
    
    # Cost
    cost_values = [results[s]['total_cost'] for s in strategies]
    colors = ['gray' if s == 'Baseline' else 'green' for s in strategies]
    axes[2].bar(x, cost_values, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(strategies, rotation=45, ha='right')
    axes[2].set_ylabel('Total Cost ($/year)')
    axes[2].set_title('Annual Operating Cost')
    
    plt.tight_layout()
    plt.savefig('figures/fig5_annual_performance.png', dpi=150)
    plt.close()
    print("✓ Figure 5: Annual performance comparison saved")
    
    # Figure: Monthly breakdown for smart strategy
    if 'smart' in results and 'hourly_data' in results['smart']:
        hourly = results['smart']['hourly_data']
        hourly['month'] = (hourly['hour'] // 720) % 12 + 1
        
        monthly = hourly.groupby('month').agg({
            'rad_used_kW': 'sum',
            'electricity_kW': 'sum',
            'water_L': 'sum',
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax.bar(monthly['month'], monthly['rad_used_kW']/1000, color='steelblue')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.set_xlabel('Month')
        ax.set_ylabel('Radiative Cooling (MWh)')
        ax.set_title('Monthly Radiative Cooling Contribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('figures/fig_monthly_radiative.png', dpi=150)
        plt.close()


if __name__ == "__main__":
    run_annual_comparison('data/weather/phoenix_az_tmy.csv')
