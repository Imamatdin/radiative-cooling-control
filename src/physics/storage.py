"""
Thermal Energy Storage (Cold Water Tank) Model - CORRECTED.

Properly sized for a 10 m² radiative cooling panel system.
"""

import numpy as np


class ThermalStorageTank:
    """
    Cold water storage tank for time-shifting radiative cooling.
    
    Physics:
    - Water has specific heat of 4186 J/(kg·K)
    - 1 m³ of water = 1000 kg
    - To store 10 kWh with ΔT of 10°C: need ~860 kg = 0.86 m³
    
    We size for ~20 kWh storage (conservative) = ~2 m³ tank
    """
    
    def __init__(
        self,
        volume_m3: float = 2.0,            # Realistic size for 10m² panel
        T_initial_C: float = 25.0,
        T_min_C: float = 12.0,             # Can cool to 12°C on good nights
        T_max_C: float = 30.0,             # Above this, tank provides no cooling
        T_ambient_C: float = 25.0,
        UA_loss: float = 5.0,              # W/K, well-insulated tank
        cp_water: float = 4186.0,
        rho_water: float = 1000.0,
    ):
        self.volume = volume_m3
        self.T_min = T_min_C + 273.15
        self.T_max = T_max_C + 273.15
        self.T_ambient = T_ambient_C + 273.15
        self.UA_loss = UA_loss
        self.cp = cp_water
        self.rho = rho_water
        
        self.mass = volume_m3 * rho_water
        self.thermal_capacity = self.mass * cp_water
        
        self.T = T_initial_C + 273.15
        
        self.total_charged_kWh = 0.0
        self.total_discharged_kWh = 0.0
        self.total_loss_kWh = 0.0
    
    def reset(self, T_initial_C: float = 25.0):
        """Reset tank."""
        self.T = T_initial_C + 273.15
        self.total_charged_kWh = 0.0
        self.total_discharged_kWh = 0.0
        self.total_loss_kWh = 0.0
    
    @property
    def T_C(self) -> float:
        return self.T - 273.15
    
    @property
    def state_of_charge(self) -> float:
        """SOC: 1 = cold (charged), 0 = warm (discharged)."""
        return np.clip((self.T_max - self.T) / (self.T_max - self.T_min), 0, 1)
    
    @property 
    def capacity_kWh(self) -> float:
        """Total storage capacity."""
        delta_T = self.T_max - self.T_min
        return self.thermal_capacity * delta_T / 3.6e6
    
    @property
    def available_cooling_kWh(self) -> float:
        """Cooling available (cold stored)."""
        delta_T = self.T_max - self.T
        return max(0, self.thermal_capacity * delta_T / 3.6e6)
    
    def charge(self, Q_cooling_W: float, dt_s: float = 3600.0) -> dict:
        """
        Charge tank with cold from radiative cooling.
        
        Cooling removes heat → temperature drops → SOC increases
        """
        # Heat loss (ambient warms the tank)
        Q_loss_W = self.UA_loss * (self.T_ambient - self.T)
        
        # Net heat flow: loss adds heat, cooling removes heat
        Q_net_W = Q_loss_W - Q_cooling_W  # Negative = cooling dominates
        
        # Temperature change
        dT = Q_net_W * dt_s / self.thermal_capacity
        
        T_old = self.T
        self.T = np.clip(self.T + dT, self.T_min, self.T_max)
        actual_dT = self.T - T_old
        
        # Energy accounting
        energy_stored_kWh = -actual_dT * self.thermal_capacity / 3.6e6
        loss_kWh = max(0, Q_loss_W) * dt_s / 3.6e6
        
        if energy_stored_kWh > 0:
            self.total_charged_kWh += energy_stored_kWh
        self.total_loss_kWh += loss_kWh
        
        return {
            'energy_stored_kWh': max(0, energy_stored_kWh),
            'T_tank_C': self.T_C,
            'SOC': self.state_of_charge,
            'limited': self.T <= self.T_min + 0.1,
        }
    
    def discharge(self, Q_demand_W: float, dt_s: float = 3600.0) -> dict:
        """
        Discharge tank to meet cooling demand.
        
        Demand adds heat → temperature rises → SOC decreases
        """
        # Heat loss
        Q_loss_W = self.UA_loss * (self.T_ambient - self.T)
        
        # Total heat addition
        Q_net_W = Q_demand_W + max(0, Q_loss_W)
        
        # Temperature change
        dT = Q_net_W * dt_s / self.thermal_capacity
        
        T_old = self.T
        self.T = np.clip(self.T + dT, self.T_min, self.T_max)
        actual_dT = self.T - T_old
        
        # Energy delivered (positive if we warmed up = delivered cold)
        energy_delivered_kWh = actual_dT * self.thermal_capacity / 3.6e6
        
        # Did we meet demand?
        demanded_kWh = Q_demand_W * dt_s / 3.6e6
        met_fraction = min(1.0, energy_delivered_kWh / max(0.001, demanded_kWh))
        
        if energy_delivered_kWh > 0:
            self.total_discharged_kWh += energy_delivered_kWh
        
        return {
            'energy_delivered_kWh': max(0, energy_delivered_kWh),
            'demand_met_fraction': met_fraction,
            'T_tank_C': self.T_C,
            'SOC': self.state_of_charge,
            'depleted': self.T >= self.T_max - 0.1,
        }
    
    def get_stats(self) -> dict:
        """Get statistics."""
        efficiency = self.total_discharged_kWh / max(0.001, self.total_charged_kWh)
        return {
            'T_tank_C': self.T_C,
            'SOC': self.state_of_charge,
            'capacity_kWh': self.capacity_kWh,
            'total_charged_kWh': self.total_charged_kWh,
            'total_discharged_kWh': self.total_discharged_kWh,
            'total_loss_kWh': self.total_loss_kWh,
            'efficiency': min(1.0, efficiency),  # Cap at 100%
        }


if __name__ == "__main__":
    print("=" * 60)
    print("THERMAL STORAGE TANK - VALIDATION (CORRECTED)")
    print("=" * 60)
    
    # Realistic sizing: 2 m³ tank for 10 m² panel system
    tank = ThermalStorageTank(
        volume_m3=2.0,
        T_initial_C=25.0,
        T_min_C=12.0,
        T_max_C=30.0,
    )
    
    print(f"\nTank Specifications:")
    print(f"  Volume: {tank.volume} m³ ({tank.mass:.0f} kg water)")
    print(f"  Temp range: {tank.T_min-273.15:.0f}°C to {tank.T_max-273.15:.0f}°C")
    print(f"  Total capacity: {tank.capacity_kWh:.1f} kWh")
    
    print(f"\nInitial state:")
    print(f"  Temperature: {tank.T_C:.1f}°C")
    print(f"  SOC: {tank.state_of_charge:.1%}")
    print(f"  Available cooling: {tank.available_cooling_kWh:.1f} kWh")
    
    # Simulate night charging (10 hours, ~800 W average from panel)
    print(f"\n--- Night Charging (10 hours @ 800 W average) ---")
    for hour in range(10):
        # Varying cooling power (better early night, worse toward dawn)
        Q_cool = 1000 - hour * 50  # 1000W → 550W
        result = tank.charge(Q_cooling_W=Q_cool, dt_s=3600)
        print(f"Hour {hour+1}: Q={Q_cool}W → T={result['T_tank_C']:.1f}°C, SOC={result['SOC']:.1%}")
    
    print(f"\nAfter night: {tank.available_cooling_kWh:.1f} kWh stored")
    
    # Simulate day discharging (14 hours, variable datacenter load)
    print(f"\n--- Day Discharging (14 hours, variable load) ---")
    loads = [300, 400, 600, 800, 1000, 1200, 1500, 1500, 1200, 1000, 800, 600, 400, 300]
    
    for hour, load in enumerate(loads):
        result = tank.discharge(Q_demand_W=load, dt_s=3600)
        status = "" if result['demand_met_fraction'] > 0.99 else f"⚠ {result['demand_met_fraction']:.0%} met"
        depleted = " DEPLETED!" if result['depleted'] else ""
        print(f"Hour {hour+1}: Load={load}W → T={result['T_tank_C']:.1f}°C, SOC={result['SOC']:.1%} {status}{depleted}")
        
        if result['depleted']:
            print("  Tank depleted - would need backup cooling!")
            break
    
    # Final stats
    stats = tank.get_stats()
    print(f"\n--- Final Statistics ---")
    print(f"  Total charged: {stats['total_charged_kWh']:.2f} kWh")
    print(f"  Total discharged: {stats['total_discharged_kWh']:.2f} kWh")
    print(f"  Total losses: {stats['total_loss_kWh']:.2f} kWh")
    print(f"  Round-trip efficiency: {stats['efficiency']:.1%}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
