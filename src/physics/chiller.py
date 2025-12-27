"""
Chiller Model for Datacenter Cooling.

The chiller is the main cooling system. It uses electricity to move heat
from the datacenter (cold side) to the condenser (hot side).

Key insight: Chiller efficiency (COP) depends on condenser temperature.
Lower condenser temp = higher COP = less electricity = less water at power plant.

References:
- ASHRAE Handbook: Fundamentals
- DOE Commercial Building Benchmarks
"""

import numpy as np


class Chiller:
    """
    Vapor-compression chiller model.
    
    COP (Coefficient of Performance) = Q_cooling / W_electric
    
    COP depends on:
    - Evaporator temperature (cold side, ~7°C for chilled water)
    - Condenser temperature (hot side, 30-45°C typically)
    - Part load ratio (efficiency drops at low loads)
    """
    
    def __init__(
        self,
        capacity_kW: float = 400,           # Rated cooling capacity
        rated_COP: float = 5.5,             # COP at rated conditions
        T_evap_design_C: float = 7.0,       # Design evaporator temp
        T_cond_design_C: float = 35.0,      # Design condenser temp
    ):
        """
        Initialize chiller.
        
        Args:
            capacity_kW: Rated cooling capacity [kW]
            rated_COP: COP at design conditions
            T_evap_design_C: Design evaporator temperature [°C]
            T_cond_design_C: Design condenser temperature [°C]
        """
        self.capacity = capacity_kW
        self.rated_COP = rated_COP
        self.T_evap_design = T_evap_design_C
        self.T_cond_design = T_cond_design_C
        
        # Carnot efficiency factor (real chillers achieve ~60% of Carnot)
        self.carnot_factor = 0.6
    
    def calculate_COP(
        self,
        T_evap_C: float,
        T_cond_C: float,
        part_load_ratio: float = 1.0,
    ) -> float:
        """
        Calculate COP at given conditions.
        
        Uses modified Carnot model with empirical corrections.
        
        Args:
            T_evap_C: Evaporator temperature [°C]
            T_cond_C: Condenser temperature [°C]  
            part_load_ratio: Fraction of full load [0-1]
        
        Returns:
            COP [-]
        """
        # Convert to Kelvin
        T_evap_K = T_evap_C + 273.15
        T_cond_K = T_cond_C + 273.15
        
        # Carnot COP (theoretical maximum)
        if T_cond_K <= T_evap_K:
            return self.rated_COP  # Avoid division issues
        
        COP_carnot = T_evap_K / (T_cond_K - T_evap_K)
        
        # Real COP (fraction of Carnot)
        COP_real = self.carnot_factor * COP_carnot
        
        # Part load correction (chillers less efficient at part load)
        # Typical curve: efficiency peaks at 70-80% load
        if part_load_ratio < 0.3:
            plr_factor = 0.7 + part_load_ratio
        elif part_load_ratio < 0.8:
            plr_factor = 1.0
        else:
            plr_factor = 1.0 - 0.1 * (part_load_ratio - 0.8) / 0.2
        
        COP = COP_real * plr_factor
        
        # Clamp to reasonable range
        return np.clip(COP, 2.0, 8.0)
    
    def operate(
        self,
        Q_cooling_kW: float,
        T_cond_C: float,
        T_evap_C: float = 7.0,
    ) -> dict:
        """
        Operate chiller to meet cooling demand.
        
        Args:
            Q_cooling_kW: Required cooling [kW]
            T_cond_C: Condenser temperature [°C]
            T_evap_C: Evaporator temperature [°C]
        
        Returns:
            Dict with power consumption, heat rejection, etc.
        """
        # Limit to capacity
        Q_actual = min(Q_cooling_kW, self.capacity)
        part_load = Q_actual / self.capacity
        
        # Calculate COP
        COP = self.calculate_COP(T_evap_C, T_cond_C, part_load)
        
        # Electric power consumption
        W_electric_kW = Q_actual / COP
        
        # Heat rejected at condenser (cooling + electric input)
        Q_rejected_kW = Q_actual + W_electric_kW
        
        return {
            'Q_cooling_kW': Q_actual,
            'Q_rejected_kW': Q_rejected_kW,
            'W_electric_kW': W_electric_kW,
            'COP': COP,
            'part_load': part_load,
            'T_cond_C': T_cond_C,
        }


class CoolingTower:
    """
    Evaporative cooling tower model.
    
    Rejects heat by evaporating water. Very effective but consumes water.
    
    Performance depends on wet-bulb temperature.
    """
    
    def __init__(
        self,
        capacity_kW: float = 600,           # Heat rejection capacity
        approach_C: float = 5.0,            # Approach to wet-bulb temp
        fan_power_kW: float = 15.0,         # Fan power at full speed
        water_consumption_L_kWh: float = 1.8,  # Water evaporated per kWh rejected
    ):
        """
        Initialize cooling tower.
        
        Args:
            capacity_kW: Maximum heat rejection [kW]
            approach_C: Approach temperature (outlet - wet bulb) [°C]
            fan_power_kW: Fan power at full load [kW]
            water_consumption_L_kWh: Water consumption [L/kWh]
        """
        self.capacity = capacity_kW
        self.approach = approach_C
        self.fan_power = fan_power_kW
        self.water_rate = water_consumption_L_kWh
    
    def calculate_wet_bulb(self, T_air_C: float, rh: float) -> float:
        """
        Estimate wet-bulb temperature.
        
        Using simplified Stull formula.
        """
        T = T_air_C
        RH = rh * 100  # Convert to percent
        
        T_wb = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
               np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
               0.00391838 * RH**1.5 * np.arctan(0.023101 * RH) - 4.686035
        
        return T_wb
    
    def operate(
        self,
        Q_reject_kW: float,
        T_air_C: float,
        rh: float,
    ) -> dict:
        """
        Operate cooling tower to reject heat.
        
        Args:
            Q_reject_kW: Heat to reject [kW]
            T_air_C: Ambient air temperature [°C]
            rh: Relative humidity [0-1]
        
        Returns:
            Dict with condenser water temp, water consumption, etc.
        """
        # Calculate wet-bulb
        T_wb = self.calculate_wet_bulb(T_air_C, rh)
        
        # Outlet water temperature (approach above wet-bulb)
        T_water_out_C = T_wb + self.approach
        
        # Part load
        Q_actual = min(Q_reject_kW, self.capacity)
        part_load = Q_actual / self.capacity
        
        # Fan power (cubic relationship with airflow)
        W_fan_kW = self.fan_power * part_load ** 2.5
        
        # Water consumption (evaporation)
        water_L = Q_actual * self.water_rate
        
        return {
            'Q_rejected_kW': Q_actual,
            'T_water_out_C': T_water_out_C,
            'T_wetbulb_C': T_wb,
            'W_fan_kW': W_fan_kW,
            'water_consumption_L': water_L,
            'part_load': part_load,
        }


class RadiativeHeatRejection:
    """
    Radiative panel system for heat rejection.
    
    Alternative to cooling tower - rejects heat to sky without water.
    Works best at night when sky is cold.
    """
    
    def __init__(
        self,
        panel_area_m2: float = 2000,
        efficiency: float = 0.70,           # Heat exchanger efficiency
    ):
        """
        Initialize radiative heat rejection system.
        
        Args:
            panel_area_m2: Total panel area [m²]
            efficiency: Heat exchange efficiency
        """
        self.area = panel_area_m2
        self.efficiency = efficiency
    
    def calculate_capacity(
        self,
        T_air_C: float,
        rh: float,
        cloud_fraction: float,
        T_fluid_in_C: float,
    ) -> float:
        """
        Calculate heat rejection capacity.
        
        Args:
            T_air_C: Ambient temperature [°C]
            rh: Relative humidity [0-1]
            cloud_fraction: Cloud cover [0-1]
            T_fluid_in_C: Inlet fluid temperature [°C]
        
        Returns:
            Maximum heat rejection capacity [kW]
        """
        # Sky temperature (Berdahl-Martin model simplified)
        T_air_K = T_air_C + 273.15
        T_dp_C = T_air_C - ((1 - rh) * 20)  # Approximate dew point
        T_dp_K = T_dp_C + 273.15
        
        # Clear sky temperature
        T_sky_clear = T_air_K * (0.711 + 0.0056 * T_dp_C + 0.000073 * T_dp_C**2) ** 0.25
        
        # Cloud correction
        T_sky_K = T_sky_clear + cloud_fraction * (T_air_K - T_sky_clear) * 0.8
        
        # Radiative heat transfer
        T_surface_K = T_fluid_in_C + 273.15 - 2  # Surface slightly below fluid
        sigma = 5.67e-8
        emissivity = 0.95
        
        q_rad = emissivity * sigma * (T_surface_K**4 - T_sky_K**4)  # W/m²
        
        # Net cooling (subtract convective gain if surface below ambient)
        h_conv = 10  # W/m²·K
        q_conv = h_conv * (T_air_K - T_surface_K)
        
        q_net = q_rad - q_conv  # W/m²
        q_net = max(0, q_net)
        
        # Total capacity
        Q_capacity_kW = self.area * q_net * self.efficiency / 1000
        
        return Q_capacity_kW
    
    def operate(
        self,
        Q_reject_kW: float,
        T_air_C: float,
        rh: float,
        cloud_fraction: float,
        T_fluid_in_C: float,
        flow_rate_kg_s: float = 10.0,
    ) -> dict:
        """
        Operate radiative heat rejection system.
        
        Args:
            Q_reject_kW: Heat to reject [kW]
            T_air_C: Ambient temperature [°C]
            rh: Relative humidity [0-1]
            cloud_fraction: Cloud cover [0-1]
            T_fluid_in_C: Inlet fluid temperature [°C]
            flow_rate_kg_s: Fluid flow rate [kg/s]
        
        Returns:
            Dict with actual heat rejected, outlet temp, etc.
        """
        # Calculate capacity
        capacity = self.calculate_capacity(T_air_C, rh, cloud_fraction, T_fluid_in_C)
        
        # Actual heat rejected
        Q_actual = min(Q_reject_kW, capacity)
        
        # Outlet temperature
        cp = 4186  # J/kg·K
        if flow_rate_kg_s > 0.1:
            dT = Q_actual * 1000 / (flow_rate_kg_s * cp)
        else:
            dT = 0
        
        T_fluid_out_C = T_fluid_in_C - dT
        
        # Pump power (small compared to cooling tower fan)
        W_pump_kW = 2.0 * (flow_rate_kg_s / 10.0) ** 2
        
        return {
            'Q_rejected_kW': Q_actual,
            'Q_capacity_kW': capacity,
            'T_fluid_out_C': T_fluid_out_C,
            'W_pump_kW': W_pump_kW,
            'water_consumption_L': 0.0,  # Zero water!
            'utilization': Q_actual / max(0.1, capacity),
        }


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHILLER & HEAT REJECTION VALIDATION")
    print("=" * 60)
    
    # Test chiller
    chiller = Chiller(capacity_kW=400, rated_COP=5.5)
    
    print("\n--- Chiller COP vs Condenser Temperature ---")
    for T_cond in [25, 30, 35, 40, 45]:
        COP = chiller.calculate_COP(T_evap_C=7, T_cond_C=T_cond)
        result = chiller.operate(Q_cooling_kW=340, T_cond_C=T_cond)
        print(f"T_cond={T_cond}°C: COP={COP:.2f}, Power={result['W_electric_kW']:.1f} kW")
    
    # Test cooling tower
    tower = CoolingTower(capacity_kW=600)
    
    print("\n--- Cooling Tower Performance ---")
    for T_air, rh in [(25, 0.5), (30, 0.6), (35, 0.4), (40, 0.3)]:
        result = tower.operate(Q_reject_kW=500, T_air_C=T_air, rh=rh)
        print(f"T_air={T_air}°C, RH={rh:.0%}: T_water={result['T_water_out_C']:.1f}°C, "
              f"Water={result['water_consumption_L']:.0f} L/h")
    
    # Test radiative panels
    panels = RadiativeHeatRejection(panel_area_m2=2000)
    
    print("\n--- Radiative Panel Performance ---")
    print("(Night conditions, T_fluid_in=35°C)")
    for T_air, rh, cloud in [(20, 0.3, 0.0), (25, 0.5, 0.2), (30, 0.7, 0.5), (35, 0.8, 0.8)]:
        result = panels.operate(
            Q_reject_kW=500, T_air_C=T_air, rh=rh, 
            cloud_fraction=cloud, T_fluid_in_C=35
        )
        print(f"T_air={T_air}°C, RH={rh:.0%}, Cloud={cloud:.0%}: "
              f"Capacity={result['Q_capacity_kW']:.1f} kW, "
              f"Rejected={result['Q_rejected_kW']:.1f} kW")
    
    print("\n--- Comparison: 340 kW Cooling Load ---")
    
    # Scenario: Hot day (35°C, 50% RH)
    T_air, rh = 35, 0.5
    Q_cooling = 340
    
    # With cooling tower only
    tower_result = tower.operate(Q_reject_kW=500, T_air_C=T_air, rh=rh)
    chiller_tower = chiller.operate(Q_cooling, T_cond_C=tower_result['T_water_out_C'])
    
    print(f"\nCooling Tower Only (T_air={T_air}°C):")
    print(f"  T_condenser: {chiller_tower['T_cond_C']:.1f}°C")
    print(f"  COP: {chiller_tower['COP']:.2f}")
    print(f"  Chiller Power: {chiller_tower['W_electric_kW']:.1f} kW")
    print(f"  Water: {tower_result['water_consumption_L']:.0f} L/h")
    
    # With radiative assist (night, clear)
    panels_result = panels.operate(
        Q_reject_kW=500, T_air_C=20, rh=0.3, 
        cloud_fraction=0.1, T_fluid_in_C=35
    )
    
    # Radiative pre-cools, then tower finishes
    Q_remaining = 500 - panels_result['Q_rejected_kW']
    if Q_remaining > 0:
        tower_assist = tower.operate(Q_reject_kW=Q_remaining, T_air_C=20, rh=0.3)
        T_cond_mixed = (panels_result['T_fluid_out_C'] + tower_assist['T_water_out_C']) / 2
    else:
        T_cond_mixed = panels_result['T_fluid_out_C']
        tower_assist = {'water_consumption_L': 0}
    
    chiller_hybrid = chiller.operate(Q_cooling, T_cond_C=T_cond_mixed)
    
    print(f"\nHybrid Radiative+Tower (Night, T_air=20°C):")
    print(f"  Radiative rejected: {panels_result['Q_rejected_kW']:.1f} kW")
    print(f"  T_condenser: {chiller_hybrid['T_cond_C']:.1f}°C")
    print(f"  COP: {chiller_hybrid['COP']:.2f}")
    print(f"  Chiller Power: {chiller_hybrid['W_electric_kW']:.1f} kW")
    print(f"  Water: {tower_assist['water_consumption_L']:.0f} L/h")
    
    # Savings
    power_savings = chiller_tower['W_electric_kW'] - chiller_hybrid['W_electric_kW']
    water_savings = tower_result['water_consumption_L'] - tower_assist['water_consumption_L']
    
    print(f"\n--- SAVINGS ---")
    print(f"  Power saved: {power_savings:.1f} kW ({power_savings/chiller_tower['W_electric_kW']*100:.1f}%)")
    print(f"  Water saved: {water_savings:.0f} L/h ({water_savings/tower_result['water_consumption_L']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
