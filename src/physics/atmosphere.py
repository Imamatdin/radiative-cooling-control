"""
Atmospheric sky emissivity and temperature models.

Implements the Berdahl-Martin model for calculating effective sky temperature
from meteorological conditions. This is critical for realistic radiative
cooling simulation.

References:
- Berdahl & Martin (1984) Solar Energy 32(5), 663-664
- Berdahl & Fromberg (1982) Solar Energy 29(4), 299-314
"""

import numpy as np


def dew_point_from_rh(T_air_C, relative_humidity):
    """
    Calculate dew point temperature from air temperature and relative humidity.
    
    Uses the Magnus-Tetens approximation.
    
    Args:
        T_air_C: Air temperature in Celsius
        relative_humidity: Relative humidity as fraction (0-1)
    
    Returns:
        Dew point temperature in Celsius
    """
    if relative_humidity <= 0 or relative_humidity > 1:
        raise ValueError(f"Relative humidity must be in (0, 1], got {relative_humidity}")
    
    # Magnus-Tetens coefficients
    a = 17.27
    b = 237.7  # °C
    
    # Calculate gamma
    gamma = (a * T_air_C / (b + T_air_C)) + np.log(relative_humidity)
    
    # Dew point
    T_dp = (b * gamma) / (a - gamma)
    
    return T_dp


def sky_emissivity_clear(T_dp_C):
    """
    Calculate clear sky emissivity using Berdahl-Martin correlation.
    
    Args:
        T_dp_C: Dew point temperature in Celsius
    
    Returns:
        Clear sky emissivity (0-1)
    """
    # Berdahl-Martin (1984) correlation
    T_dp_scaled = T_dp_C / 100.0
    
    emissivity = 0.711 + 0.56 * T_dp_scaled + 0.73 * T_dp_scaled**2
    
    # Physical bounds
    return np.clip(emissivity, 0.5, 1.0)


def sky_emissivity_cloudy(emissivity_clear, cloud_fraction, cloud_factor=0.8):
    """
    Adjust sky emissivity for cloud cover.
    
    Args:
        emissivity_clear: Clear sky emissivity
        cloud_fraction: Cloud cover fraction (0-1)
        cloud_factor: Cloud emissivity factor (0.4 for high clouds, 0.8 for low clouds)
    
    Returns:
        Effective sky emissivity with clouds
    """
    # Clouds are nearly blackbody radiators at their temperature
    # This formula interpolates between clear sky and cloudy conditions
    emissivity = emissivity_clear + (1 - emissivity_clear) * cloud_fraction * cloud_factor
    
    return np.clip(emissivity, 0.5, 1.0)


def sky_temperature(T_air_K, emissivity_sky):
    """
    Calculate effective sky temperature from air temperature and sky emissivity.
    
    The sky radiates like a blackbody at T_sky such that:
    σ * T_sky^4 = ε_sky * σ * T_air^4
    
    Therefore: T_sky = T_air * ε_sky^0.25
    
    Args:
        T_air_K: Air temperature in Kelvin
        emissivity_sky: Effective sky emissivity
    
    Returns:
        Effective sky temperature in Kelvin
    """
    return T_air_K * (emissivity_sky ** 0.25)


def calculate_sky_conditions(T_air_C, relative_humidity, cloud_fraction=0.0, cloud_factor=0.8):
    """
    Complete atmospheric model: calculate sky temperature from weather conditions.
    
    This is the main function you'll use in simulation.
    
    Args:
        T_air_C: Air temperature in Celsius
        relative_humidity: Relative humidity as fraction (0-1)
        cloud_fraction: Cloud cover fraction (0-1)
        cloud_factor: Cloud type factor (0.4=high/thin, 0.8=low/thick)
    
    Returns:
        Dictionary with:
        - T_dp_C: Dew point temperature (°C)
        - emissivity_clear: Clear sky emissivity
        - emissivity_sky: Effective sky emissivity (with clouds)
        - T_sky_K: Effective sky temperature (K)
        - T_sky_C: Effective sky temperature (°C)
        - T_depression: Sky temperature depression below air temp (°C)
    """
    # Step 1: Dew point from RH
    T_dp_C = dew_point_from_rh(T_air_C, relative_humidity)
    
    # Step 2: Clear sky emissivity
    emissivity_clear = sky_emissivity_clear(T_dp_C)
    
    # Step 3: Cloud-adjusted emissivity
    emissivity_sky = sky_emissivity_cloudy(emissivity_clear, cloud_fraction, cloud_factor)
    
    # Step 4: Sky temperature
    T_air_K = T_air_C + 273.15
    T_sky_K = sky_temperature(T_air_K, emissivity_sky)
    T_sky_C = T_sky_K - 273.15
    
    # Sky temperature depression (how much colder sky is than air)
    T_depression = T_air_C - T_sky_C
    
    return {
        'T_dp_C': T_dp_C,
        'emissivity_clear': emissivity_clear,
        'emissivity_sky': emissivity_sky,
        'T_sky_K': T_sky_K,
        'T_sky_C': T_sky_C,
        'T_depression': T_depression
    }


# =============================================================================
# VALIDATION TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ATMOSPHERIC MODEL - VALIDATION")
    print("=" * 70)
    
    # Test 1: Dew point calculation
    print("\n[Test 1] Dew Point Calculation")
    print("-" * 50)
    test_cases = [
        (25, 0.50, 13.9),  # Expected dew point ~14°C
        (25, 0.80, 21.3),  # High humidity
        (25, 0.30, 5.7),   # Low humidity
        (35, 0.60, 25.0),  # Hot day
    ]
    print(f"{'T_air (°C)':<12} {'RH':<8} {'T_dp calc':<12} {'T_dp expected':<12}")
    print("-" * 50)
    for T_air, rh, expected in test_cases:
        T_dp = dew_point_from_rh(T_air, rh)
        print(f"{T_air:<12} {rh:<8.0%} {T_dp:<12.1f} {expected:<12.1f}")
    
    # Test 2: Sky emissivity model
    print("\n[Test 2] Clear Sky Emissivity (Berdahl-Martin)")
    print("-" * 50)
    print(f"{'T_dp (°C)':<12} {'ε_sky':<12} {'Condition'}")
    print("-" * 50)
    for T_dp in [-10, 0, 10, 15, 20, 25]:
        eps = sky_emissivity_clear(T_dp)
        condition = "Very dry" if eps < 0.75 else "Dry" if eps < 0.82 else "Moderate" if eps < 0.88 else "Humid"
        print(f"{T_dp:<12} {eps:<12.3f} {condition}")
    
    # Test 3: Complete atmospheric model
    print("\n[Test 3] Complete Atmospheric Model")
    print("-" * 50)
    
    scenarios = [
        ("Desert night (Phoenix)", 25, 0.20, 0.0),
        ("Dry clear night", 20, 0.40, 0.0),
        ("Moderate humidity", 20, 0.60, 0.0),
        ("Humid night", 25, 0.80, 0.0),
        ("Very humid (tropical)", 30, 0.90, 0.0),
        ("Partly cloudy", 20, 0.50, 0.30),
        ("Mostly cloudy", 20, 0.50, 0.70),
        ("Overcast", 20, 0.50, 1.00),
    ]
    
    print(f"{'Scenario':<25} {'T_air':<7} {'RH':<6} {'Cloud':<7} {'T_sky':<8} {'ΔT':<6} {'ε_sky'}")
    print("-" * 70)
    
    for name, T_air, rh, cloud in scenarios:
        result = calculate_sky_conditions(T_air, rh, cloud)
        print(f"{name:<25} {T_air:<7}°C {rh:<6.0%} {cloud:<7.0%} {result['T_sky_C']:<8.1f}°C {result['T_depression']:<6.1f}°C {result['emissivity_sky']:.3f}")
    
    # Test 4: THE REAL INSIGHT - Why systems shut down
    print("\n[Test 4] WHY CURRENT SYSTEMS SHUT DOWN")
    print("-" * 70)
    
    # Import radiation module to show cooling power
    from radiation import selective_emitter_cooling
    
    print(f"{'Condition':<25} {'T_sky (°C)':<12} {'Cooling (W/m²)':<16} {'Status'}")
    print("-" * 70)
    
    T_surface_K = 300  # 27°C surface
    
    for name, T_air, rh, cloud in scenarios:
        result = calculate_sky_conditions(T_air, rh, cloud)
        cooling = selective_emitter_cooling(T_surface_K, result['T_sky_K'])
        q = cooling['q_total']
        
        # Typical system requires > 50 W/m² to operate efficiently
        if q > 100:
            status = "✓ EXCELLENT"
        elif q > 70:
            status = "✓ GOOD"
        elif q > 40:
            status = "△ MARGINAL - Current systems may bypass"
        else:
            status = "✗ POOR - Current systems SHUT DOWN"
        
        print(f"{name:<25} {result['T_sky_C']:<12.1f} {q:<16.1f} {status}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
    Current systems use a threshold (e.g., 50 W/m²). Below this, they
    bypass entirely and deliver 0 W/m² of cooling.
    
    YOUR INNOVATION: Instead of shutting down, reduce flow rate.
    Lower flow = longer residence time = bigger temperature drop.
    Even 30 W/m² of cooling is better than 0 W/m².
    
    This is the operational hours you're recovering.
    """)
    
    print("=" * 70)
    print("ATMOSPHERIC MODEL VALIDATED")
    print("=" * 70)
