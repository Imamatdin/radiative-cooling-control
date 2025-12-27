"""Wind-dependent convection model (stub).
"""

def convective_coefficient(wind_speed, base_h=5.0):
    """Return convective heat transfer coefficient (W/m2K)."""
    # Simple empirical relation
    return base_h + 2.0 * wind_speed
