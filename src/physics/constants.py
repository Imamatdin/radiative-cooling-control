"""
Physical constants for radiative cooling simulation.
"""

# Stefan-Boltzmann constant [W/(m²·K⁴)]
STEFAN_BOLTZMANN = 5.670374419e-8

# Planck constant [J·s]
PLANCK = 6.62607015e-34

# Speed of light [m/s]
SPEED_OF_LIGHT = 2.99792458e8

# Boltzmann constant [J/K]
BOLTZMANN = 1.380649e-23

# Space temperature [K]
T_SPACE = 3.0

# Atmospheric window bounds [μm]
WINDOW_LOW = 8.0
WINDOW_HIGH = 13.0

# Reference temperature for window fraction calculation [K]
T_REF = 300.0

# Fraction of blackbody radiation in 8-13 μm window at 300K
# (pre-calculated from Planck integral)
F_WINDOW_300K = 0.346
