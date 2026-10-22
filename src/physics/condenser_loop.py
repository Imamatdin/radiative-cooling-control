"""
Energy-consistent condenser-water loop: chiller -> radiative panels (precooler) -> cooling tower (trim).

Replaces the per-component bookkeeping previously in datacenter_env.step(), which
(a) held the panel surface at a fixed 33 C while letting the fluid leave at a
temperature computed from a 10 kg/s flow, so the fluid could exit colder than
both the panel surface and the ambient air (second-law violation), and
(b) used a clear sky for every hour (cloud column absent from the weather files).

Plant layout (constant-flow condenser loop, flow m_cw):

    chiller condenser --T_ret--> [ split: alpha*m through panel field, (1-alpha)*m bypass ]
        --T_mix--> cooling tower (trims to its setpoint, or bypassed) --T_cw_in--> chiller

Panel field: fluid is marched through the field area. Local heat flux leaving
the fluid per m^2 (front face; back face insulated):

    q(T) = F' * [ eps*sigma*T^4 - eps*L_down + h(v)*(T - T_air) - a_sol*GHI ]

with L_down the measured/reanalysis downwelling longwave (PVGIS 'IR(h)', ERA5),
h(v) = 5.7 + 3.8 v (McAdams / Duffie & Beckman wind correlation), and F' a plate
efficiency factor (surface close to local fluid temperature). Flux is clipped at
zero: a valve bypasses the field rather than letting it heat the loop. The fluid
can therefore never cool below the panel stagnation temperature, which is the
second-law bound tested in tests/test_physics.py.

Everything is numpy-vectorized so control baselines (DP / MPC) can evaluate
many (load, alpha) points per hour in one call.
"""

import numpy as np

SIGMA = 5.670374419e-8
CP_W = 4186.0


class PlantConfig:
    """Design parameters. Values and their basis are listed so reviewers can attack them."""

    def __init__(self, **kw):
        # Chiller
        self.chiller_capacity_kW = 1200.0   # 1.2x peak IT cooling (N+margin); 1000 kW could not meet peak load + tank charging
        self.carnot_factor = 0.6            # unchanged from original model
        self.T_evap_C = 7.0                 # unchanged
        self.cond_approach_K = 2.0          # refrigerant condensing T above leaving condenser water
        self.cop_min, self.cop_max = 2.0, 8.0
        self.T_cw_min_C = 15.0              # minimum entering condenser water for a centrifugal chiller
        # Condenser loop
        self.cw_range_K = 5.5               # design range at full rejection
        self.Q_reject_design_kW = 1200.0 * 1.17
        # Cooling tower
        self.tower_approach_K = 5.0         # unchanged
        self.tower_capacity_kW = 1800.0
        self.tower_fan_kW = 18.0            # ~0.01 kW fan per kW rejected at design
        self.tower_water_L_per_kWh = 1.8    # evaporation ~1.48 L/kWh (3.6 MJ / 2.43 MJ/kg) + blowdown at ~5 cycles
        # Panel field
        self.panel_area_m2 = 5000.0
        self.emissivity = 0.95
        self.solar_absorptance = 0.05       # good multilayer film: 3-5 %; sensitivity-test this
        self.plate_factor = 0.9             # F', surface-to-fluid
        self.panel_dp_kPa = 100.0           # extra pressure drop through field + HX
        self.pump_eff = 0.7
        self.n_segments = 40
        self.radiative = True               # False -> same field as a pure convective dry cooler (baseline)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(f"unknown PlantConfig field {k}")
            setattr(self, k, v)

    @property
    def m_cw(self):
        return self.Q_reject_design_kW * 1000.0 / (CP_W * self.cw_range_K)


def wet_bulb_C(T_air_C, rh):
    """Stull (2011) wet-bulb approximation; rh in 0-1. Valid ~5-99 % RH, -20..50 C."""
    T = np.asarray(T_air_C, float)
    RH = np.clip(np.asarray(rh, float), 0.05, 0.99) * 100
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH)
            - np.arctan(RH - 1.676331) + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)


def panel_flux_W_m2(T_K, T_air_C, L_down, ghi, wind, cfg):
    """Heat flux leaving the fluid [W/m2] at local fluid temperature T_K (before clipping)."""
    h = 5.7 + 3.8 * np.asarray(wind, float)
    q = h * (T_K - (np.asarray(T_air_C, float) + 273.15))
    if cfg.radiative:
        q = q + cfg.emissivity * (SIGMA * T_K ** 4 - np.asarray(L_down, float)) - cfg.solar_absorptance * np.asarray(ghi, float)
    return cfg.plate_factor * q


def stagnation_T_K(T_air_C, L_down, ghi, wind, cfg, shape=None):
    """Fluid temperature at which the field's net flux is zero (lowest reachable outlet). Bisection."""
    lo = np.full(np.broadcast(np.asarray(T_air_C, float), np.asarray(L_down, float)).shape if shape is None else shape, 150.0)
    hi = np.full_like(lo, 400.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        q = panel_flux_W_m2(mid, T_air_C, L_down, ghi, wind, cfg)
        lo = np.where(q < 0, mid, lo)
        hi = np.where(q < 0, hi, mid)
    return 0.5 * (lo + hi)


def panel_outlet(T_in_C, mdot, T_air_C, L_down, ghi, wind, cfg, T_floor=None):
    """March fluid through the field. Returns (T_out_C, Q_kW). Vectorized over all inputs.
    Each segment's outlet is floored at the stagnation temperature, so the fluid can never be
    cooled below the temperature at which the panel stops rejecting heat (second-law bound).
    mdot ~ 0 means the field is valved off: outlet = inlet, no heat rejected."""
    T_in = np.asarray(T_in_C, float)
    mdot = np.asarray(mdot, float)
    shape = np.broadcast(T_in, mdot, np.asarray(T_air_C, float), np.asarray(L_down, float)).shape
    off = np.broadcast_to(mdot < 1e-3, shape)
    m_safe = np.broadcast_to(np.maximum(mdot, 1e-3), shape)
    if T_floor is None:
        T_floor = stagnation_T_K(T_air_C, L_down, ghi, wind, cfg)
    T_floor = np.broadcast_to(T_floor, shape)
    T = np.broadcast_to(T_in + 273.15, shape).copy()
    dA = cfg.panel_area_m2 / cfg.n_segments
    for _ in range(cfg.n_segments):
        q = np.maximum(panel_flux_W_m2(T, T_air_C, L_down, ghi, wind, cfg), 0.0)
        T = np.where(q > 0, np.maximum(T - q * dA / (m_safe * CP_W), T_floor), T)
    T_out = np.where(off, np.broadcast_to(T_in, shape), T - 273.15)
    Q_kW = np.where(off, 0.0, m_safe * CP_W * (T_in - T_out) / 1000.0)
    return T_out, Q_kW


def chiller_cop(T_cw_leave_C, plr, cfg):
    Te = cfg.T_evap_C + 273.15
    Tc = np.asarray(T_cw_leave_C, float) + cfg.cond_approach_K + 273.15
    cop = cfg.carnot_factor * Te / np.maximum(Tc - Te, 1e-3)
    plr = np.asarray(plr, float)
    plr_f = np.where(plr < 0.3, 0.7 + plr, np.where(plr < 0.8, 1.0, 1.0 - 0.1 * (plr - 0.8) / 0.2))
    return np.clip(cop * plr_f, cfg.cop_min, cfg.cop_max)


def plant_hour(Q_evap_kW, alpha, wx, cfg, n_iter=20):
    """
    One hour of steady plant operation.

    Q_evap_kW : chiller evaporator load [kW] (array ok). Must be <= chiller capacity.
    alpha     : fraction of condenser flow routed through the panel field [0,1].
    wx        : dict with T_air_C, rh, L_down (W/m2), ghi (W/m2), wind (m/s); scalars or broadcastable arrays.
    Returns dict of arrays (kW, L per hour, C).
    """
    Q_evap = np.asarray(Q_evap_kW, float)
    alpha = np.clip(np.asarray(alpha, float), 0.0, 1.0)
    shape = np.broadcast(Q_evap, alpha, np.asarray(wx["T_air_C"])).shape
    Q_evap = np.broadcast_to(Q_evap, shape)
    alpha = np.broadcast_to(alpha, shape)
    m = cfg.m_cw
    mcp = m * CP_W / 1000.0  # kW/K
    T_wb = wet_bulb_C(wx["T_air_C"], wx["rh"])
    T_set = np.broadcast_to(np.maximum(T_wb + cfg.tower_approach_K, cfg.T_cw_min_C), shape)
    plr = Q_evap / cfg.chiller_capacity_kW

    def chiller_side(T_cw_in):
        T_ret = T_cw_in + Q_evap * 1.17 / mcp
        for _ in range(4):
            cop = chiller_cop(T_ret, plr, cfg)
            Q_rej = Q_evap * (1 + 1 / cop)
            T_ret = T_cw_in + Q_rej / mcp
        return cop, Q_evap / cop, Q_rej, T_ret

    T_floor = stagnation_T_K(wx["T_air_C"], wx["L_down"], wx["ghi"], wx["wind"], cfg)

    def loop_side(T_ret):
        T_p_out, _ = panel_outlet(T_ret, alpha * m, wx["T_air_C"], wx["L_down"], wx["ghi"], wx["wind"], cfg, T_floor)
        T_mix = alpha * T_p_out + (1 - alpha) * T_ret
        return T_p_out, T_mix

    # Steady state: entering condenser water T satisfies one of three regimes.
    #  (1) tower on : T = T_set and the field alone cannot reach T_set
    #  (2) valve    : T = T_cw_min and the field could overcool, so part of the flow bypasses it
    #  (3) floating : tower off, field sheds exactly the rejected heat, T_mix(T) = T, T in (T_cw_min, T_set)
    def h(T):
        _, _, _, T_ret_ = chiller_side(T)
        _, T_mix_ = loop_side(T_ret_)
        return T_mix_ - T

    T_lo = np.full(shape, cfg.T_cw_min_C)
    tower_regime = h(T_set) >= 0
    valve_regime = (~tower_regime) & (h(T_lo) <= 0)
    lo, hi = T_lo.copy(), T_set.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        pos = h(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    T_cw_in = np.where(tower_regime, T_set, np.where(valve_regime, T_lo, 0.5 * (lo + hi)))

    cop, W_ch, Q_rej, T_ret = chiller_side(T_cw_in)
    T_p_out, T_mix = loop_side(T_ret)
    Q_panel = alpha * mcp * (T_ret - T_p_out)
    Q_tower = np.where(tower_regime, mcp * np.maximum(T_mix - T_set, 0.0), 0.0)
    # valve regime: field would overcool below the chiller minimum, so it is throttled to shed exactly Q_rej
    Q_panel = np.where(valve_regime, Q_rej, Q_panel)
    # steady state: heat the loop sheds (panel + tower) equals heat the chiller rejects into it
    balance_residual_kW = Q_rej - (Q_panel + Q_tower)

    tower_plr = np.minimum(Q_tower / cfg.tower_capacity_kW, 1.0)
    W_fan = cfg.tower_fan_kW * tower_plr ** 2.5
    water_L = Q_tower * cfg.tower_water_L_per_kWh
    W_pump = alpha * (cfg.panel_dp_kPa * 1000 * m / 1000.0 / cfg.pump_eff) / 1000.0
    W_pump = np.where(alpha > 1e-6, W_pump, 0.0)
    return {
        "W_chiller_kW": W_ch, "W_fan_kW": W_fan, "W_pump_kW": W_pump,
        "W_total_kW": W_ch + W_fan + W_pump,
        "water_L": water_L, "Q_reject_kW": Q_rej, "Q_panel_kW": Q_panel, "Q_tower_kW": Q_tower,
        "T_cw_in_C": T_cw_in, "balance_residual_kW": balance_residual_kW, "T_cw_ret_C": T_ret, "T_panel_out_C": T_p_out, "COP": cop, "T_wb_C": T_wb,
    }


def panel_capacity_kW(wx, cfg, T_in_C=35.0):
    """Field heat rejection with full design flow and a 35 C inlet (state feature only)."""
    _, Q = panel_outlet(T_in_C, cfg.m_cw, wx["T_air_C"], wx["L_down"], wx["ghi"], wx["wind"], cfg)
    return Q
