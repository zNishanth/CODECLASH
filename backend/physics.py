"""Digital Twin Building Physics & Thermodynamic Energy Model.

Strict mathematical formulas:
- Sensible heat load from real detected occupants: Q_sensible = N_occupants * 100 W
- Envelope conductive heat gain: Q_envelope = U * A * (T_ambient - T_setpoint)
- Total HVAC Thermal Load: Q_total = Q_sensible + Q_envelope
- Chiller Compressor Power (COP = 3.2) + Ventilation Fan Power (150 W)
- Real-time waste wattage: All energized circuits running in vacant zones (> 3 minutes)
- Grid carbon emission rate: 0.82 kg CO2 / kWh
- Commercial grid tariff: 8.50 INR / kWh
- First-order transient Newton cooling model for room temperature
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

OCCUPANT_SENSIBLE_W = 100.0          # 100 W per occupant
U_VALUE_W_M2K = 1.2                  # Envelope U-value (W/m²·K)
ENVELOPE_TO_FLOOR_RATIO = 2.2        # Effective surface area factor
HVAC_COP = 3.2                       # Chiller COP
HVAC_FAN_W = 150.0                   # Air handling unit fan (W)
HVAC_STANDBY_W = 30.0                # Electronics standby (W)
HVAC_SETBACK_C = 28.0                # Unoccupied setback temperature (°C)
HVAC_COMFORT_C = 24.0                # Comfort temperature (°C)
AMBIENT_C_DEFAULT = 30.0             # Design outdoor ambient temperature (°C)
THERMAL_TAU_MIN = 12.0               # Thermal time constant (minutes)

CARBON_KG_PER_KWH = 0.82             # kg CO2 / kWh
TARIFF_INR_PER_KWH = 8.50            # INR / kWh
VACANCY_WASTE_THRESHOLD_S = 180.0    # 3 minutes vacancy threshold


def point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray-casting algorithm for normalized floor point containment."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / denom + xi:
            inside = not inside
        j = i
    return inside


def polygon_area(poly: list) -> float:
    """Shoelace formula for normalized polygon area."""
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def calculate_envelope_gain(area_m2: float, ambient_c: float, setpoint_c: float) -> float:
    """q_envelope = U * A * (T_ambient - T_setpoint)"""
    delta_t = max(0.0, ambient_c - setpoint_c)
    envelope_area = area_m2 * ENVELOPE_TO_FLOOR_RATIO
    return U_VALUE_W_M2K * envelope_area * delta_t


def calculate_hvac_power(has_hvac: bool, occupants: int, setpoint_c: float, ambient_c: float, floor_area_m2: float) -> float:
    """Computes HVAC power demand using COP=3.2 chiller + fan model."""
    if not has_hvac:
        return 0.0
    if occupants == 0 and setpoint_c >= HVAC_SETBACK_C:
        return HVAC_STANDBY_W

    q_sensible = max(0, occupants) * OCCUPANT_SENSIBLE_W
    q_envelope = calculate_envelope_gain(floor_area_m2, ambient_c, setpoint_c)
    q_total = q_sensible + q_envelope

    compressor_power = q_total / HVAC_COP
    return compressor_power + HVAC_FAN_W


def next_simulated_temperature(
    has_hvac: bool,
    current_temp: float,
    setpoint_c: float,
    occupants: int,
    ambient_c: float,
    dt_minutes: float
) -> float:
    """Newton's cooling law transient model."""
    if not has_hvac:
        target = ambient_c - 1.0 + (0.05 * occupants)
    elif occupants == 0 and setpoint_c >= HVAC_SETBACK_C:
        target = ambient_c - 2.0
    else:
        target = min(setpoint_c, ambient_c)
    
    k = 1.0 - math.exp(-max(dt_minutes, 0.0) / THERMAL_TAU_MIN)
    return current_temp + (target - current_temp) * k
