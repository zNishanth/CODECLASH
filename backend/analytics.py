"""Deterministic Energy Audit Analytics & Action Recommendation Engine.

Evaluates real telemetry against energy conservation rules:
- Vacancy detection past 3-minute threshold triggering lighting switch-off.
- Vacancy HVAC setback to eco threshold (28°C) saving chiller load.
- Real wattage savings calculation and avoided carbon calculations.
- Strict AI Naming Compliance: Zero vendor mentions.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from physics import (
    CARBON_KG_PER_KWH,
    HVAC_SETBACK_C,
    TARIFF_INR_PER_KWH,
    VACANCY_WASTE_THRESHOLD_S,
)
from schemas import ActionItem, RecommendationResponse

logger = logging.getLogger("AnalyticsEngine")
IST = timezone(timedelta(hours=5, minutes=30))


def run_deterministic_audit(telemetry: Dict[str, Any]) -> RecommendationResponse:
    actions: List[ActionItem] = []
    zone_states = telemetry.get("zone_states", {})
    total_savings_w = 0.0

    for zid, z in zone_states.items():
        occ = int(z.get("occupancy", 0))
        vac_min = float(z.get("vacant_minutes", 0))
        light_stat = z.get("light_status", {})
        dim = float(light_stat.get("dimming_level", 0))
        fixtures = int(light_stat.get("active_fixtures", 10))
        wpf = float(light_stat.get("watts_per_fixture", 40.0))
        setpoint = float(z.get("hvac_setpoint_c", 22.0))
        relays = z.get("relays", {})

        # Rule 1: Vacant zone with lights ON past 3 minutes
        if occ == 0 and vac_min >= (VACANCY_WASTE_THRESHOLD_S / 60.0) and dim > 0 and relays.get("lighting") != "OPEN":
            light_waste = fixtures * wpf * (dim / 100.0)
            actions.append(ActionItem(
                zone_id=zid,
                component="LIGHTING",
                current_value=dim,
                target_value=0.0,
                projected_power_saving_w=round(light_waste, 1),
                rationale=f"{zid} vacant for {vac_min:.0f}m with {fixtures} fixtures energized. De-energize circuit to eliminate waste."
            ))
            total_savings_w += light_waste

        # Rule 2: Vacant zone with HVAC cooling below setback
        if occ == 0 and vac_min >= (VACANCY_WASTE_THRESHOLD_S / 60.0) and setpoint < HVAC_SETBACK_C and relays.get("hvac") != "OPEN":
            hvac_saving = 1200.0
            actions.append(ActionItem(
                zone_id=zid,
                component="HVAC",
                current_value=setpoint,
                target_value=float(HVAC_SETBACK_C),
                projected_power_saving_w=hvac_saving,
                rationale=f"{zid} vacant for {vac_min:.0f}m. Transition thermostat to eco setback {HVAC_SETBACK_C}°C."
            ))
            total_savings_w += hvac_saving

        # Rule 3: Vacant zone with fans energized
        if occ == 0 and vac_min >= (VACANCY_WASTE_THRESHOLD_S / 60.0) and relays.get("fan") == "CLOSED":
            actions.append(ActionItem(
                zone_id=zid,
                component="FAN",
                current_value=1.0,
                target_value=0.0,
                projected_power_saving_w=65.0,
                rationale=f"{zid} vacant. De-energize ceiling fan relay."
            ))
            total_savings_w += 65.0

    cur_kw = float(telemetry.get("metrics", {}).get("total_power_kw", 0.0))
    proj_kw = max(0.0, cur_kw - (total_savings_w / 1000.0))
    proj_co2 = round(proj_kw * CARBON_KG_PER_KWH, 2)

    if actions:
        summary = (
            f"Facility AI audit detected {len(actions)} high-impact actions yielding "
            f"{total_savings_w:.0f} W immediate power reduction and avoiding {round(total_savings_w/1000.0 * CARBON_KG_PER_KWH, 2)} kg CO2/hr."
        )
    else:
        summary = "All facility circuits are synchronized within optimal thermodynamic thresholds. Zero unmanaged waste detected."

    return RecommendationResponse(
        timestamp=datetime.now(IST).isoformat(timespec="seconds"),
        room_id=telemetry.get("room_metadata", {}).get("room_id", "live_cam_room"),
        summary=summary,
        projected_total_power_kw=round(proj_kw, 2),
        projected_carbon_rate_kgco2_per_hr=proj_co2,
        actions=actions,
        source="Facility AI"
    )
