"""Facility AI Copilot: Conversational Building Telemetry Q&A and Actuation Intent Parser.

Strictly branded as 'Facility AI Copilot' with zero vendor mentions.
Capabilities:
1. Natural language telemetry queries (power draw, waste, emissions, occupants).
2. Autonomous equipment actuation commands (switch off lights, HVAC setback, relay overrides).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from twin_engine import hub

load_dotenv()
logger = logging.getLogger("FacilityAICopilot")

try:
    from openai import OpenAI
    HAS_CLIENT = True
except ImportError:
    OpenAI = None
    HAS_CLIENT = False

LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
PRIMARY_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")


def parse_control_command(question: str) -> Optional[Dict[str, Any]]:
    q_low = question.lower()

    # Lighting shutoff
    if ("turn off" in q_low or "switch off" in q_low or "de-energize" in q_low) and ("light" in q_low or "lights" in q_low):
        target_zone = "Zone_B" if "zone b" in q_low else ("Zone_A" if "zone a" in q_low else ("Workstation_Area" if "workstation" in q_low else None))
        if target_zone:
            hub.set_control(target_zone, "LIGHTING", 0.0, source="AI:Copilot", note="Commanded via AI Copilot")
            return {
                "action": "actuate",
                "message": f"[AI ACTUATION EXECUTED] De-energized lighting circuit in {target_zone}. Power reduced immediately."
            }
        else:
            count = 0
            room = hub.current_room
            for z, s in room["zones"].items():
                if s["occupancy"] == 0:
                    hub.set_control(z, "LIGHTING", 0.0, source="AI:Copilot", note="Vacant lights off")
                    count += 1
            return {
                "action": "actuate",
                "message": f"[AI ACTUATION EXECUTED] De-energized lighting in {count} vacant zones across active space."
            }

    # HVAC setback
    if ("turn off" in q_low or "setback" in q_low or "eco" in q_low) and ("ac" in q_low or "hvac" in q_low or "cooling" in q_low):
        count = 0
        room = hub.current_room
        for z, s in room["zones"].items():
            if s["occupancy"] == 0 and s.get("has_hvac"):
                hub.set_control(z, "HVAC", 28.0, source="AI:Copilot", note="Vacant HVAC setback")
                count += 1
        return {
            "action": "actuate",
            "message": f"[AI ACTUATION EXECUTED] Shifted HVAC in {count} vacant zones to eco setback 28.0°C."
        }

    # Fans
    if ("turn off" in q_low or "kill" in q_low) and ("fan" in q_low or "fans" in q_low):
        room = hub.current_room
        count = 0
        for z in room["zones"]:
            hub.toggle_relay(z, "fan", "OPEN")
            count += 1
        return {
            "action": "actuate",
            "message": f"[AI ACTUATION EXECUTED] De-energized ceiling fans in {count} zones."
        }

    # Mode changes
    if "autonomous" in q_low or "autopilot" in q_low:
        hub.mode = "autonomous"
        hub.comfort_guard()
        return {
            "action": "mode",
            "message": "[AI NOTICE] Autonomous AI Mode ENGAGED. System is self-balancing circuits closed-loop."
        }

    if "manual" in q_low:
        hub.mode = "manual"
        return {
            "action": "mode",
            "message": "[AI NOTICE] Manual Audit Mode ENGAGED. Autonomous overrides paused."
        }

    return None


def ask_facility_copilot(question: str, telemetry: Optional[Dict[str, Any]] = None) -> str:
    tel = telemetry or hub.build_telemetry_dict()
    q_low = question.lower()

    # 1. Actionable command
    cmd = parse_control_command(question)
    if cmd:
        return cmd["message"]

    # 2. Instant telemetry facts
    kpis = hub.snapshot()["kpis"]
    total_occ = kpis["total_occupancy"]
    total_kw = kpis["total_power_kw"]
    wasted_kw = kpis["wasted_power_kw"]
    carbon_rate = kpis["carbon_rate_kgco2_per_hr"]
    avoided_co2 = kpis["co2_avoided_kg"]
    saved_inr = kpis["cost_saved_inr"]
    room_name = hub.current_room["name"]

    if "power" in q_low or "watt" in q_low or "load" in q_low or "draw" in q_low:
        return f"{room_name} active load is {total_kw} kW. Measured waste is {wasted_kw} kW (costing INR {kpis['waste_cost_inr_per_hr']}/hr)."

    if "waste" in q_low or "red" in q_low or "flagged" in q_low:
        return f"Wasted power in {room_name} is {wasted_kw} kW, primarily driven by vacant zones maintaining energized fixtures and cooling loads."

    if "carbon" in q_low or "co2" in q_low or "emission" in q_low:
        return f"Current grid emission rate is {carbon_rate} kg CO2/hr (grid factor: 0.82 kg/kWh). System has avoided {avoided_co2} kg CO2 today (saved INR {saved_inr})."

    if "occupan" in q_low or "people" in q_low or "count" in q_low:
        return f"Real-time vision tracking confirms {total_occ} active occupants in {room_name}."

    # 3. LLM reasoning call (branded strictly as Facility AI Copilot)
    if HAS_CLIENT and LLM_API_KEY:
        try:
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            system_msg = (
                "You are the Facility AI Copilot for a high-efficiency Green Building Digital Twin. "
                "Answer questions strictly grounded in the telemetry JSON. "
                "Keep answers technical, crisp, and concise (2-3 sentences). Never mention any AI vendor or third-party company name."
            )
            resp = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": f"{system_msg}\nTelemetry:\n{json.dumps(tel)}"},
                    {"role": "user", "content": question}
                ],
                timeout=7.0
            )
            return resp.choices[0].message.content or "No response from AI Copilot."
        except Exception as exc:
            logger.warning(f"AI chat fallback: {exc}")

    return f"Facility AI Status: {room_name} operating with {total_occ} occupants at {total_kw} kW active load. Mode: {hub.mode.upper()}."
