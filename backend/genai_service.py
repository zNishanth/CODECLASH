"""Facility AI Optimization Strategy Service.

Strictly branded as 'Facility AI Copilot' with zero vendor mentions.
Features:
- Telemetry energy audit reasoning and structured actuation target synthesis.
- 20-second in-memory TTL caching with SHA-256 state hashing to eliminate API rate limits.
- Resilient deterministic fallback engine guaranteeing zero-downtime offline operation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from analytics import run_deterministic_audit
from physics import CARBON_KG_PER_KWH
from schemas import ActionItem, RecommendationResponse

load_dotenv()
logger = logging.getLogger("FacilityAI")
IST = timezone(timedelta(hours=5, minutes=30))

try:
    from openai import OpenAI
    HAS_CLIENT = True
except ImportError:
    OpenAI = None
    HAS_CLIENT = False

LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
PRIMARY_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]

# In-memory TTL Cache: state_hash -> (timestamp, RecommendationResponse)
_RECOMMENDATION_CACHE: Dict[str, Tuple[float, RecommendationResponse]] = {}
CACHE_TTL_SECONDS = 20.0


def compute_telemetry_hash(telemetry: Dict[str, Any]) -> str:
    zs = telemetry.get("zone_states", {})
    parts = []
    for zid in sorted(zs.keys()):
        z = zs[zid]
        occ = z.get("occupancy", 0)
        dim = z.get("light_status", {}).get("dimming_level", 0)
        sp = z.get("hvac_setpoint_c", 22)
        vm = z.get("vacant_minutes", 0)
        parts.append(f"{zid}:{occ}_{dim}_{sp}_{vm}")
    sig = "|".join(parts)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def get_optimization_recommendation(telemetry: Dict[str, Any]) -> RecommendationResponse:
    state_hash = compute_telemetry_hash(telemetry)
    now = time.time()

    # 1. Check TTL cache
    if state_hash in _RECOMMENDATION_CACHE:
        cached_ts, cached_resp = _RECOMMENDATION_CACHE[state_hash]
        if (now - cached_ts) < CACHE_TTL_SECONDS:
            return cached_resp

    # 2. Compute deterministic baseline
    fallback_resp = run_deterministic_audit(telemetry)

    if not HAS_CLIENT or not LLM_API_KEY:
        _RECOMMENDATION_CACHE[state_hash] = (now, fallback_resp)
        return fallback_resp

    # 3. LLM Reasoning Call
    system_prompt = (
        "You are the Facility AI Copilot and Autonomous Building Energy Engineer. "
        "Analyze the provided room telemetry and generate structured actuation targets to eliminate waste while preserving comfort. "
        "Strictly output a JSON object conforming to: "
        '{"summary": string, "actions": [{"zone_id": string, "component": "LIGHTING"|"HVAC"|"FAN", "current_value": float, "target_value": float, "projected_power_saving_w": float, "rationale": string}]}'
    )

    user_prompt = f"Live Telemetry:\n{json.dumps(telemetry, indent=2)}\nProposed Baseline:\n{json.dumps(fallback_resp.dict(), indent=2)}"

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    for model_name in [PRIMARY_MODEL] + FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                timeout=8.0
            )

            raw_text = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_text)

            actions = []
            for item in parsed.get("actions", []):
                actions.append(ActionItem(
                    zone_id=item.get("zone_id", "Zone_B"),
                    component=str(item.get("component", "LIGHTING")).upper(),
                    current_value=float(item.get("current_value", 100)),
                    target_value=float(item.get("target_value", 0)),
                    projected_power_saving_w=float(item.get("projected_power_saving_w", 400)),
                    rationale=str(item.get("rationale", "AI efficiency actuation")),
                    status="pending"
                ))

            if not actions and fallback_resp.actions:
                actions = fallback_resp.actions

            tot_saving_w = sum(a.projected_power_saving_w for a in actions)
            cur_kw = float(telemetry.get("metrics", {}).get("total_power_kw", 0.0))
            proj_kw = max(0.0, cur_kw - (tot_saving_w / 1000.0))

            ai_recommendation = RecommendationResponse(
                timestamp=datetime.now(IST).isoformat(timespec="seconds"),
                room_id=telemetry.get("room_metadata", {}).get("room_id", "live_cam_room"),
                summary=parsed.get("summary", fallback_resp.summary),
                projected_total_power_kw=round(proj_kw, 2),
                projected_carbon_rate_kgco2_per_hr=round(proj_kw * CARBON_KG_PER_KWH, 2),
                actions=actions,
                source="Facility AI"
            )

            _RECOMMENDATION_CACHE[state_hash] = (now, ai_recommendation)
            return ai_recommendation

        except Exception as exc:
            logger.warning(f"AI service retry: {exc}")
            continue

    _RECOMMENDATION_CACHE[state_hash] = (now, fallback_resp)
    return fallback_resp
