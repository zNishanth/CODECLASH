"""Digital Twin Engine & BIM Spatial Entity Graph.

Strict Real-Data Architecture:
- In-memory ROOM_REGISTRY for active spaces
- Live occupancy driven purely by real Computer Vision detections
- Real-time thermodynamic transient physics
- Full hardware relay control matrix
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from physics import (
    AMBIENT_C_DEFAULT,
    CARBON_KG_PER_KWH,
    HVAC_COMFORT_C,
    HVAC_SETBACK_C,
    TARIFF_INR_PER_KWH,
    VACANCY_WASTE_THRESHOLD_S,
    calculate_hvac_power,
    next_simulated_temperature,
    point_in_polygon,
    polygon_area,
)

logger = logging.getLogger("TwinEngine")
IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_ROOM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "live_cam_room": {
        "room_id": "live_cam_room",
        "name": "Live Webcam Vision Room",
        "dim": [6.0, 5.0, 3.0],
        "capacity": 10,
        "target_temp_c": 23.0,
        "zones": {
            "Workstation_Area": {
                "poly": [[0.0, 0.0], [0.65, 0.0], [0.65, 1.0], [0.0, 1.0]],
                "occupancy": 0,
                "dimming_level": 100,
                "setpoint_c": 22.0,
                "measured_temp_c": 23.0,
                "vacant_s": 0.0,
                "fixtures": 6,
                "watts_per_fixture": 35.0,
                "has_hvac": True,
                "relays": {"lighting": "CLOSED", "hvac": "CLOSED", "fan": "CLOSED", "sockets": "CLOSED"},
            },
            "Meeting_Corner": {
                "poly": [[0.65, 0.0], [1.0, 0.0], [1.0, 1.0], [0.65, 1.0]],
                "occupancy": 0,
                "dimming_level": 100,
                "setpoint_c": 23.0,
                "measured_temp_c": 23.5,
                "vacant_s": 0.0,
                "fixtures": 4,
                "watts_per_fixture": 35.0,
                "has_hvac": False,
                "relays": {"lighting": "CLOSED", "hvac": "OPEN", "fan": "CLOSED", "sockets": "CLOSED"},
            }
        },
        "static_assets": {
            "desks": [
                {"id": "desk_cam_1", "polygon": [[1.2, 1.0], [3.2, 1.0], [3.2, 2.0], [1.2, 2.0]], "elevation": 0.75},
                {"id": "desk_cam_2", "polygon": [[1.2, 3.0], [3.2, 3.0], [3.2, 4.0], [1.2, 4.0]], "elevation": 0.75},
            ],
            "laptops": [
                {"id": "lap_cam_1", "coordinates": [2.0, 1.5, 0.8]},
                {"id": "lap_cam_2", "coordinates": [2.0, 3.5, 0.8]},
            ],
            "fans": [
                {"id": "fan_cam_1", "coordinates": [2.2, 2.5, 2.85], "state": "ON"},
                {"id": "fan_cam_2", "coordinates": [4.8, 2.5, 2.85], "state": "ON"},
            ],
            "luminaires": [
                {"id": "lum_cam_1", "coordinates": [1.8, 1.5, 2.95], "zone": "Workstation_Area", "state": "ON"},
                {"id": "lum_cam_2", "coordinates": [1.8, 3.5, 2.95], "zone": "Workstation_Area", "state": "ON"},
                {"id": "lum_cam_3", "coordinates": [4.5, 2.5, 2.95], "zone": "Meeting_Corner", "state": "ON"},
            ],
            "hvac_vents": [
                {"id": "vent_cam_1", "polygon": [[2.0, 2.2], [2.8, 2.2], [2.8, 2.8], [2.0, 2.8]], "elevation": 2.9},
            ]
        }
    },
    "Lab_302": {
        "room_id": "Lab_302",
        "name": "Industrial Lab 302",
        "dim": [15.0, 10.0, 3.0],
        "capacity": 40,
        "target_temp_c": 22.0,
        "zones": {
            "Zone_A": {
                "poly": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.78], [0.0, 0.78]],
                "occupancy": 0,
                "dimming_level": 100,
                "setpoint_c": 22.0,
                "measured_temp_c": 22.5,
                "vacant_s": 0.0,
                "fixtures": 10,
                "watts_per_fixture": 40.0,
                "has_hvac": True,
                "relays": {"lighting": "CLOSED", "hvac": "CLOSED", "fan": "CLOSED", "sockets": "CLOSED"},
            },
            "Zone_B": {
                "poly": [[0.5, 0.0], [1.0, 0.0], [1.0, 0.78], [0.5, 0.78]],
                "occupancy": 0,
                "dimming_level": 100,
                "setpoint_c": 22.0,
                "measured_temp_c": 23.5,
                "vacant_s": 240.0,  # 4 minutes vacant
                "fixtures": 10,
                "watts_per_fixture": 40.0,
                "has_hvac": True,
                "relays": {"lighting": "CLOSED", "hvac": "CLOSED", "fan": "CLOSED", "sockets": "CLOSED"},
            },
            "Corridor": {
                "poly": [[0.0, 0.78], [1.0, 0.78], [1.0, 1.0], [0.0, 1.0]],
                "occupancy": 0,
                "dimming_level": 50,
                "setpoint_c": 26.0,
                "measured_temp_c": 25.5,
                "vacant_s": 120.0,
                "fixtures": 4,
                "watts_per_fixture": 40.0,
                "has_hvac": False,
                "relays": {"lighting": "CLOSED", "hvac": "OPEN", "fan": "OPEN", "sockets": "OPEN"},
            }
        },
        "static_assets": {
            "desks": [
                {"id": "table_1", "polygon": [[2.0, 2.0], [5.0, 2.0], [5.0, 3.2], [2.0, 3.2]], "elevation": 0.75},
                {"id": "table_2", "polygon": [[2.0, 5.0], [5.0, 5.0], [5.0, 6.2], [2.0, 6.2]], "elevation": 0.75},
                {"id": "table_3", "polygon": [[8.0, 2.0], [11.0, 2.0], [11.0, 3.2], [8.0, 3.2]], "elevation": 0.75},
                {"id": "table_4", "polygon": [[8.0, 5.0], [11.0, 5.0], [11.0, 6.2], [8.0, 6.2]], "elevation": 0.75},
            ],
            "laptops": [
                {"id": "lap_1", "coordinates": [3.5, 2.6, 0.8]},
                {"id": "lap_2", "coordinates": [3.5, 5.6, 0.8]},
                {"id": "lap_3", "coordinates": [9.5, 2.6, 0.8]},
                {"id": "lap_4", "coordinates": [9.5, 5.6, 0.8]},
            ],
            "diffusers": [
                {"id": "diffuser_A1", "coordinates": [3.5, 3.2, 2.95], "zone": "Zone_A", "state": "ON"},
                {"id": "diffuser_B1", "coordinates": [10.0, 3.2, 2.95], "zone": "Zone_B", "state": "ON"},
            ],
            "luminaires": [
                {"id": "lum_A1", "coordinates": [2.0, 2.0, 2.95], "zone": "Zone_A", "state": "ON"},
                {"id": "lum_A2", "coordinates": [4.0, 2.0, 2.95], "zone": "Zone_A", "state": "ON"},
                {"id": "lum_A3", "coordinates": [2.0, 4.5, 2.95], "zone": "Zone_A", "state": "ON"},
                {"id": "lum_A4", "coordinates": [4.0, 4.5, 2.95], "zone": "Zone_A", "state": "ON"},
                {"id": "lum_B1", "coordinates": [9.0, 2.0, 2.95], "zone": "Zone_B", "state": "ON"},
                {"id": "lum_B2", "coordinates": [11.0, 2.0, 2.95], "zone": "Zone_B", "state": "ON"},
                {"id": "lum_B3", "coordinates": [9.0, 4.5, 2.95], "zone": "Zone_B", "state": "ON"},
                {"id": "lum_B4", "coordinates": [11.0, 4.5, 2.95], "zone": "Zone_B", "state": "ON"},
                {"id": "lum_C1", "coordinates": [3.5, 8.5, 2.95], "zone": "Corridor", "state": "ON"},
                {"id": "lum_C2", "coordinates": [7.5, 8.5, 2.95], "zone": "Corridor", "state": "ON"},
            ],
            "hvac_vents": [
                {"id": "vent_A", "polygon": [[3.0, 4.0], [4.0, 4.0], [4.0, 4.8], [3.0, 4.8]], "elevation": 2.9},
                {"id": "vent_B", "polygon": [[9.0, 4.0], [10.0, 4.0], [10.0, 4.8], [9.0, 4.8]], "elevation": 2.9},
            ]
        }
    }
}


class DigitalTwinEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.room_registry: Dict[str, Dict[str, Any]] = copy.deepcopy(DEFAULT_ROOM_REGISTRY)
        self.active_room_id: str = "live_cam_room"

        self.mode = "manual"
        self.ambient_c = AMBIENT_C_DEFAULT
        self.time_scale = 10.0
        self.sim_now = datetime.now(IST).replace(microsecond=0)
        self._last_tick = time.time()
        self._last_hist = 0.0

        self.cum = {
            "wasted_kwh": 0.0,
            "total_kwh": 0.0,
            "baseline_kwh": 0.0,
            "saved_kwh": 0.0
        }

        # Real CV state
        self.cv = {
            "frames": 0,
            "frame_id": 0,
            "last_rx": 0.0,
            "timestamp": None,
            "occupancy_total": 0,
            "persons": [],
            "source": "cv_stream"
        }

        self.history = deque(maxlen=300)
        self.actuation_log = deque(maxlen=100)
        self.latest_recommendation = None
        self.agent_status = {
            "source": "none",
            "error": None,
            "latency_ms": None,
            "calls": 0,
            "last_call": 0.0
        }

        self.calc_cache: Dict[str, Dict[str, Any]] = {}
        self._recompute_active_physics()

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TwinEngineLoop")
        self._thread.start()

    @property
    def current_room(self) -> Dict[str, Any]:
        return self.room_registry.get(self.active_room_id, self.room_registry["live_cam_room"])

    def switch_active_room(self, room_id: str) -> bool:
        with self.lock:
            if room_id in self.room_registry:
                self.active_room_id = room_id
                self._recompute_active_physics()
                logger.info(f"Switched active 3D room to: {room_id}")
                return True
            return False

    def create_or_update_room(
        self,
        room_id: str,
        name: str,
        length_m: float,
        width_m: float,
        height_m: float,
        target_temp_c: float = 23.0
    ) -> Dict[str, Any]:
        with self.lock:
            l = float(length_m)
            w = float(width_m)
            h = float(height_m)
            
            zones = {
                "Zone_Main": {
                    "poly": [[0.0, 0.0], [0.7, 0.0], [0.7, 1.0], [0.0, 1.0]],
                    "occupancy": 0,
                    "dimming_level": 100,
                    "setpoint_c": float(target_temp_c),
                    "measured_temp_c": float(target_temp_c) + 0.5,
                    "vacant_s": 0.0,
                    "fixtures": max(4, int((l * w) / 12)),
                    "watts_per_fixture": 40.0,
                    "has_hvac": True,
                    "relays": {"lighting": "CLOSED", "hvac": "CLOSED", "fan": "CLOSED", "sockets": "CLOSED"},
                },
                "Zone_Perimeter": {
                    "poly": [[0.7, 0.0], [1.0, 0.0], [1.0, 1.0], [0.7, 1.0]],
                    "occupancy": 0,
                    "dimming_level": 80,
                    "setpoint_c": float(target_temp_c) + 1.0,
                    "measured_temp_c": float(target_temp_c) + 1.2,
                    "vacant_s": 0.0,
                    "fixtures": max(2, int((l * w) / 25)),
                    "watts_per_fixture": 40.0,
                    "has_hvac": False,
                    "relays": {"lighting": "CLOSED", "hvac": "OPEN", "fan": "CLOSED", "sockets": "CLOSED"},
                }
            }

            static_assets = {
                "desks": [
                    {"id": f"desk_{room_id}_1", "polygon": [[l * 0.2, w * 0.25], [l * 0.45, w * 0.25], [l * 0.45, w * 0.45], [l * 0.2, w * 0.45]], "elevation": 0.75},
                    {"id": f"desk_{room_id}_2", "polygon": [[l * 0.2, w * 0.55], [l * 0.45, w * 0.55], [l * 0.45, w * 0.75], [l * 0.2, w * 0.75]], "elevation": 0.75},
                ],
                "laptops": [
                    {"id": f"lap_{room_id}_1", "coordinates": [l * 0.32, w * 0.35, 0.8]},
                    {"id": f"lap_{room_id}_2", "coordinates": [l * 0.32, w * 0.65, 0.8]},
                ],
                "fans": [
                    {"id": f"fan_{room_id}_1", "coordinates": [l * 0.35, w * 0.5, h - 0.15], "state": "ON"},
                    {"id": f"fan_{room_id}_2", "coordinates": [l * 0.85, w * 0.5, h - 0.15], "state": "ON"},
                ],
                "luminaires": [
                    {"id": f"lum_{room_id}_1", "coordinates": [l * 0.25, w * 0.35, h - 0.05], "zone": "Zone_Main", "state": "ON"},
                    {"id": f"lum_{room_id}_2", "coordinates": [l * 0.25, w * 0.65, h - 0.05], "zone": "Zone_Main", "state": "ON"},
                    {"id": f"lum_{room_id}_3", "coordinates": [l * 0.85, w * 0.50, h - 0.05], "zone": "Zone_Perimeter", "state": "ON"},
                ],
                "hvac_vents": [
                    {"id": f"vent_{room_id}_1", "polygon": [[l * 0.3, w * 0.45], [l * 0.4, w * 0.45], [l * 0.4, w * 0.55], [l * 0.3, w * 0.55]], "elevation": h - 0.1},
                ]
            }

            self.room_registry[room_id] = {
                "room_id": room_id,
                "name": name or room_id,
                "dim": [l, w, h],
                "capacity": max(6, int((l * w) / 4)),
                "target_temp_c": float(target_temp_c),
                "zones": zones,
                "static_assets": static_assets,
            }

            self.active_room_id = room_id
            self._recompute_active_physics()
            return {
                "status": "created",
                "room_id": room_id,
                "name": name,
                "dimensions": [l, w, h],
                "zones": list(zones.keys())
            }

    def _calc_zone_physics(self, room: dict, zid: str) -> Dict[str, Any]:
        z = room["zones"][zid]
        l, w, _ = room["dim"]
        floor_area_m2 = l * w * polygon_area(z["poly"])
        
        # 1. Lighting
        light_relay = z["relays"].get("lighting", "CLOSED") == "CLOSED"
        dim = z["dimming_level"] if light_relay else 0.0
        light_w = z["fixtures"] * z["watts_per_fixture"] * (dim / 100.0)

        # 2. Fan
        fan_relay = z["relays"].get("fan", "CLOSED") == "CLOSED"
        fan_w = 65.0 if fan_relay else 0.0

        # 3. Sockets
        sockets_relay = z["relays"].get("sockets", "CLOSED") == "CLOSED"
        sockets_w = (65.0 * max(1, z["occupancy"])) if sockets_relay else 0.0

        # 4. HVAC (Q = sensible + envelope)
        hvac_relay = z["relays"].get("hvac", "CLOSED") == "CLOSED"
        hvac_w = 0.0
        if z["has_hvac"] and hvac_relay:
            hvac_w = calculate_hvac_power(
                has_hvac=True,
                occupants=z["occupancy"],
                setpoint_c=z["setpoint_c"],
                ambient_c=self.ambient_c,
                floor_area_m2=floor_area_m2
            )

        total_w = light_w + fan_w + sockets_w + hvac_w

        # Real-time waste: devices energized in vacant zones (> 3 minutes)
        wasted_w = 0.0
        if z["occupancy"] == 0:
            if z["vacant_s"] >= VACANCY_WASTE_THRESHOLD_S:
                wasted_w = total_w
            else:
                wasted_w = light_w + fan_w

        # Zone Status
        if z["occupancy"] == 0:
            if wasted_w >= 250.0 and z["vacant_s"] >= VACANCY_WASTE_THRESHOLD_S:
                status = "RED"
                reasons = [f"[CRITICAL WASTE] Vacant {z['vacant_s']/60:.1f}m drawing {wasted_w:.0f} W unmanaged power"]
            elif wasted_w > 40.0:
                status = "YELLOW"
                reasons = [f"[VACANCY GRACE] Vacant {z['vacant_s']/60:.1f}m"]
            else:
                status = "GREEN"
                reasons = ["[ECO IDLE] Circuits powered down"]
        else:
            status = "GREEN"
            reasons = ["[OPTIMAL] Occupied, comfort active"]

        return {
            "lighting_w": round(light_w, 1),
            "fan_w": round(fan_w, 1),
            "sockets_w": round(sockets_w, 1),
            "hvac_w": round(hvac_w, 1),
            "power_draw_w": round(total_w, 1),
            "wasted_power_w": round(wasted_w, 1),
            "status": status,
            "reasons": reasons,
            "area_m2": round(floor_area_m2, 1)
        }

    def _recompute_active_physics(self):
        room = self.current_room
        calc = {}
        for zid in room["zones"]:
            calc[zid] = self._calc_zone_physics(room, zid)
        self.calc_cache = calc

    def tick(self, now: Optional[float] = None):
        with self.lock:
            now = now or time.time()
            dt_real = min(max(now - self._last_tick, 0.0), 3.0)
            self._last_tick = now
            dt_sim = dt_real * self.time_scale

            room = self.current_room
            if dt_sim > 0:
                for zid, z in room["zones"].items():
                    c = self.calc_cache.get(zid, {})
                    h = dt_sim / 3600.0 / 1000.0
                    pow_w = c.get("power_draw_w", 0.0)
                    wst_w = c.get("wasted_power_w", 0.0)
                    base_w = pow_w + 250.0

                    self.cum["total_kwh"] += pow_w * h
                    self.cum["wasted_kwh"] += wst_w * h
                    self.cum["baseline_kwh"] += base_w * h
                    self.cum["saved_kwh"] += max(0.0, (base_w - pow_w)) * h

                    z["vacant_s"] = (z["vacant_s"] + dt_sim) if z["occupancy"] == 0 else 0.0
                    z["measured_temp_c"] = next_simulated_temperature(
                        has_hvac=z["has_hvac"],
                        current_temp=z["measured_temp_c"],
                        setpoint_c=z["setpoint_c"],
                        occupants=z["occupancy"],
                        ambient_c=self.ambient_c,
                        dt_minutes=dt_sim / 60.0
                    )

                self.sim_now += timedelta(seconds=dt_sim)

            self._recompute_active_physics()

            if now - self._last_hist >= 1.0:
                self._last_hist = now
                tot = sum(c.get("power_draw_w", 0.0) for c in self.calc_cache.values())
                wst = sum(c.get("wasted_power_w", 0.0) for c in self.calc_cache.values())
                self.history.append({
                    "t": self.sim_now.strftime("%H:%M:%S"),
                    "total_kw": round(tot / 1000.0, 3),
                    "wasted_kw": round(wst / 1000.0, 3),
                })

    def _run_loop(self):
        while self._running:
            try:
                time.sleep(1.0)
                self.tick()
                if self.mode == "autonomous":
                    self.comfort_guard()
            except Exception as e:
                logger.error(f"Error in twin loop: {e}")

    def comfort_guard(self):
        room = self.current_room
        for zid, z in room["zones"].items():
            if z["occupancy"] > 0:
                if z["dimming_level"] < 70 or z["relays"]["lighting"] == "OPEN":
                    z["relays"]["lighting"] = "CLOSED"
                    z["dimming_level"] = 100
                if z["has_hvac"] and z["setpoint_c"] > 25.0:
                    z["setpoint_c"] = HVAC_COMFORT_C

    def set_control(self, zone_id: str, component: str, value: float, source: str = "manual", note: str = "") -> bool:
        with self.lock:
            room = self.current_room
            if zone_id not in room["zones"]:
                return False
            z = room["zones"][zone_id]
            comp = component.upper()

            if comp == "LIGHTING":
                val = int(min(max(round(value), 0), 100))
                z["dimming_level"] = val
                z["relays"]["lighting"] = "OPEN" if val == 0 else "CLOSED"
            elif comp == "HVAC":
                if not z["has_hvac"]:
                    return False
                z["setpoint_c"] = round(min(max(float(value), 16.0), 30.0), 1)
            elif comp == "FAN":
                state = "OPEN" if value == 0 else "CLOSED"
                z["relays"]["fan"] = state
            elif comp == "SOCKETS":
                state = "OPEN" if value == 0 else "CLOSED"
                z["relays"]["sockets"] = state
            else:
                return False

            self.actuation_log.appendleft({
                "t": self.sim_now.strftime("%H:%M:%S"),
                "zone": zone_id,
                "component": comp,
                "to": value,
                "source": source,
                "note": note
            })
            self._recompute_active_physics()
            return True

    def toggle_relay(self, zone_id: str, relay_name: str, state: str) -> bool:
        with self.lock:
            room = self.current_room
            if zone_id in room["zones"]:
                z = room["zones"][zone_id]
                key = relay_name.lower()
                if key == "fan" and "diffuser" in z["relays"] and "fan" not in z["relays"]:
                    key = "diffuser"
                elif key == "diffuser" and "fan" in z["relays"] and "diffuser" not in z["relays"]:
                    key = "fan"

                if key in z["relays"]:
                    z["relays"][key] = state
                    if key == "lighting":
                        z["dimming_level"] = 0 if state == "OPEN" else 100
                    self._recompute_active_physics()
                    return True
            return False

    def ingest_cv_payload(self, payload: dict, source: str = "webcam") -> dict:
        """Ingests live computer vision detections and maps them into room metric space."""
        with self.lock:
            self.tick()
            dets = payload.get("detections", [])
            target_room = self.room_registry.get("live_cam_room") if source == "webcam" else self.current_room
            l, w, _ = target_room["dim"]

            persons = []
            zone_counts = {zid: 0 for zid in target_room["zones"]}

            for idx, d in enumerate(dets):
                if not isinstance(d, dict):
                    continue
                cls_name = str(d.get("class_name", "person")).lower()
                floor_pos = d.get("floor_pos_2d") or [0.5, 0.5]
                fx, fy = float(floor_pos[0]), float(floor_pos[1])

                metric_x = round(fx * l, 2)
                metric_y = round(fy * w, 2)

                if cls_name == "person":
                    matched_z = None
                    for zid, zdata in target_room["zones"].items():
                        if point_in_polygon(fx, fy, zdata["poly"]):
                            matched_z = zid
                            break
                    matched_z = matched_z or list(target_room["zones"].keys())[0]
                    zone_counts[matched_z] = zone_counts.get(matched_z, 0) + 1

                    persons.append({
                        "track_id": int(d.get("track_id", idx + 1)),
                        "class_name": "person",
                        "confidence": d.get("confidence", 0.92),
                        "floor_pos_2d": [fx, fy],
                        "metric_pos_2d": [metric_x, metric_y],
                        "zone_id": matched_z,
                        "bbox_2d": d.get("bbox_2d", [0, 0, 0.1, 0.1])
                    })

            for zid, cnt in zone_counts.items():
                if zid in target_room["zones"]:
                    target_room["zones"][zid]["occupancy"] = cnt

            self.cv.update({
                "frames": self.cv["frames"] + 1,
                "frame_id": payload.get("frame_id", self.cv["frame_id"] + 1),
                "last_rx": time.time(),
                "timestamp": payload.get("timestamp", self.sim_now.isoformat()),
                "occupancy_total": len(persons),
                "persons": persons,
                "source": source
            })
            self._recompute_active_physics()

            if self.mode == "autonomous":
                self.comfort_guard()

            return {"accepted": True, "occupancy_total": len(persons), "zones": zone_counts}

    def build_telemetry_dict(self) -> Dict[str, Any]:
        with self.lock:
            room = self.current_room
            zs = {}
            for zid, z in room["zones"].items():
                c = self.calc_cache.get(zid, {})
                zs[zid] = {
                    "occupancy": int(z["occupancy"]),
                    "light_status": {
                        "active_fixtures": z["fixtures"] if z["relays"]["lighting"] == "CLOSED" else 0,
                        "dimming_level": int(z["dimming_level"]),
                        "watts_per_fixture": z["watts_per_fixture"]
                    },
                    "hvac_setpoint_c": z["setpoint_c"],
                    "measured_temp_c": round(z["measured_temp_c"], 1),
                    "power_draw_w": int(round(c.get("power_draw_w", 0.0))),
                    "wasted_power_w": int(round(c.get("wasted_power_w", 0.0))),
                    "vacant_minutes": int(z["vacant_s"] // 60),
                    "relays": z["relays"]
                }

            kw = sum(c.get("power_draw_w", 0.0) for c in self.calc_cache.values()) / 1000.0
            wst_kw = sum(c.get("wasted_power_w", 0.0) for c in self.calc_cache.values()) / 1000.0
            return {
                "timestamp": self.sim_now.isoformat(timespec="seconds"),
                "room_metadata": {
                    "room_id": room["room_id"],
                    "name": room["name"],
                    "capacity": room["capacity"],
                    "dimensions": {
                        "length_m": room["dim"][0],
                        "width_m": room["dim"][1],
                        "height_m": room["dim"][2]
                    }
                },
                "zone_states": zs,
                "metrics": {
                    "total_power_kw": round(kw, 2),
                    "wasted_power_kw": round(wst_kw, 3),
                    "cumulative_wasted_kwh": round(self.cum["wasted_kwh"], 2),
                    "carbon_rate_kgco2_per_hr": round(kw * CARBON_KG_PER_KWH, 2),
                    "waste_cost_inr_per_hr": round(wst_kw * TARIFF_INR_PER_KWH, 2)
                }
            }

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            self.tick()
            room = self.current_room
            zones_data = {}
            for zid, z in room["zones"].items():
                c = self.calc_cache.get(zid, {})
                zones_data[zid] = {
                    **z,
                    **c,
                    "vacant_minutes": round(z["vacant_s"] / 60.0, 1),
                }

            tot_w = sum(c.get("power_draw_w", 0.0) for c in self.calc_cache.values())
            wst_w = sum(c.get("wasted_power_w", 0.0) for c in self.calc_cache.values())
            total_occ = sum(z["occupancy"] for z in room["zones"].values())

            return copy.deepcopy({
                "room_id": room["room_id"],
                "room_name": room["name"],
                "room_dim": room["dim"],
                "room_capacity": room["capacity"],
                "rooms_list": [{"id": k, "name": v["name"], "dim": v["dim"]} for k, v in self.room_registry.items()],
                "mode": self.mode,
                "sim_time": self.sim_now.isoformat(timespec="seconds"),
                "ambient_c": self.ambient_c,
                "time_scale": self.time_scale,
                "zones": zones_data,
                "kpis": {
                    "total_occupancy": total_occ,
                    "total_power_w": round(tot_w),
                    "total_power_kw": round(tot_w / 1000.0, 2),
                    "wasted_power_w": round(wst_w),
                    "wasted_power_kw": round(wst_w / 1000.0, 3),
                    "waste_cost_inr_per_hr": round(wst_w / 1000.0 * TARIFF_INR_PER_KWH, 2),
                    "carbon_rate_kgco2_per_hr": round(tot_w / 1000.0 * CARBON_KG_PER_KWH, 3),
                    "cumulative_wasted_kwh": round(self.cum["wasted_kwh"], 3),
                    "cumulative_saved_kwh": round(self.cum["saved_kwh"], 3),
                    "co2_avoided_kg": round(self.cum["saved_kwh"] * CARBON_KG_PER_KWH, 3),
                    "cost_saved_inr": round(self.cum["saved_kwh"] * TARIFF_INR_PER_KWH, 2),
                },
                "cv": self.cv,
                "history": list(self.history),
                "actuation_log": list(self.actuation_log),
                "recommendation": self.latest_recommendation,
                "agent_status": self.agent_status,
                "static_assets": room["static_assets"],
            })


hub = DigitalTwinEngine()
