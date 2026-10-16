"""Pydantic Contracts for Computer Vision, Digital Twin, and Facility AI."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CVDetection(BaseModel):
    track_id: int
    class_name: str = "person"
    confidence: Optional[float] = 0.90
    bbox_2d: List[float] = Field(default_factory=list)
    floor_pos_2d: List[float] = Field(default_factory=list)
    metric_pos_2d: Optional[List[float]] = None
    zone_id: Optional[str] = None


class CVFramePayload(BaseModel):
    timestamp: Optional[str] = None
    frame_id: Optional[int] = 0
    room_id: Optional[str] = "live_cam_room"
    occupancy_total: Optional[int] = 0
    detections: List[CVDetection] = Field(default_factory=list)


class RoomMetadata(BaseModel):
    room_id: str = "live_cam_room"
    name: Optional[str] = "Live Webcam Vision Room"
    capacity: int = 12
    dimensions: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"length_m": 6.0, "width_m": 5.0, "height_m": 3.0}
    )


class LightStatus(BaseModel):
    active_fixtures: int
    dimming_level: int
    watts_per_fixture: Optional[float] = 40.0


class ZoneState(BaseModel):
    occupancy: int
    light_status: LightStatus
    hvac_setpoint_c: float
    measured_temp_c: float
    power_draw_w: int
    wasted_power_w: int
    vacant_minutes: int
    relays: Optional[Dict[str, str]] = None


class TwinMetrics(BaseModel):
    total_power_kw: float
    wasted_power_kw: Optional[float] = 0.0
    cumulative_wasted_kwh: float
    carbon_rate_kgco2_per_hr: float
    waste_cost_inr_per_hr: Optional[float] = 0.0


class TwinTelemetry(BaseModel):
    timestamp: str
    room_metadata: RoomMetadata
    zone_states: Dict[str, ZoneState]
    metrics: TwinMetrics


class ActionItem(BaseModel):
    zone_id: str
    component: str  # "LIGHTING" | "HVAC" | "FAN" | "SOCKETS"
    current_value: float
    target_value: float
    projected_power_saving_w: float
    rationale: str
    status: Optional[str] = "pending"


class RecommendationResponse(BaseModel):
    timestamp: str
    room_id: Optional[str] = "live_cam_room"
    summary: str
    projected_total_power_kw: Optional[float] = None
    projected_carbon_rate_kgco2_per_hr: Optional[float] = None
    actions: List[ActionItem] = Field(default_factory=list)
    source: Optional[str] = "Facility AI"


class CreateRoomRequest(BaseModel):
    room_id: str
    name: Optional[str] = None
    length_m: float = 8.0
    width_m: float = 6.0
    height_m: float = 3.0
    target_temp_c: float = 23.0


class SelectRoomRequest(BaseModel):
    room_id: str


class RelayToggleRequest(BaseModel):
    zone_id: str
    relay: str  # "lighting" | "hvac" | "fan" | "sockets"
    state: str  # "OPEN" | "CLOSED"


class ChatRequest(BaseModel):
    question: str
    telemetry: Optional[Dict[str, Any]] = None
