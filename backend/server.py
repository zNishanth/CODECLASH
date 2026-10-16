"""FastAPI Master Backend Server: Green Building Digital Twin & CV Telemetry Hub.

Core Architectures:
1. Lifespan Management:
   - Initializes YOLOv8 model & homography calibration once on startup.
   - Launches async background telemetry broadcast task.
   - Cleanly cancels tasks, closes WebSockets, and releases OpenCV resources upon shutdown.
2. Async ThreadPool Offloading:
   - Offloads frame decoding, YOLO inference, and matrix calculations to `cv_engine.executor`
     via `asyncio.get_running_loop().run_in_executor()` so the async event loop is never blocked.
3. WebSockets:
   - `/ws/cv/stream`: Ingests binary JPEG frames from browser webcams with adaptive FPS backpressure control.
   - `/ws/telemetry`: Real-time bidirectional telemetry hub broadcasting room states, occupancy, and relays,
     with dynamic room-switching and thread-safe connection pooling.
4. Resilient Fallbacks:
   - `/api/cv/video_feed` & `/api/cv/stream`: MJPEG multipart stream.
   - `/api/cv/frame`: HTTP POST JPEG frame ingestion.
5. Strict Real Data & Strict AI Naming: Only genuine detections, zero third-party vendor names.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from copilot import ask_facility_copilot
from cv_engine import cv_engine
from genai_service import get_optimization_recommendation
from schemas import CreateRoomRequest, RelayToggleRequest, SelectRoomRequest, ChatRequest
from twin_engine import hub

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GreenTwinServer")
IST = timezone(timedelta(hours=5, minutes=30))

API_PORT = int(os.getenv("PORT", "8000"))
API_HOST = os.getenv("HOST", "0.0.0.0")


# =========================================================================
# WEBSOCKET CONNECTION POOLING MANAGER
# =========================================================================
class TelemetryConnectionManager:
    """Thread-safe WebSocket connection pool with dead-socket cleanup."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        """Broadcasts JSON payload to all active clients, automatically pruning disconnected sockets."""
        async with self._lock:
            clients = list(self.active_connections)

        if not clients:
            return

        dead_clients = []
        payload_str = json.dumps(data)

        for ws in clients:
            try:
                await ws.send_text(payload_str)
            except Exception:
                dead_clients.append(ws)

        if dead_clients:
            async with self._lock:
                for dead in dead_clients:
                    self.active_connections.discard(dead)


ws_manager = TelemetryConnectionManager()
telemetry_broadcast_task: Optional[asyncio.Task] = None


async def telemetry_broadcast_worker():
    """Background async worker broadcasting digital twin telemetry every 600ms."""
    logger.info("Starting background WebSocket telemetry broadcaster...")
    try:
        while True:
            await asyncio.sleep(0.6)
            if ws_manager.active_connections:
                state_snapshot = hub.snapshot()
                await ws_manager.broadcast({
                    "type": "telemetry_update",
                    "timestamp": datetime.now(IST).isoformat(),
                    "room_id": state_snapshot["room_id"],
                    "state": state_snapshot
                })
    except asyncio.CancelledError:
        logger.info("Telemetry broadcast worker cancelled cleanly.")
    except Exception as e:
        logger.error(f"Error in telemetry broadcast worker: {e}")


# =========================================================================
# LIFESPAN CONTEXT MANAGER
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes YOLO, calibration, and background tasks on startup, and releases handles on shutdown."""
    global telemetry_broadcast_task
    logger.info("Starting Green Building Digital Twin Master Server...")

    # Start background broadcaster
    telemetry_broadcast_task = asyncio.create_task(telemetry_broadcast_worker())

    yield

    logger.info("Initiating server shutdown sequence...")
    # 1. Cancel background worker
    if telemetry_broadcast_task and not telemetry_broadcast_task.done():
        telemetry_broadcast_task.cancel()
        try:
            await telemetry_broadcast_task
        except asyncio.CancelledError:
            pass

    # 2. Release camera & thread pool
    cv_engine.shutdown()
    logger.info("Shutdown sequence completed.")


app = FastAPI(
    title="Green Building Digital Twin Core",
    description="Production-grade Digital Twin with YOLOv8 Computer Vision, 3D WebGL BIM & Facility AI",
    version="6.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# =========================================================================
# STATIC FILE & MODULE DELIVERY
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Digital Twin Frontend Loading...</h2>")


@app.get("/DigitalTwinScene.js")
async def serve_digital_twin_scene():
    scene_path = os.path.join(BASE_DIR, "DigitalTwinScene.js")
    if os.path.exists(scene_path):
        return FileResponse(scene_path, media_type="application/javascript")
    raise HTTPException(404, "DigitalTwinScene.js not found")


# =========================================================================
# REAL-TIME WEBSOCKET ENDPOINTS
# =========================================================================
@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """WebSocket endpoint to broadcast live digital twin telemetry and receive control commands."""
    await ws_manager.connect(websocket)
    # Immediately send active room state upon connection
    initial_state = hub.snapshot()
    await websocket.send_text(json.dumps({
        "type": "initial_state",
        "room_id": initial_state["room_id"],
        "state": initial_state
    }))

    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                msg = json.loads(data_text)
                msg_type = msg.get("type", "")

                if msg_type == "select_room":
                    room_id = msg.get("room_id", "")
                    if room_id in hub.room_registry:
                        hub.switch_active_room(room_id)
                        snap = hub.snapshot()
                        await ws_manager.broadcast({
                            "type": "room_switched",
                            "room_id": room_id,
                            "state": snap
                        })

                elif msg_type == "toggle_relay":
                    zone_id = msg.get("zone_id", "")
                    relay = msg.get("relay", "")
                    state = msg.get("state", "CLOSED")
                    hub.toggle_relay(zone_id, relay, state)
                    snap = hub.snapshot()
                    await ws_manager.broadcast({
                        "type": "telemetry_update",
                        "room_id": snap["room_id"],
                        "state": snap
                    })

                elif msg_type == "set_mode":
                    mode = msg.get("mode", "manual")
                    if mode in ("manual", "autonomous"):
                        hub.mode = mode
                        if mode == "autonomous":
                            hub.comfort_guard()
                        snap = hub.snapshot()
                        await ws_manager.broadcast({
                            "type": "mode_changed",
                            "mode": hub.mode,
                            "state": snap
                        })

                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket telemetry exception: {e}")
        await ws_manager.disconnect(websocket)


@app.websocket("/ws/cv/stream")
async def websocket_cv_stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint to ingest browser webcam frames with adaptive FPS backpressure control."""
    await websocket.accept()
    logger.info("Browser CV stream WebSocket connected.")
    loop = asyncio.get_running_loop()

    frame_counter = 0
    is_busy = False

    try:
        while True:
            # Receive binary frame or JSON message
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                image_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    parsed = json.loads(message["text"])
                    if parsed.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                        continue
                    # Check base64 frame payload
                    b64_str = parsed.get("frame") or parsed.get("image")
                    if b64_str:
                        import base64
                        if "," in b64_str:
                            b64_str = b64_str.split(",", 1)[1]
                        image_bytes = base64.b64decode(b64_str)
                    else:
                        continue
                except Exception:
                    continue
            else:
                continue

            # Adaptive Backpressure: Skip frame if previous inference is still executing
            if is_busy:
                continue

            is_busy = True
            frame_counter += 1

            try:
                active_room = hub.current_room
                # Offload heavy frame decode, YOLO inference & homography to ThreadPoolExecutor
                _, detections = await loop.run_in_executor(
                    cv_engine.executor,
                    cv_engine.decode_and_infer,
                    image_bytes,
                    active_room["dim"],
                    active_room.get("zones", {})
                )

                persons = [d for d in detections if d.get("class_name") == "person"]
                payload = {
                    "timestamp": datetime.now(IST).isoformat(),
                    "frame_id": frame_counter,
                    "room_id": active_room["room_id"],
                    "occupancy_total": len(persons),
                    "detections": detections
                }

                # Update in-memory digital twin entity graph & write atomic logs
                hub.ingest_cv_payload(payload, source="webcam_ws")
                cv_engine.write_atomic_output(payload)

                # Return detection result to the streaming client
                await websocket.send_text(json.dumps({
                    "type": "cv_telemetry",
                    "frame_id": frame_counter,
                    "occupancy_total": len(persons),
                    "detections": detections,
                    "fps": cv_engine.fps_estimate,
                    "calibrated": cv_engine.projector.is_calibrated
                }))

                # Broadcast state update to telemetry clients
                state_snapshot = hub.snapshot()
                await ws_manager.broadcast({
                    "type": "telemetry_update",
                    "room_id": state_snapshot["room_id"],
                    "state": state_snapshot
                })

            finally:
                is_busy = False

    except WebSocketDisconnect:
        logger.info("Browser CV stream WebSocket disconnected.")
    except Exception as e:
        logger.warning(f"Error on CV stream WebSocket: {e}")


# =========================================================================
# HTTP COMPUTER VISION INGESTION & FALLBACKS
# =========================================================================
@app.post("/api/cv/frame")
@app.post("/api/telemetry/frame")
async def ingest_cv_frame(request: Request):
    """HTTP endpoint to ingest browser-captured webcam frames or detection JSON."""
    content_type = request.headers.get("content-type", "")
    loop = asyncio.get_running_loop()
    active_room = hub.current_room

    if "image" in content_type or "octet-stream" in content_type:
        image_bytes = await request.body()
        _, detections = await loop.run_in_executor(
            cv_engine.executor,
            cv_engine.decode_and_infer,
            image_bytes,
            active_room["dim"],
            active_room.get("zones", {})
        )
        persons = [d for d in detections if d.get("class_name") == "person"]
        payload = {
            "timestamp": datetime.now(IST).isoformat(),
            "frame_id": hub.cv["frame_id"] + 1,
            "room_id": active_room["room_id"],
            "occupancy_total": len(persons),
            "detections": detections
        }
        res = hub.ingest_cv_payload(payload, source="http_frame")
        cv_engine.write_atomic_output(payload)
        return res
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "Body must be valid JPEG binary or JSON")
        res = hub.ingest_cv_payload(body, source="webcam")
        cv_engine.write_atomic_output(body)
        return res


@app.get("/api/cv/video_feed")
@app.get("/api/cv/stream")
def cv_video_feed():
    """Resilient MJPEG streaming fallback for live debugging and testing."""
    return StreamingResponse(
        cv_engine.generate_mjpeg_stream(hub),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/cv/upload")
async def upload_image_for_detection(file: UploadFile = File(...)):
    """Upload single or multi-room image to detect occupants and project into twin space."""
    content = await file.read()
    active_room = hub.current_room
    loop = asyncio.get_running_loop()

    _, detections = await loop.run_in_executor(
        cv_engine.executor,
        cv_engine.decode_and_infer,
        content,
        active_room["dim"],
        active_room.get("zones", {})
    )

    persons = [d for d in detections if d.get("class_name") == "person"]
    payload = {
        "timestamp": datetime.now(IST).isoformat(),
        "frame_id": hub.cv["frame_id"] + 1,
        "room_id": active_room["room_id"],
        "occupancy_total": len(persons),
        "detections": detections
    }
    hub.ingest_cv_payload(payload, source="uploaded_image")
    cv_engine.write_atomic_output(payload)

    return {
        "filename": file.filename,
        "status": "processed",
        "room_id": active_room["room_id"],
        "occupancy_total": len(persons),
        "detections_count": len(detections),
        "detections": detections
    }


# =========================================================================
# SYSTEM HEALTH & ROOM REGISTRY
# =========================================================================
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Green Building Digital Twin Core",
        "active_room": hub.active_room_id,
        "mode": hub.mode,
        "rooms": list(hub.room_registry.keys()),
        "cv_online": (time.time() - hub.cv["last_rx"]) < 5.0 if hub.cv["last_rx"] else False,
        "homography_calibrated": cv_engine.projector.is_calibrated,
        "ws_connections": len(ws_manager.active_connections)
    }


@app.get("/api/rooms")
def get_rooms():
    return {
        "active_room_id": hub.active_room_id,
        "rooms": [
            {
                "id": k,
                "name": v["name"],
                "dimensions": v["dim"],
                "capacity": v["capacity"],
                "zones": list(v["zones"].keys())
            }
            for k, v in hub.room_registry.items()
        ]
    }


@app.post("/api/rooms/select")
def select_room(req: SelectRoomRequest):
    success = hub.switch_active_room(req.room_id)
    if not success:
        raise HTTPException(404, f"Room {req.room_id} not found")
    return {"status": "ok", "active_room_id": hub.active_room_id}


@app.get("/api/rooms/{room_id}/status")
@app.get("/api/rooms/{room_id}/telemetry")
def get_room_telemetry(room_id: str):
    if room_id in hub.room_registry:
        hub.switch_active_room(room_id)
    return hub.build_telemetry_dict()


@app.post("/api/rooms/create")
@app.post("/api/twin/create-room")
def create_room(req: CreateRoomRequest):
    result = hub.create_or_update_room(
        room_id=req.room_id,
        name=req.name or req.room_id,
        length_m=req.length_m,
        width_m=req.width_m,
        height_m=req.height_m,
        target_temp_c=req.target_temp_c
    )
    return result


# =========================================================================
# HARDWARE RELAYS & ACTUATION
# =========================================================================
@app.post("/api/relays/toggle")
def toggle_relay(req: RelayToggleRequest):
    success = hub.toggle_relay(req.zone_id, req.relay.lower(), req.state.upper())
    if not success:
        raise HTTPException(400, "Failed to toggle relay: invalid zone or relay name")
    return {"status": "ok", "zone_id": req.zone_id, "relay": req.relay, "state": req.state}


@app.post("/api/actuate")
@app.post("/api/agent/actions")
async def actuate_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(422, "Invalid JSON payload")

    applied = 0
    if isinstance(body, list):
        for item in body:
            if isinstance(item, dict) and "zone_id" in item:
                val = item.get("target_value", item.get("value", 0))
                comp = str(item.get("component", "LIGHTING")).upper()
                if hub.set_control(item["zone_id"], comp, val, "api:actuate", item.get("rationale", "")):
                    applied += 1
        return {"status": "ok", "applied": applied, "mode": hub.mode}

    if not isinstance(body, dict):
        raise HTTPException(422, "Expected object or list")

    if "mode" in body:
        m = str(body["mode"]).lower()
        if m in ("manual", "autonomous"):
            hub.mode = m

    if body.get("action") == "apply_all" or body.get("apply_all"):
        if hub.latest_recommendation and "actions" in hub.latest_recommendation:
            for act in hub.latest_recommendation["actions"]:
                if act.get("status") != "applied":
                    if hub.set_control(act["zone_id"], act["component"], act["target_value"], "AI:apply_all", act.get("rationale", "")):
                        act["status"] = "applied"
                        applied += 1

    if "index" in body and isinstance(body["index"], int):
        idx = body["index"]
        if hub.latest_recommendation and idx < len(hub.latest_recommendation.get("actions", [])):
            act = hub.latest_recommendation["actions"][idx]
            if hub.set_control(act["zone_id"], act["component"], act["target_value"], "AI:apply_one", act.get("rationale", "")):
                act["status"] = "applied"
                applied += 1

    if "zone_id" in body and ("value" in body or "target_value" in body):
        val = body.get("target_value", body.get("value"))
        comp = str(body.get("component", "LIGHTING")).upper()
        if hub.set_control(body["zone_id"], comp, val, "manual:direct", body.get("rationale", "")):
            applied += 1

    return {"status": "ok", "applied": applied, "mode": hub.mode}


@app.post("/api/mode")
async def set_mode(request: Request):
    b = await request.json()
    mode = str(b.get("mode", "manual")).lower()
    if mode in ("manual", "autonomous"):
        hub.mode = mode
        if mode == "autonomous":
            hub.comfort_guard()
            if hub.latest_recommendation and "actions" in hub.latest_recommendation:
                for act in hub.latest_recommendation["actions"]:
                    if act.get("status") != "applied":
                        hub.set_control(act["zone_id"], act["component"], act["target_value"], "AI:autonomous", act.get("rationale", ""))
                        act["status"] = "applied"
    return {"mode": hub.mode}


# =========================================================================
# DIGITAL TWIN SPATIAL STATE & TELEMETRY
# =========================================================================
@app.get("/api/twin/state")
@app.get("/api/state")
@app.get("/state")
def get_spatial_state():
    return hub.snapshot()


@app.get("/api/twin/telemetry")
@app.get("/api/telemetry")
@app.get("/telemetry")
def get_telemetry():
    return hub.build_telemetry_dict()


# =========================================================================
# FACILITY AI COPILOT & OPTIMIZATION
# =========================================================================
@app.post("/api/agent/recommend")
def get_recommendation():
    telemetry = hub.build_telemetry_dict()
    recommendation = get_optimization_recommendation(telemetry)
    hub.latest_recommendation = recommendation.dict()
    hub.agent_status.update({
        "source": "Facility AI",
        "calls": hub.agent_status["calls"] + 1,
        "last_call": time.time(),
        "error": None
    })

    if hub.mode == "autonomous" and recommendation.actions:
        for act in recommendation.actions:
            hub.set_control(act.zone_id, act.component, act.target_value, source="AI:autonomous", note=act.rationale[:80])
            act.status = "applied"
        hub.comfort_guard()

    return recommendation


@app.post("/api/agent/chat")
async def chat_endpoint(body: ChatRequest):
    answer = ask_facility_copilot(body.question, body.telemetry)
    return {
        "answer": answer,
        "room_id": hub.active_room_id,
        "mode": hub.mode,
        "sim_time": hub.sim_now.strftime("%H:%M:%S")
    }


@app.post("/api/reset")
def reset_engine():
    hub.__init__()
    return {"status": "reset", "mode": hub.mode}


if __name__ == "__main__":
    logger.info(f"Launching Green Building Digital Twin Server on http://{API_HOST}:{API_PORT}")
    uvicorn.run("server:app", host=API_HOST, port=API_PORT, reload=False)
