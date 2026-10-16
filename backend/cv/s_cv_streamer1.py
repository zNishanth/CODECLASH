#!/usr/bin/env python3
"""
s_cv_streamer.py — S: Computer Vision & Spatial Mapping (ALL-IN-ONE).

Digital twin hackathon, role S. Everything S needs in ONE file:

  pipeline : YOLO person tracking -> bbox bottom-center -> homography floor
             projection -> point-in-polygon zone assignment
             + LIGHT detection (bright fixtures that are ON)
             + FAN detection (fixtures with sustained spinning motion)
             -> contract-valid JSON -> POST {hub}/api/telemetry/frame
             + JSON capture to file (cv_output.jsonl live, cv_output.json on stop)

subcommands
  (default) stream      live webcam/video -> hub     (the production run)
  calibrate             click 4 floor corners -> writes cv_calibration.json
  mock-hub              mini receiver on :8000 to test S alone
  send-sample           POST one fake frame (connectivity + schema check)
  test                  offline contract tests (no camera / no net)

JSON OUTPUT (files)
  every run writes by default:
    cv_output.jsonl     one JSON packet per line, written live (tail -f works)
    cv_output.json      valid JSON array of ALL packets, written when you stop
  --save-json NAME      capture to NAME.jsonl / NAME.json instead
  --no-save-json        write no files
  --log-json            ALSO print each packet to stdout while POSTing
  NOTE: the file is a capture/log artifact (for the team, demos, replay).
        The live S->N handoff is the HTTP POST — never read these files
        from N's live loop.

quickstart
  pip install ultralytics opencv-python numpy requests
  python s_cv_streamer.py test
  python s_cv_streamer.py mock-hub            # terminal 2
  python s_cv_streamer.py send-sample
  python s_cv_streamer.py calibrate           # once per camera position
  python s_cv_streamer.py                     # live! (+ cv_output.json* written)

S -> N contract (exact). class_name may be "person", "light" or "fan":
  {"timestamp": ISO-8601, "frame_id": int, "occupancy_total": int (people only),
   "detections": [{"track_id": int, "class_name": "person"|"light"|"fan",
                   "bbox_2d": [x1,y1,x2,y2] (0..1), "floor_pos_2d": [x,y] (0..1),
                   "zone_id": "Zone_A"}]}
  Semantics: a "light" detection exists ONLY while the light is ON.
             a "fan" detection exists ONLY while the fan is SPINNING.
             light ids start at 10000, fan ids at 20000 (stable per fixture).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # mock-hub / test / schema-only usage still work

log = logging.getLogger("s_cv")

# ══════════════════════ CONFIG — EDIT THIS BLOCK ══════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))

# S -> N connection
HUB_URL = os.getenv("TWIN_HUB_URL", "http://127.0.0.1:8000")  # N's BASE url
TELEMETRY_PATH = "/api/telemetry/frame"                        # per contract
TELEMETRY_URL = HUB_URL.rstrip("/") + TELEMETRY_PATH

# vision source
SOURCE = 0              # 0 = default webcam, or "clips/demo.mp4"
LOOP_VIDEO = True       # replay video files forever (demo-day fallback)
MIRROR_WEBCAM = True
MODEL_WEIGHTS = "yolov8n.pt"   # n = fast on CPU; s/m = more accurate
DEVICE = "cpu"                 # "cpu" | "cuda:0" | "mps"
TRACKER = "bytetrack.yaml"     # or "botsort.yaml" (better, heavier)
CONF_THRESHOLD = 0.35
WARMUP_FRAMES = 10             # no emission while auto-exposure settles

# emission throttling / network resilience
EMIT_EVERY_N_FRAMES = 5        # payload every N processed frames
MIN_EMIT_INTERVAL_S = 0.4      # ...and no faster than this (~2.5 pkt/s)
POST_TIMEOUT_S = 2.0
POST_RETRIES = 2
RETRY_BACKOFF_S = 0.5
QUEUE_MAXSIZE = 8              # slow hub NEVER stalls the camera loop

# classes (add 62 "tv" / 63 "laptop" to also mask screens out of
# light/fan detection — they will then also appear in the payload)
PERSON_CLASS_ID = 0
TRACK_CLASS_IDS = [PERSON_CLASS_ID]

# zone map on the normalized floor plan (0..1) — EDIT to match your room.
# zone_id strings MUST match what N's twin expects (per the contract).
ZONE_DEFAULT = "Unassigned"
ZONES = {
    "Zone_A":   [[0.00, 0.00], [0.45, 0.00], [0.45, 0.52], [0.00, 0.52]],
    "Zone_B":   [[0.55, 0.00], [1.00, 0.00], [1.00, 0.52], [0.55, 0.52]],
    "Corridor": [[0.00, 0.60], [1.00, 0.60], [1.00, 1.00], [0.00, 1.00]],
}

# floor calibration — 'calibrate' subcommand writes cv_calibration.json
CALIBRATION_FILE = os.path.join(_HERE, "cv_calibration.json")
FLOOR_RECT = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]  # floor corners
FLOOR_MODE = "linear"          # auto-upgraded to "homography" when calibrated
CALIB_IMAGE_POINTS = None
if os.path.exists(CALIBRATION_FILE):
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as _f:
            _cal = json.load(_f)
        FLOOR_MODE = _cal.get("mode", "homography")
        CALIB_IMAGE_POINTS = _cal.get("image_points")
    except Exception as _e:
        print(f"[warn] could not read {CALIBRATION_FILE}: {_e}")

# stability
ZONE_SWITCH_CONFIRM_FRAMES = 3   # new zone must win K frames (no flicker)
FLOOR_SMOOTHING_ALPHA = 0.55     # EMA on reported floor_pos_2d (0=frozen,1=raw)
STALE_TRACK_FRAMES = 45          # forget person tracks unseen this long

# fixture detection (lights / fans)
DETECT_LIGHTS = True
DETECT_FANS = True

LIGHT_BRIGHTNESS_THRESH = 232    # 0-255; lower = more sensitive (see --light-thresh)
LIGHT_MIN_AREA_FRAC = 0.00008    # min blob area as fraction of frame area
LIGHT_MAX_AREA_FRAC = 0.15       # reject huge bright regions (windows)
LIGHT_ON_CONFIRM_FRAMES = 3      # frames a light must persist before reported
LIGHT_HOLD_FRAMES = 8            # keep reporting through brief dropouts
LIGHT_ID_BASE = 10000

FAN_DIFF_THRESH = 12             # per-pixel frame-diff threshold (see --fan-motion)
FAN_ENERGY_ALPHA = 0.15          # EMA rate of the sustained-motion energy map
FAN_ENERGY_ON = 60.0             # sustained-energy level that means "spinning"
FAN_MIN_AREA_FRAC = 0.0003
FAN_MAX_AREA_FRAC = 0.25
FAN_ON_CONFIRM_FRAMES = 12       # ~1-2s of continuous spin before reported
FAN_HOLD_FRAMES = 10             # keep reporting through brief dropouts
FAN_ID_BASE = 20000

OCCLUDER_GROW_FRAC = 0.015       # dilate YOLO boxes by this fraction of max dim
FIXTURE_STALE_FRAMES = 900       # forget fixtures unseen this long
FIXTURE_ZONE_FALLBACK_NEAREST = True  # fixture outside all zones -> nearest zone
LIGHT_COLOR = (0, 230, 230)      # BGR, debug overlay
FAN_COLOR = (255, 60, 255)       # BGR, debug overlay

# ── NEW: JSON capture to file ──────────────────────────────────────────
SAVE_JSON_BASE = "cv_output"     # cv_output.jsonl (live) + cv_output.json (on stop)
SAVED_PACKETS_CAP = 20000        # bound memory for the final array file
# ───────────────────────────────────────────────────────────────────────

# debug overlay
SHOW_DEBUG_WINDOW = True
WINDOW_NAME = "S: CV Streamer  (q/Esc = quit)"
ZONE_COLORS = {  # BGR
    "Zone_A": (60, 200, 60), "Zone_B": (80, 150, 255),
    "Corridor": (190, 160, 40), "Unassigned": (128, 128, 128),
}
# ═════════════════════════════════════════════════════════════════════

# ═════════════════ CONTRACT — builders + validator ═════════════════
PERSON = "person"
LIGHT = "light"
FAN = "fan"
_TOP_KEYS = {"timestamp", "frame_id", "occupancy_total", "detections"}
_DET_KEYS = {"track_id", "class_name", "bbox_2d", "floor_pos_2d", "zone_id"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(v) -> float:
    return min(max(float(v), 0.0), 1.0)


def make_detection(track_id, class_name, bbox_2d, floor_pos_2d, zone_id) -> dict:
    """Build a contract-valid detection dict (values clamped to 0..1)."""
    x1, y1, x2, y2 = (float(v) for v in bbox_2d)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return {
        "track_id": int(track_id),
        "class_name": str(class_name),
        "bbox_2d": [_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)],
        "floor_pos_2d": [_clamp01(floor_pos_2d[0]), _clamp01(floor_pos_2d[1])],
        "zone_id": str(zone_id),
    }


def make_payload(frame_id: int, detections: List[dict]) -> dict:
    """Build a contract-valid CV Output Schema packet (people only counted)."""
    return {
        "timestamp": utc_now_iso(),
        "frame_id": int(frame_id),
        "occupancy_total": sum(1 for d in detections if d["class_name"] == PERSON),
        "detections": list(detections),
    }


def validate_payload(p) -> List[str]:
    """Exact contract check. Returns [] when valid — safe to hand to N-side."""
    errors: List[str] = []
    if not isinstance(p, dict):
        return ["payload is not a JSON object"]
    keys = set(p)
    if keys != _TOP_KEYS:
        if _TOP_KEYS - keys:
            errors.append(f"missing keys: {sorted(_TOP_KEYS - keys)}")
        if keys - _TOP_KEYS:
            errors.append(f"unexpected keys: {sorted(keys - _TOP_KEYS)}")

    ts = p.get("timestamp")
    if not isinstance(ts, str):
        errors.append("timestamp must be an ISO-8601 string")
    else:
        try:
            if datetime.fromisoformat(ts).tzinfo is None:
                errors.append("timestamp has no timezone offset")
        except ValueError:
            errors.append("timestamp is not parseable ISO-8601")

    def _int_ok(v):
        return isinstance(v, int) and not isinstance(v, bool)

    fid = p.get("frame_id")
    if not _int_ok(fid) or fid < 0:
        errors.append("frame_id must be a non-negative integer")
    occ = p.get("occupancy_total")
    if not _int_ok(occ) or occ < 0:
        errors.append("occupancy_total must be a non-negative integer")

    dets = p.get("detections")
    if not isinstance(dets, list):
        errors.append("detections must be a list")
        dets = []
    persons = 0
    for i, d in enumerate(dets):
        if not isinstance(d, dict):
            errors.append(f"detections[{i}] is not an object")
            continue
        dk = set(d)
        if dk != _DET_KEYS:
            if _DET_KEYS - dk:
                errors.append(f"detections[{i}] missing {sorted(_DET_KEYS - dk)}")
            if dk - _DET_KEYS:
                errors.append(f"detections[{i}] unexpected {sorted(dk - _DET_KEYS)}")
        tid = d.get("track_id")
        if not _int_ok(tid) or tid < 0:
            errors.append(f"detections[{i}].track_id must be a non-negative int")
        cn = d.get("class_name")
        if not isinstance(cn, str) or not cn:
            errors.append(f"detections[{i}].class_name must be a non-empty string")
        for fname, ln in (("bbox_2d", 4), ("floor_pos_2d", 2)):
            v = d.get(fname)
            ok = (isinstance(v, (list, tuple)) and len(v) == ln
                  and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                          for x in v))
            if not ok:
                errors.append(f"detections[{i}].{fname} must be {ln} numbers")
            elif not all(0.0 <= float(x) <= 1.0 for x in v):
                errors.append(f"detections[{i}].{fname} out of range 0..1")
        bb = d.get("bbox_2d")
        if (isinstance(bb, (list, tuple)) and len(bb) == 4
                and all(isinstance(x, (int, float)) for x in bb)
                and (bb[0] > bb[2] or bb[1] > bb[3])):
            errors.append(f"detections[{i}].bbox_2d must satisfy x1<=x2, y1<=y2")
        zid = d.get("zone_id")
        if not isinstance(zid, str) or not zid:
            errors.append(f"detections[{i}].zone_id must be a non-empty string")
        if cn == PERSON:
            persons += 1
    if _int_ok(occ) and occ != persons:
        errors.append(f"occupancy_total={occ} but detections contain {persons} person(s)")
    return errors


# ═════════════════ GEOMETRY — projection + zones ═════════════════

def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray-casting test. polygon = [[x, y], ...]. No shapely needed."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def assign_zone(x: float, y: float, zones: dict = None, default: str = None) -> str:
    zones = ZONES if zones is None else zones
    default = ZONE_DEFAULT if default is None else default
    for zone_id, poly in zones.items():
        if point_in_polygon(x, y, poly):
            return zone_id
    return default


def nearest_zone(x: float, y: float, zones: dict = None):
    """Closest zone by centroid distance (fixture fallback, never Unassigned)."""
    zones = ZONES if zones is None else zones
    best, best_d = None, float("inf")
    for zid, poly in zones.items():
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best, best_d = zid, d
    return best


def build_homography(image_points, floor_points) -> np.ndarray:
    """4+ correspondences (image px -> normalized floor) -> 3x3 H."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for homography calibration")
    src = np.asarray(image_points, dtype=np.float32)
    dst = np.asarray(floor_points, dtype=np.float32)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError("need >= 4 calibration points")
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("degenerate calibration points (collinear?)")
    return H


class FloorProjector:
    """Image pixel points -> normalized [0..1] floor coordinates."""

    def __init__(self, mode: str = "linear", homography: np.ndarray = None):
        if mode not in ("linear", "homography"):
            raise ValueError(f"unknown projection mode: {mode}")
        if mode == "homography":
            if homography is None:
                raise ValueError("homography mode requires a calibrated H matrix")
            if cv2 is None:
                raise RuntimeError("opencv-python is required for homography mode")
        self.mode = mode
        self.H = homography
        self.H_inv = np.linalg.inv(homography) if homography is not None else None

    def to_floor(self, points, frame_size) -> np.ndarray:
        """(N,2) image pixels -> (N,2) normalized floor coords, clamped 0..1."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if self.mode == "linear":
            w, h = frame_size
            out = pts / np.array([w, h], dtype=np.float32)
        else:
            out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        return np.clip(out, 0.0, 1.0)

    def floor_to_image(self, points, frame_size) -> np.ndarray:
        """Inverse mapping — draws zone polygons over the camera image."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if self.mode == "linear":
            w, h = frame_size
            out = pts * np.array([w, h], dtype=np.float32)
        else:
            out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.H_inv).reshape(-1, 2)
        return out


# ═════════════════ FIXTURE DETECTION (lights ON / fans spinning) ═════════════════

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter))


@dataclass
class _Fixture:
    fix_id: int
    kind: str
    box: np.ndarray            # [x1,y1,x2,y2] px, EMA-smoothed
    hit_streak: int = 0
    miss_streak: int = 0
    last_seen: int = -1
    confirmed: bool = False


class LightBlobDetector:
    """Finds bright blobs = lights that are ON. Person/screen areas masked out."""

    def __init__(self, thresh: int):
        self.thresh = int(thresh)

    def detect(self, gray, occluder, frame_wh):
        w, h = frame_wh
        _, bright = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY)
        if occluder is not None:
            bright = cv2.bitwise_and(bright, cv2.bitwise_not(occluder))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=2)  # merge glow
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)                 # drop specks
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lo, hi = LIGHT_MIN_AREA_FRAC * w * h, LIGHT_MAX_AREA_FRAC * w * h
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if lo <= a <= hi:
                x, y, bw, bh = cv2.boundingRect(c)
                out.append(np.array([x, y, x + bw, y + bh], np.float32))
        return out


class FanMotionDetector:
    """Finds regions of SUSTAINED motion = spinning fans. People masked out.

    A fan blade crossing the sensor causes motion pixels every frame in the
    same place; we accumulate an EMA 'energy' map so a spot must keep moving
    for many frames before it counts. A person walking by moves and is masked,
    so it never accumulates.
    """

    def __init__(self, diff_thresh: int):
        self.diff_thresh = int(diff_thresh)
        self.prev = None
        self.energy = None

    def detect(self, gray, occluder, frame_wh):
        w, h = frame_wh
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray.copy()
            self.energy = None
            return []
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray.copy()
        _, motion = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        if occluder is not None:
            motion = cv2.bitwise_and(motion, cv2.bitwise_not(occluder))
        m = motion.astype(np.float32)
        if self.energy is None or self.energy.shape != gray.shape:
            self.energy = m.copy()
        else:
            a = FAN_ENERGY_ALPHA
            self.energy *= (1.0 - a)
            self.energy += a * m
            if occluder is not None:
                self.energy[occluder > 0] = 0.0   # people can't accumulate energy
        sustained = np.zeros(gray.shape, np.uint8)
        sustained[self.energy >= FAN_ENERGY_ON] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sustained = cv2.morphologyEx(sustained, cv2.MORPH_OPEN, k)
        sustained = cv2.morphologyEx(sustained, cv2.MORPH_CLOSE, k)  # merge blade flicker
        cnts, _ = cv2.findContours(sustained, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lo, hi = FAN_MIN_AREA_FRAC * w * h, FAN_MAX_AREA_FRAC * w * h
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if lo <= a <= hi:
                x, y, bw, bh = cv2.boundingRect(c)
                out.append(np.array([x, y, x + bw, y + bh], np.float32))
        return out


class FixtureStream:
    """Matches per-frame blobs to persistent fixtures -> stable IDs + debounce.

    A fixture is REPORTED only after ON_CONFIRM consecutive hits, and keeps
    being reported through up to HOLD missed frames (no flicker on/off).
    """

    def __init__(self, kind: str, id_base: int, on_confirm: int, hold: int, detector):
        self.kind = kind
        self._next_id = id_base
        self.on_confirm = on_confirm
        self.hold = hold
        self.detector = detector
        self.fixtures: List[_Fixture] = []

    def update(self, gray, occluder, frame_id, frame_wh) -> List[_Fixture]:
        blobs = self.detector.detect(gray, occluder, frame_wh)

        # greedy IoU matching: fixtures <-> this frame's blobs
        pairs = []
        for i, f in enumerate(self.fixtures):
            for j, b in enumerate(blobs):
                v = _iou(f.box, b)
                if v > 0.15:
                    pairs.append((v, i, j))
        pairs.sort(key=lambda t: t[0], reverse=True)
        used_f, used_b = set(), set()
        for v, i, j in pairs:
            if i in used_f or j in used_b:
                continue
            used_f.add(i)
            used_b.add(j)
            f = self.fixtures[i]
            f.box = 0.35 * blobs[j] + 0.65 * f.box      # EMA-smooth stable bbox
            f.hit_streak += 1
            f.miss_streak = 0
            f.last_seen = frame_id
            if f.hit_streak >= self.on_confirm:
                f.confirmed = True

        for i, f in enumerate(self.fixtures):
            if i not in used_f:
                f.hit_streak = 0
                f.miss_streak += 1
                if f.miss_streak > self.hold:
                    f.confirmed = False

        for j, b in enumerate(blobs):                    # brand-new fixture
            if j not in used_b:
                self.fixtures.append(_Fixture(fix_id=self._next_id, kind=self.kind,
                                              box=b.copy(), hit_streak=1,
                                              miss_streak=0, last_seen=frame_id))
                self._next_id += 1

        self.fixtures = [f for f in self.fixtures
                         if frame_id - f.last_seen <= FIXTURE_STALE_FRAMES]

        return [f for f in self.fixtures if f.confirmed and f.miss_streak <= self.hold]


# ═════════════════ STREAMER ═════════════════

def _tn(x):
    """torch tensor -> numpy (safe on CPU-only installs too)."""
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


@dataclass
class _TrackState:
    zone_id: str
    votes: deque = field(default_factory=deque)
    floor_x: Optional[float] = None
    floor_y: Optional[float] = None
    last_seen: int = -1


class CVStreamer:
    def __init__(self, source=None, hub_url=None, dry_run=False, show=None,
                 weights=None, conf=None, emit_every=None, min_interval=None,
                 device=None, tracker=None, loop=None, mirror=None,
                 detect_lights=None, detect_fans=None,
                 light_thresh=None, fan_motion=None,
                 log_json=False, save_json=None):        # ── NEW
        self.source = SOURCE if source is None else source
        self.dry_run = dry_run
        self.show = SHOW_DEBUG_WINDOW if show is None else show
        self.emit_every = EMIT_EVERY_N_FRAMES if emit_every is None else emit_every
        self.min_interval = MIN_EMIT_INTERVAL_S if min_interval is None else min_interval
        self.conf = CONF_THRESHOLD if conf is None else conf
        self.weights = weights or MODEL_WEIGHTS
        self.device = device or DEVICE
        self.tracker = tracker or TRACKER
        self.loop = LOOP_VIDEO if loop is None else loop
        self.mirror = MIRROR_WEBCAM if mirror is None else mirror
        self.hub_url = (hub_url.rstrip("/") + TELEMETRY_PATH) if hub_url else TELEMETRY_URL

        if cv2 is None:
            raise SystemExit("opencv-python is required: pip install opencv-python")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise SystemExit("ultralytics is required: pip install ultralytics") from e

        log.info("loading YOLO '%s' on %s ...", self.weights, self.device)
        self.model = YOLO(self.weights)
        try:
            self.model.to(self.device)
        except Exception:
            log.warning("could not move model to '%s'; staying on CPU", self.device)

        if FLOOR_MODE == "homography" and CALIB_IMAGE_POINTS:
            self.projector = FloorProjector(
                "homography", build_homography(CALIB_IMAGE_POINTS, FLOOR_RECT))
            log.info("floor projection: HOMOGRAPHY (calibrated)")
        else:
            self.projector = FloorProjector("linear")
            log.warning("floor projection: LINEAR fallback — run "
                        "'python %s calibrate' for accurate floor_pos_2d"
                        % os.path.basename(sys.argv[0]))

        # fixture streams (stable-id light / fan tracking)
        self.light_stream = None
        self.fan_stream = None
        if DETECT_LIGHTS if detect_lights is None else detect_lights:
            self.light_stream = FixtureStream(
                LIGHT, LIGHT_ID_BASE, LIGHT_ON_CONFIRM_FRAMES, LIGHT_HOLD_FRAMES,
                LightBlobDetector(light_thresh or LIGHT_BRIGHTNESS_THRESH))
            log.info("light detection: ON (thresh=%d, ids %d+)",
                     self.light_stream.detector.thresh, LIGHT_ID_BASE)
        if DETECT_FANS if detect_fans is None else detect_fans:
            self.fan_stream = FixtureStream(
                FAN, FAN_ID_BASE, FAN_ON_CONFIRM_FRAMES, FAN_HOLD_FRAMES,
                FanMotionDetector(fan_motion or FAN_DIFF_THRESH))
            log.info("fan detection: ON (diff=%d, ids %d+)",
                     self.fan_stream.detector.diff_thresh, FAN_ID_BASE)

        # ── NEW: JSON capture state
        self.log_json = log_json
        if save_json is False:
            self.save_base = None
        else:
            self.save_base = str(save_json) if save_json else SAVE_JSON_BASE
        self._saved_packets: List[dict] = []
        self._save_fh = None
        if self.save_base:
            log.info("JSON capture -> %s.jsonl (live) + %s.json (on stop) "
                     " [--no-save-json disables]", self.save_base, self.save_base)

        self.zones = ZONES
        self.tracks: Dict[int, _TrackState] = {}
        self.frame_id = 0
        self._last_emit_t = 0.0
        self._fps = 0.0
        self._warned_no_ids = False
        self._stop = threading.Event()
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._session = None
        self._worker: Optional[threading.Thread] = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if self.dry_run:
            return
        self._worker = threading.Thread(
            target=self._post_worker, name="s-post", daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2.0)

    # ------------------------------------------------------------ per frame
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """Full pipeline for one BGR frame -> contract-valid payload dict.
        Public on purpose: N can import and call this directly (in-process)."""
        self.frame_id += 1
        fid = self.frame_id
        h, w = frame_bgr.shape[:2]
        try:
            results = self.model.track(
                source=frame_bgr, persist=True, tracker=self.tracker,
                classes=list(TRACK_CLASS_IDS), conf=self.conf, verbose=False)
        except Exception as e:
            log.warning("inference failed on frame %d: %s", fid, e)
            return make_payload(fid, [])

        detections: List[dict] = []
        all_boxes: List[tuple] = []
        res = results[0] if results else None
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.id is None:
            if not self._warned_no_ids:
                self._warned_no_ids = True
                log.info("no tracked ids yet (tracker warming up)")
        else:
            xyxy = _tn(boxes.xyxy).astype(float)
            ids = _tn(boxes.id).astype(int)
            clss = _tn(boxes.cls).astype(int)
            all_boxes = [tuple(b) for b in xyxy]
            # bottom-center of each bbox: x_mid=(x1+x2)/2, y_bottom=y2
            bottom = np.stack([(xyxy[:, 0] + xyxy[:, 2]) * 0.5, xyxy[:, 3]],
                              axis=1).astype(np.float32)
            floors = self.projector.to_floor(bottom, (w, h))
            for (x1, y1, x2, y2), tid, cls_id, (fx, fy) in zip(xyxy, ids, clss, floors):
                zone_id, sx, sy = self._update_track(int(tid), float(fx), float(fy), fid)
                detections.append(make_detection(
                    int(tid), self.model.names.get(int(cls_id), str(cls_id)),
                    (x1 / w, y1 / h, x2 / w, y2 / h), (sx, sy), zone_id))

        # fixtures (lights ON / fans spinning), masked away from people
        if self.light_stream or self.fan_stream:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            occ = self._build_occluder_mask(all_boxes, (w, h))
            if self.light_stream:
                for f in self.light_stream.update(gray, occ, fid, (w, h)):
                    detections.extend(self._fixture_detection(f, LIGHT, w, h))
            if self.fan_stream:
                for f in self.fan_stream.update(gray, occ, fid, (w, h)):
                    detections.extend(self._fixture_detection(f, FAN, w, h))

        self._prune_tracks(fid)
        return make_payload(fid, detections)

    def _build_occluder_mask(self, boxes_px, frame_wh) -> np.ndarray:
        """255 inside dilated YOLO boxes — keeps people/screens out of fixtures."""
        w, h = frame_wh
        mask = np.zeros((h, w), np.uint8)
        for (x1, y1, x2, y2) in boxes_px:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        grow = int(max(w, h) * OCCLUDER_GROW_FRAC)
        if grow > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
            mask = cv2.dilate(mask, k)
        return mask

    def _fixture_scene_obj(self, fixture: "_Fixture", obj_type: str,
                           w: int, h: int) -> dict:
        """Convert a confirmed fixture -> scene-object dict (lights / fans array)."""
        b = fixture.box
        cx_px = (float(b[0]) + float(b[2])) * 0.5
        cy_px = (float(b[1]) + float(b[3])) * 0.5
        pt = np.array([[cx_px, float(b[3])]], np.float32)
        fx, fy = self.projector.to_floor(pt, (w, h))[0]
        return {
            "type":         obj_type,
            "fixture_id":   fixture.fix_id,
            "bbox_2d": [
                _clamp01(float(b[0]) / w), _clamp01(float(b[1]) / h),
                _clamp01(float(b[2]) / w), _clamp01(float(b[3]) / h),
            ],
            "center":       [_clamp01(cx_px / w), _clamp01(cy_px / h)],
            "floor_pos_2d": [_clamp01(float(fx)), _clamp01(float(fy))],
        }

    def _update_track(self, tid, fx, fy, fid) -> Tuple[str, float, float]:
        st = self.tracks.get(tid)
        if st is None:
            st = _TrackState(zone_id=ZONE_DEFAULT,
                             votes=deque(maxlen=ZONE_SWITCH_CONFIRM_FRAMES))
            self.tracks[tid] = st
        # EMA smoothing of the REPORTED floor position (less jitter for the twin)
        a = FLOOR_SMOOTHING_ALPHA
        if st.floor_x is None:
            st.floor_x, st.floor_y = fx, fy
        else:
            st.floor_x = a * fx + (1.0 - a) * st.floor_x
            st.floor_y = a * fy + (1.0 - a) * st.floor_y
        # zone from the RAW point; reported zone flips only after K matching votes
        raw_zone = assign_zone(fx, fy, self.zones)
        st.votes.append(raw_zone)
        k = ZONE_SWITCH_CONFIRM_FRAMES
        if k <= 1:
            st.zone_id = raw_zone
        elif (len(st.votes) >= k and len(set(st.votes)) == 1
                and st.votes[-1] != st.zone_id):
            st.zone_id = raw_zone
        st.last_seen = fid
        return st.zone_id, st.floor_x, st.floor_y

    def _prune_tracks(self, fid: int):
        stale = [t for t, st in self.tracks.items()
                 if fid - st.last_seen > STALE_TRACK_FRAMES]
        for t in stale:
            self.tracks.pop(t, None)

    # ------------------------------------------------------------ emission
    def _should_emit(self) -> bool:
        if self.frame_id <= WARMUP_FRAMES:
            return False
        if self.frame_id % self.emit_every != 0:
            return False
        if time.monotonic() - self._last_emit_t < self.min_interval:
            return False
        return True

    def _emit(self, payload: dict):
        self._last_emit_t = time.monotonic()
        errors = validate_payload(payload)
        if errors:  # belt & suspenders — should never happen
            log.error("INTERNAL: invalid payload, NOT sent: %s", errors)
            return
        if self.dry_run or self.log_json:               # ── NEW: live stdout
            print(json.dumps(payload))
        if self.save_base:                              # ── NEW: live file
            self._save_packet(payload)
        if self.dry_run:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.warning("send queue full — dropping frame_id=%s (hub too slow)",
                        payload.get("frame_id"))

    def _save_packet(self, payload: dict):              # ── NEW
        """Append one packet to BASE.jsonl (flushed immediately) + memory list."""
        if self._save_fh is None:
            try:
                self._save_fh = open(self.save_base + ".jsonl", "w", encoding="utf-8")
            except OSError as e:
                log.error("cannot open %s.jsonl (%s) — file capture disabled",
                          self.save_base, e)
                self.save_base = None
                return
        if len(self._saved_packets) < SAVED_PACKETS_CAP:
            self._saved_packets.append(payload)
        self._save_fh.write(json.dumps(payload) + "\n")
        self._save_fh.flush()

    def _finalize_save(self):                           # ── NEW
        """Called when the feed stops: close .jsonl, write BASE.json array."""
        if self._save_fh is not None:
            try:
                self._save_fh.close()
            except Exception:
                pass
            self._save_fh = None
        if self.save_base and self._saved_packets:
            path = self.save_base + ".json"
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._saved_packets, f, indent=2)
                log.info("JSON capture done: %d packets -> %s (live log: %s.jsonl)",
                         len(self._saved_packets), path, self.save_base)
            except OSError as e:
                log.error("could not write %s: %s", path, e)
        elif self.save_base:
            log.info("JSON capture: no packets were emitted, no .json written")

    def _post_worker(self):
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._post_with_retry(payload)

    def _post_with_retry(self, payload: dict):
        import requests  # lazy: dry-run never needs it
        if self._session is None:
            self._session = requests.Session()
        last_err = None
        for attempt in range(POST_RETRIES + 1):
            try:
                r = self._session.post(self.hub_url, json=payload,
                                       timeout=POST_TIMEOUT_S)
                if 200 <= r.status_code < 300:
                    log.debug("POST ok frame_id=%s", payload.get("frame_id"))
                else:
                    log.warning("hub answered HTTP %s for frame_id=%s",
                                r.status_code, payload.get("frame_id"))
                return
            except requests.RequestException as e:
                last_err = e
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
        log.error("gave up on frame_id=%s (%s)", payload.get("frame_id"), last_err)

    # ------------------------------------------------------------ debug view
    def _draw_debug(self, img: np.ndarray, payload: dict) -> np.ndarray:
        h, w = img.shape[:2]
        for zone_id, poly in self.zones.items():
            pts = self.projector.floor_to_image(np.array(poly, np.float32), (w, h))
            pts_i = np.round(pts).astype(int).reshape(-1, 1, 2)
            color = ZONE_COLORS.get(zone_id, (200, 200, 200))
            cv2.polylines(img, [pts_i], True, color, 2)
            cv2.putText(img, zone_id,
                        (int(pts_i[0][0][0]) + 4, int(pts_i[0][0][1]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        for d in payload["detections"]:
            cn = d["class_name"]
            if cn == LIGHT:
                color, tag, dot = LIGHT_COLOR, "L", LIGHT_COLOR
            elif cn == FAN:
                color, tag, dot = FAN_COLOR, "F", FAN_COLOR
            else:
                color = ZONE_COLORS.get(d["zone_id"], (128, 128, 128))
                tag, dot = "", (0, 0, 255)
            x1, y1, x2, y2 = d["bbox_2d"]
            p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            cv2.rectangle(img, p1, p2, color, 2)
            cv2.putText(img, f"{tag}#{d['track_id']} {d['zone_id']}",
                        (p1[0], max(14, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            fp = self.projector.floor_to_image(
                np.array([[d["floor_pos_2d"][0], d["floor_pos_2d"][1]]], np.float32),
                (w, h))
            cv2.circle(img, (int(fp[0][0]), int(fp[0][1])), 5, dot, -1)
        n_lights = sum(1 for d in payload["detections"] if d["class_name"] == LIGHT)
        n_fans = sum(1 for d in payload["detections"] if d["class_name"] == FAN)
        n_saved = len(self._saved_packets)              # ── NEW
        hud = (f"frame {payload['frame_id']}  people {payload['occupancy_total']}  "
               f"lights {n_lights}  fans {n_fans}  fps {self._fps:4.1f}  "
               f"saved {n_saved}  mode {self.projector.mode}  -> "
               f"{'dry-run' if self.dry_run else self.hub_url}")
        cv2.putText(img, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------ main loop
    def run(self):
        src = self.source
        is_cam = isinstance(src, int) or (
            isinstance(src, str) and str(src).strip().isdigit())
        cap = cv2.VideoCapture(int(src) if is_cam else str(src))
        if not cap.isOpened():
            raise SystemExit(f"cannot open video source: {src!r} "
                             "(try --source 1, or --source clip.mp4)")
        if is_cam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        log.info("streaming from %s | emit every %d frames -> %s",
                 "webcam" if is_cam else src, self.emit_every,
                 "stdout (dry-run)" if self.dry_run else self.hub_url)
        self.start()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if (not is_cam) and self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.error("source ended or read failed; exiting")
                    break
                if is_cam and self.mirror:
                    frame = cv2.flip(frame, 1)

                t0 = time.monotonic()
                payload = self.process_frame(frame)
                if self._should_emit():
                    self._emit(payload)
                dt = time.monotonic() - t0
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = inst if not self._fps else 0.9 * self._fps + 0.1 * inst

                if self.show:
                    self._draw_debug(frame, payload)
                    try:
                        cv2.imshow(WINDOW_NAME, frame)
                    except Exception:
                        log.warning("no display available — disabling overlay "
                                    "(use --no-display to skip)")
                        self.show = False
                        continue
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.stop()
            cap.release()
            self._finalize_save()                       # ── NEW: write .json file
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("stopped after %d frames", self.frame_id)


# ═════════════════ CALIBRATE (interactive) ═════════════════

_CAL_ORDER = ["far-left (0,0)", "far-right (1,0)", "near-right (1,1)", "near-left (0,1)"]


def _grab_reference_frame(source) -> np.ndarray:
    is_cam = isinstance(source, int) or str(source).strip().isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else str(source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source: {source!r}")
    acc, total = None, 0
    for _ in range((30 if is_cam else 0) + 10):  # warm exposure, then average
        ok, f = cap.read()
        if not ok:
            break
        f = np.float32(f)
        acc = f if acc is None else acc + f
        total += 1
    cap.release()
    if total == 0:
        raise SystemExit("could not read any frame from the source")
    return (acc / total).astype(np.uint8)


def cmd_calibrate(args):
    src = SOURCE if args.source is None else args.source
    frame = _grab_reference_frame(src)
    h, w = frame.shape[:2]
    pts: List[Tuple[int, int]] = []
    H = None
    win = "calibrate: click 4 floor corners  (r=reset  s=save  q=quit)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("Click the 4 WALKABLE-floor corners in this order:")
    for i, label in enumerate(_CAL_ORDER):
        print(f"  {i + 1}. {label}")
    while True:
        img = frame.copy()
        if H is not None:  # zone polygons warped back onto the image (sanity check)
            inv = np.linalg.inv(H)
            for zone_id, poly in ZONES.items():
                p = np.array(poly, np.float32).reshape(-1, 1, 2)
                q = cv2.perspectiveTransform(p, inv).reshape(-1, 2)
                q = np.round(q).astype(int).reshape(-1, 1, 2)
                cv2.polylines(img, [q], True,
                              ZONE_COLORS.get(zone_id, (200, 200, 200)), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 7, (0, 0, 255), 2)
            cv2.putText(img, f"{i + 1}: {_CAL_ORDER[i]}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if len(pts) == 4 and H is None:
            H = build_homography(pts, FLOOR_RECT)
            fp = cv2.perspectiveTransform(
                np.array(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
            for i, (x, y) in enumerate(fp):
                print(f"click {i + 1} -> floor ({x:.3f}, {y:.3f})   "
                      f"expected {tuple(FLOOR_RECT[i])}")
            print("Zones are drawn warped onto the image — check they sit on the "
                  "real floor.")
            print("s = save,  r = redo,  q = quit without saving")
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            pts, H = [], None
        if key == ord("s") and H is not None:
            data = {"mode": "homography",
                    "image_points": [[float(x), float(y)] for x, y in pts],
                    "floor_points": FLOOR_RECT,
                    "image_size": [int(w), int(h)],
                    "created": utc_now_iso()}
            with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"saved {CALIBRATION_FILE} — the streamer auto-loads it on next "
                  f"start. Re-calibrate whenever the camera moves.")
            break
    cv2.destroyAllWindows()


# ═════════════════ MOCK HUB (stdlib only — test S alone) ═════════════════

class _HubHandler(BaseHTTPRequestHandler):
    def _respond(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "up"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != TELEMETRY_PATH:
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._respond(400, {"status": "rejected",
                                "errors": ["body is not valid JSON"]})
            return
        errors = validate_payload(payload)
        if errors:
            print(f"[mock-hub] REJECT frame_id={payload.get('frame_id')}: {errors}",
                  flush=True)
            self._respond(422, {"status": "rejected", "errors": errors})
            return
        per_zone = {}
        for d in payload["detections"]:
            z = per_zone.setdefault(
                d["zone_id"], {"person": 0, "light": 0, "fan": 0})
            if d["class_name"] in z:
                z[d["class_name"]] += 1
        print(f"[mock-hub] OK frame_id={payload['frame_id']} "
              f"occupancy={payload['occupancy_total']} zones={per_zone}", flush=True)
        self._respond(200, {"status": "ok", "frame_id": payload["frame_id"],
                            "occupancy_total": payload["occupancy_total"]})

    def log_message(self, fmt, *args):
        pass  # silence default request logging


def cmd_mock_hub(args):
    addr = (args.host, args.port)
    print(f"mock hub listening on http://{addr[0]}:{addr[1]}{TELEMETRY_PATH} "
          f"(Ctrl+C to stop)")
    print("validates every packet against the exact CV Output Schema; "
          "422 = contract drift")
    try:
        ThreadingHTTPServer(addr, _HubHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nmock hub stopped")


# ═════════════════ SEND SAMPLE (connectivity check) ═════════════════

def cmd_send_sample(args):
    import requests
    url = (args.hub_url or HUB_URL).rstrip("/") + TELEMETRY_PATH
    zone_names = list(ZONES) or [ZONE_DEFAULT]
    dets = []
    for i in range(max(1, args.people)):
        z = zone_names[i % len(zone_names)]
        poly = ZONES.get(z) or [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        dets.append(make_detection(9000 + i, PERSON,
                                   (cx - 0.04, cy - 0.25, cx + 0.04, cy),
                                   (cx, cy), z))
    dets.append(make_detection(LIGHT_ID_BASE + 1, LIGHT,
                               (0.60, 0.05, 0.68, 0.12),
                               (0.62, 0.10), "Zone_B"))
    dets.append(make_detection(FAN_ID_BASE + 1, FAN,
                               (0.20, 0.05, 0.30, 0.15),
                               (0.25, 0.10), "Zone_A"))
    payload = make_payload(args.frame_id, dets)
    assert not validate_payload(payload), validate_payload(payload)
    print("payload (contract-valid):")
    print(json.dumps(payload, indent=2))
    print(f"\nPOST {url} ...")
    try:
        r = requests.post(url, json=payload, timeout=5.0)
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        if 200 <= r.status_code < 300:
            print("[OK] hub accepted the CV Output Schema — you're wired to N.")
        elif r.status_code == 422:
            print("[FAIL] hub rejected the schema — compare N's parser with the contract!")
        else:
            print("[FAIL] unexpected status — is N's server running / route correct?")
    except requests.RequestException as e:
        print(f"[FAIL] could not reach the hub: {e}")
        print("       start the mock hub first:  python s_cv_streamer.py mock-hub")


# ═════════════════ OFFLINE TESTS (no camera / no net) ═════════════════

class _Skipped(Exception):
    pass


def test_payload_keys_exact():
    p = make_payload(1, [])
    assert set(p) == _TOP_KEYS, set(p)


def test_detection_keys_exact():
    d = make_detection(7, PERSON, (0.1, 0.2, 0.3, 0.9), (0.5, 0.6), "Zone_A")
    assert set(d) == _DET_KEYS, set(d)


def test_validator_accepts_valid():
    p = make_payload(5, [
        make_detection(1, PERSON, (0.1, 0.2, 0.3, 0.9), (0.2, 0.85), "Zone_A"),
        make_detection(2, "laptop", (0.4, 0.4, 0.5, 0.5), (0.45, 0.5), "Zone_B")])
    errs = validate_payload(p)
    assert errs == [], errs


def test_validator_rejects_extra_top_key():
    p = make_payload(1, [])
    p["oops"] = 1
    assert any("oops" in e for e in validate_payload(p))


def test_validator_rejects_extra_detection_key():
    d = make_detection(1, PERSON, (0, 0, 0.2, 0.8), (0.1, 0.8), "Zone_A")
    d["confidence"] = 0.9
    assert any("confidence" in e for e in validate_payload(make_payload(1, [d])))


def test_validator_rejects_out_of_range_bbox():
    d = make_detection(1, PERSON, (0, 0, 0.2, 0.8), (0.1, 0.8), "Zone_A")
    d["bbox_2d"] = [1.2, 0.1, 1.4, 0.8]
    errs = validate_payload(make_payload(1, [d]))
    assert any("bbox_2d" in e and "range" in e for e in errs), errs


def test_validator_catches_occupancy_mismatch():
    p = make_payload(1, [])
    p["occupancy_total"] = 7
    assert any("occupancy_total" in e for e in validate_payload(p))


def test_validator_rejects_bad_timestamp():
    p = make_payload(1, [])
    p["timestamp"] = "not-a-date"
    assert any("timestamp" in e for e in validate_payload(p))


def test_clamping_to_unit_range():
    d = make_detection(1, PERSON, (-0.2, 0.1, 1.5, 0.4), (1.7, -0.3), "Zone_B")
    assert d["bbox_2d"] == [0.0, 0.1, 1.0, 0.4], d["bbox_2d"]
    assert d["floor_pos_2d"] == [1.0, 0.0], d["floor_pos_2d"]


def test_occupancy_counts_persons_only():
    p = make_payload(3, [
        make_detection(1, PERSON, (0, 0, 0.2, 0.8), (0.1, 0.8), "Zone_A"),
        make_detection(2, "laptop", (0, 0, 0.2, 0.2), (0.1, 0.2), "Zone_A")])
    assert p["occupancy_total"] == 1 and len(p["detections"]) == 2


def test_fixture_classes_are_contract_valid():
    p = make_payload(9, [
        make_detection(1, PERSON, (0.10, 0.20, 0.20, 0.90), (0.15, 0.85), "Zone_A"),
        make_detection(10001, LIGHT, (0.60, 0.05, 0.68, 0.12), (0.62, 0.10), "Zone_B"),
        make_detection(20001, FAN, (0.20, 0.05, 0.30, 0.15), (0.25, 0.10), "Zone_A")])
    assert p["occupancy_total"] == 1        # lights/fans never inflate occupancy
    assert len(p["detections"]) == 3
    assert validate_payload(p) == []


def test_nearest_zone_fallback():
    zones = {"Zone_A": [[0, 0], [0.4, 0], [0.4, 0.4], [0, 0.4]]}
    assert nearest_zone(0.9, 0.9, zones) == "Zone_A"


def test_timestamp_iso8601_with_tz():
    assert datetime.fromisoformat(make_payload(1, [])["timestamp"]).tzinfo is not None


def test_json_roundtrip():
    p = make_payload(2, [make_detection(9, PERSON, (0, 0, 1, 1), (0.5, 0.5), "Corridor")])
    assert json.loads(json.dumps(p))["frame_id"] == 2


def test_point_in_polygon():
    square = [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6]]
    assert point_in_polygon(0.3, 0.3, square)
    assert not point_in_polygon(0.8, 0.3, square)


def test_assign_zone_default():
    zones = {"Zone_A": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]}
    assert assign_zone(0.25, 0.25, zones) == "Zone_A"
    assert assign_zone(0.75, 0.75, zones) == "Unassigned"


def test_linear_projection_normalizes():
    out = FloorProjector("linear").to_floor(
        np.array([[320.0, 720.0]], np.float32), (640, 720))
    assert np.allclose(out, [[0.5, 1.0]])


def test_homography_corner_mapping():
    if cv2 is None:
        raise _Skipped("opencv not installed")
    H = build_homography([[100, 100], [500, 100], [500, 400], [100, 400]],
                         [[0, 0], [1, 0], [1, 1], [0, 1]])
    proj = FloorProjector("homography", H)
    out = proj.to_floor(np.array([[100.0, 100.0], [300.0, 250.0], [500.0, 400.0]],
                                  np.float32), (640, 480))
    assert np.allclose(out[0], (0, 0), atol=1e-3), out[0]
    assert np.allclose(out[2], (1, 1), atol=1e-3), out[2]
    assert 0 < out[1][0] < 1 and 0 < out[1][1] < 1, out[1]


def cmd_test(args):
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except _Skipped as s:
            skipped += 1
            print(f"  SKIP  {name} ({s})")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed"
          + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)


# ═════════════════ CLI ═════════════════

def cmd_stream(args):
    CVStreamer(
        source=args.source, hub_url=args.hub_url, dry_run=args.dry_run,
        show=False if args.no_display else None, weights=args.model,
        conf=args.conf, emit_every=args.emit_every,
        min_interval=args.min_interval, device=args.device,
        tracker=args.tracker, loop=False if args.no_loop else None,
        mirror=False if args.no_mirror else None,
        detect_lights=False if args.no_lights else None,
        detect_fans=False if args.no_fans else None,
        light_thresh=args.light_thresh, fan_motion=args.fan_motion,
        log_json=args.log_json,                          # ── NEW
        save_json=False if args.no_save_json else args.save_json,  # ── NEW
    ).run()


def _parse_source(s: str):
    s = s.strip()
    return int(s) if s.isdigit() else s


def main():
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):   # allow bare flags -> 'stream'
        argv = ["stream", *argv]

    ap = argparse.ArgumentParser(
        prog="s_cv_streamer.py",
        description="S: Computer Vision & Spatial Mapping — single-file build. "
                    "Run with no subcommand (or 'stream') to go live.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stream", help="live camera/video -> hub (default)")
    p.add_argument("--source", type=_parse_source, default=None,
                   help="0 for webcam (default), 1..., or path to a video file")
    p.add_argument("--hub-url", default=None,
                   help="N hub BASE url, e.g. http://127.0.0.1:8000")
    p.add_argument("--emit-every", type=int, default=None)
    p.add_argument("--min-interval", type=float, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--model", default=None, help="yolov8n.pt / yolov8s.pt ...")
    p.add_argument("--device", default=None, help="cpu | cuda:0 | mps")
    p.add_argument("--tracker", default=None, help="bytetrack.yaml | botsort.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="print payloads to stdout, no network")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-lights", action="store_true",
                   help="disable light detection (old behavior)")
    p.add_argument("--no-fans", action="store_true",
                   help="disable fan detection (old behavior)")
    p.add_argument("--light-thresh", type=int, default=None,
                   help="brightness 0-255; lower = detect dimmer lights")
    p.add_argument("--fan-motion", type=int, default=None,
                   help="motion sensitivity 0-255; lower = detect slower fans")
    p.add_argument("--log-json", action="store_true",               # ── NEW
                   help="print each emitted JSON packet to stdout while POSTing")
    p.add_argument("--save-json", nargs="?", const=SAVE_JSON_BASE,   # ── NEW
                   default=SAVE_JSON_BASE, metavar="BASE",
                   help="capture JSON to BASE.jsonl (live) + BASE.json (on stop); "
                        "default: " + SAVE_JSON_BASE)
    p.add_argument("--no-save-json", action="store_true",           # ── NEW
                   help="do not write any JSON capture files")
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("calibrate", help="interactive 4-point floor calibration")
    p.add_argument("--source", type=_parse_source, default=None)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("mock-hub", help="mini receiver to test S alone (Ctrl+C)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_mock_hub)

    p = sub.add_parser("send-sample",
                       help="POST one fake contract-valid frame to the hub")
    p.add_argument("--hub-url", default=None)
    p.add_argument("--people", type=int, default=3)
    p.add_argument("--frame-id", type=int, default=1)
    p.set_defaults(func=cmd_send_sample)

    p = sub.add_parser("test", help="offline contract tests (no camera / no net)")
    p.set_defaults(func=cmd_test)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()