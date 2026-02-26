"""
video_to_dashboard.py  — Pipeline complet vers le dashboard
=============================================================
Lancer EN PARALLÈLE de ws_dashboard_server.py :

  # Toutes les caméras LIVE :
  .\\venv_new\\Scripts\\python.exe video_to_dashboard.py --live

  # Une seule caméra :
  .\\venv_new\\Scripts\\python.exe video_to_dashboard.py --cam imou_cam1

  # Fichier vidéo (dialogue) :
  .\\venv_new\\Scripts\\python.exe video_to_dashboard.py

  # Fichier vidéo direct :
  .\\venv_new\\Scripts\\python.exe video_to_dashboard.py --video path/to/file.mp4

Modèles chargés :
  • yolov8x.pt          — détection personnes + tracking
  • staff_detector_v8.pt — détection staff visuelle
  • table model          — détection tables (every 5 frames)
  • staff_classifier_yolo.pt / staff_classifier.pt  — classifier corps
  • face_db.pkl + ArcFace — identification par visage
"""

import argparse, cv2, json, math, pickle, queue, random, sys, threading, time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Try to import ws_dashboard_server to write _shared directly ───────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import ws_dashboard_server as _ws
    WS_SHARED   = _ws._shared
    WS_COLORS   = _ws.COLORS
    WS_NAME_MAP = _ws.NAME_MAP
    WS_PENDING  = _ws._pending_video   # ← video switch requests from HTTP API
    print("✅ ws_dashboard_server importé — push direct vers _shared")
except ImportError:
    print("⚠️  ws_dashboard_server introuvable → lancer en parallèle")
    WS_SHARED   = {"servers": {}, "tables": {}, "global": {}, "last_update": 0.0}
    WS_COLORS   = {0: "#F59E0B", 1: "#60A5FA", 2: "#F87171", 3: "#A78BFA", 4: "#34D399", 5: "#FB923C"}
    WS_NAME_MAP = {"x": "Mohamed Ali", "y": "Rami", "z": "Sadio"}
    WS_PENDING  = {"path": None}

# ── Configuration ──────────────────────────────────────────────────────────────
# Pointe vers le dossier contenant les modèles (.pt), face_db.pkl, vidéos, etc.
_SCRIPT_DIR = Path(__file__).parent
BASE = _SCRIPT_DIR.parent / "staff detection images"
if not BASE.exists():
    BASE = _SCRIPT_DIR  # Fallback si lancé depuis le dossier original

RTSP_STREAMS = {
    "imou_cam1": "rtsp://100.96.105.67:8554/imou_cam1",
    "imou_cam2": "rtsp://100.96.105.67:8554/imou_cam2",
    "imou_cam3": "rtsp://100.96.105.67:8554/imou_cam3",
    "imou_cam4": "rtsp://100.96.105.67:8554/imou_cam4",
    "imou_cam5": "rtsp://100.96.105.67:8554/imou_cam5",
    "imou_cam6": "rtsp://100.96.105.67:8554/imou_cam6",
}

STAFF_MODEL_PATH  = BASE / "staff_detector_v8.pt"
PERSON_MODEL_PATH = BASE / "yolov8x.pt"   # same as auto_test_detection.py — accurate person detection
PERSON_MODEL_FAST = BASE / "yolov8n.pt"   # fallback for live 6-cameras mode (speed over accuracy)
TABLE_MODEL_PATH  = BASE.parent / "vision-ia-restaurant/runs/detect/kitchen_table_specialist_yolo11x/weights/best.pt"
CLS_YOLO_PATH     = BASE / "staff_classifier_yolo.pt"
CLS_RESNET_PATH   = BASE / "staff_classifier.pt"
FACE_DB_PATH      = BASE / "face_db.pkl"
BYTETRACK_CFG     = BASE / "bytetrack_kitchen.yaml"

COLOR_CYCLE = list(WS_COLORS.values())
ZONES = ["Zone A — Entrée", "Zone B — Centre", "Zone C — Terrasse", "Zone D — Bar", "Zone E — Cuisine"]
TABLE_GRID = [  # (id, svgX, svgY)
    (1,12,15),(2,12,38),(3,12,60),(4,12,80),
    (5,38,15),(6,38,38),(7,38,60),(8,38,80),
    (9,63,15),(10,63,38),(11,63,60),(12,63,80),
    (13,85,25),(14,85,55),(15,85,78),
]

STAFF_THRESHOLD  = 0.45  # same as auto_test_detection.py (was 0.30 — too permissive)
FACE_THRESHOLD   = 0.45
FACE_CROP_TOP    = 0.35
FACE_INTERVAL    = 1.0   # seconds between ArcFace attempts
BODY_INTERVAL    = 0.15  # seconds between body classifier (3× faster — was 0.5s)
VOTES_TO_LOCK    = 2       # (face votes count ×2) — was 3
RECOVER_DIST     = 120     # px
NAME_MEMORY      = 20.0    # sec
PUSH_INTERVAL    = 1.0     # sec
STAFF_STALE_TIMEOUT = 28800.0 # 8h — staff cards persist the entire workday

# ── Shared aggregator (all camera threads write here) ─────────────────────────
_global_lock = threading.Lock()
_all_servers: dict[str, dict] = {}   # name → latest payload
_all_tables : list[dict] = []
_all_global : dict = {}              # global counters (clients, tables, etc.)
_cam_clients: dict[str, int] = {}    # camera → client count
_alert_history: dict[str, dict] = {} # alert_id → {firstSeen, count, sequence[], server}
_alert_frames: dict[str, bytes] = {} # frameId → JPEG bytes (max 150 entries)
_service_start = datetime.now().strftime("%H:%M")
_stop_event = threading.Event()

def _save_alert_frame(frame_id: str, frame) -> None:
    """Encode a frame as JPEG and store in _alert_frames (cap at 150 entries)."""
    import cv2 as _cv2
    try:
        ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with _global_lock:
                _alert_frames[frame_id] = buf.tobytes()
                # Keep memory bounded: drop oldest entries
                while len(_alert_frames) > 150:
                    _alert_frames.pop(next(iter(_alert_frames)))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PersonTracker (identical logic to test_staff_client.py)
# ══════════════════════════════════════════════════════════════════════════════
class PersonTracker:
    def __init__(self):
        self.positions     = deque(maxlen=60)
        self.box_ratios    = deque(maxlen=30)
        self.at_table_hist = deque(maxlen=30)
        self.total_frames  = 0
        self.v8_detections = 0
        self.pos_changes   = 0
        self.last_sig_pos  = None
        self.locked_as_staff = False
        self.staff_frames  = 0
        self._was_staff    = False

    def update(self, cx, cy, w, h, is_v8, at_table, fw, fh):
        self.total_frames += 1
        self.positions.append((cx, cy))
        if is_v8:
            self.v8_detections += 1
        self.at_table_hist.append(1 if at_table else 0)
        self.box_ratios.append(h / max(w, 1))
        if self.last_sig_pos is None:
            self.last_sig_pos = (cx, cy)
        dist = math.hypot(cx - self.last_sig_pos[0], cy - self.last_sig_pos[1])
        if dist > 50:
            self.pos_changes += 1
            self.last_sig_pos = (cx, cy)

    def speed(self):
        if len(self.positions) < 5:
            return 0.0
        pts = list(self.positions)
        return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                   for i in range(1, len(pts))) / len(pts)

    def is_standing(self):
        if len(self.box_ratios) < 3:
            return True
        # Threshold aligned with auto_test_detection.py (was 1.3 — too permissive)
        return sum(self.box_ratios) / len(self.box_ratios) > 1.8

    def is_at_table(self):
        if len(self.at_table_hist) < 3:
            return False
        return sum(self.at_table_hist) / len(self.at_table_hist) > 0.5

    def get_combined_score(self):
        if self.total_frames == 0:
            return 0.0, 0.0, 0.5, 0.5
        v8_ratio = self.v8_detections / self.total_frames
        visual   = min(v8_ratio * 1.2, 1.0)
        if self.total_frames >= 10 and visual < 0.20:
            return 0.0, visual, 0.5, 0.5
        table_s = 0.5
        if len(self.at_table_hist) >= 3:
            atr = sum(self.at_table_hist) / len(self.at_table_hist)
            if atr > 0.7 and not self.is_standing():   table_s = 0.05
            elif atr > 0.5 and not self.is_standing():  table_s = 0.15
            elif atr > 0.5 and self.is_standing():      table_s = 0.40
            elif atr < 0.2:                              table_s = 0.70
        beh = 0.5
        if self.total_frames >= 5:
            beh += 0.3 if self.is_standing() else -0.2
        if self.total_frames >= 10:
            spd = self.speed()
            beh += 0.3 if spd > 5.0 else (-0.2 if spd < 1.0 else 0)
        if self.total_frames >= 20:
            beh += 0.2 if self.pos_changes >= 3 else (-0.2 if self.pos_changes == 0 else 0)
        beh = max(0.0, min(1.0, beh))
        combined = visual * 0.70 + table_s * 0.15 + beh * 0.15
        if self.total_frames >= 15 and self.is_at_table():
            if not self.is_standing() and visual < 0.25:
                combined = min(combined, 0.25)
        # FAST_LOCK: requires BOTH minimum hits (5) AND minimum ratio (≥30%)
        # Previously: 5/10 → too easy (5 hits in 10 frames = 50% fine, but 5 in 15 = 33%
        # borderline). Clients near the counter can accumulate 5 hits quickly then stop.
        # Now: also require v8_ratio >= 0.30 = staff_v8 fires on at least 30% of frames.
        # Example: 5 hits in 10 frames (50%) → lock ✅
        #          5 hits in 15 frames (33%) → lock ✅
        #          5 hits in 20 frames (25%) → NO lock ✅ (likely a client near bar)
        if self.v8_detections >= 5 and self.total_frames >= 10 and v8_ratio >= 0.30:
            combined = max(combined, 0.60)
            self.locked_as_staff = True
        return combined, visual, table_s, beh

    def get_label(self):
        if self.total_frames < 2:
            return None, 0, 0, 0, 0
        # ── Unlock mechanism: if locked but v8 ratio drops, unlock ──────────
        # Handles: track locked early (5 hits in first 10f), then 90+ frames
        # with 0 hits → ratio drops to 5% → should be client, not staff forever.
        if self.locked_as_staff and self.total_frames >= 30:
            v8_ratio_now = self.v8_detections / self.total_frames
            if v8_ratio_now < 0.12:  # less than 12% → not consistently staff
                self.locked_as_staff = False
                self._was_staff = False
        combined, visual, table, beh = self.get_combined_score()
        if self.locked_as_staff:
            return "staff", combined, visual, table, beh
        if self._was_staff:
            if combined < 0.30:
                self._was_staff = False
                self.staff_frames = max(0, self.staff_frames - 1)
                return "client", combined, visual, table, beh
            else:
                self.staff_frames += 1
                if self.staff_frames >= 20:
                    self.locked_as_staff = True
                return "staff", combined, visual, table, beh
        else:
            if combined >= STAFF_THRESHOLD:
                self._was_staff = True
                self.staff_frames += 1
                return "staff", combined, visual, table, beh
            return "client", combined, visual, table, beh


def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / max(a1 + a2 - inter, 1)

def near_table(pbox, tbox, expand=1.3):
    pcx = (pbox[0]+pbox[2])/2; pcy = (pbox[1]+pbox[3])/2
    tcx = (tbox[0]+tbox[2])/2; tcy = (tbox[1]+tbox[3])/2
    tw = (tbox[2]-tbox[0])*expand; th = (tbox[3]-tbox[1])*expand
    return abs(pcx-tcx) < tw/2 and abs(pcy-tcy) < th/2


# ══════════════════════════════════════════════════════════════════════════════
# ServerMetrics — computes dashboard payload from raw tracking data
# ══════════════════════════════════════════════════════════════════════════════
class ServerMetrics:
    def __init__(self, name: str, color: str, camera: str):
        self.name    = name
        self.color   = color
        self.camera  = camera
        self.arrival = datetime.now().strftime("%H:%M")
        self.history : deque = deque(maxlen=8)
        self.tables_visited = 0
        self.tables_total   = 15
        self.zone_idx  = 0
        self.frames    = 0
        self.stand_f   = 0
        self._speed_s  = 50.0
        self._react_s  = 60.0
        self._cov_s    = 50.0
        self._stand_s  = 60.0
        self._svgx     = 50.0
        self._svgy     = 50.0
        self._srv_idx  = 0
        self._last_table_visit_t = None
        self._react_times: deque = deque(maxlen=10)
        self._at_table_prev = False
        self._inactive_since: float | None = None
        self._low_score_since: float | None = None
        # ── Daily persistent tracking ────────────────────────────────────
        self._away_since:  float | None = None   # timestamp when left frame
        self._active_sec:  float = 0.0           # total seconds visible
        self._active_tick: float = time.time()   # last visibility tick
        self._day_scores:  list  = []            # list of (timestamp, score) for day
        self._last_seen_str: str = datetime.now().strftime("%H:%M:%S")

    def tick(self, cx_svg, cy_svg, spd_px, is_standing, at_table, zone_idx):
        self.frames += 1
        self._svgx = cx_svg
        self._svgy = cy_svg
        self.zone_idx = zone_idx
        if is_standing:
            self.stand_f += 1

        now_t = time.time()

        # ── Reactivity: measure real time between table visits ──────────
        if at_table and not self._at_table_prev:
            # Just arrived at a new table
            if self._last_table_visit_t is not None:
                interval_min = (now_t - self._last_table_visit_t) / 60.0
                # Good reactivity = visiting tables every 1-4 min
                # Score: 100 if <=1min, 70 if 2min, 50 if 3min, 30 if >5min
                react_val = max(0, min(100, 100 - (interval_min - 1.0) * 20))
                self._react_times.append(react_val)
            self._last_table_visit_t = now_t
            self.tables_visited = min(self.tables_visited + 1, self.tables_total)
        self._at_table_prev = at_table

        # ── Speed: minimum 35 even when stationary (serving at bar/counter)
        # A server standing still at the bar is working, not idle
        if at_table:
            effective_spd = 3.5   # table service = neutral-good speed equivalent
        elif is_standing and spd_px < 0.8:
            # Standing still but probably working (bar, counter, station)
            # Give minimum score of 35 so score doesn't collapse unfairly
            effective_spd = 2.0   # → spd_s = 36
        else:
            effective_spd = spd_px
        spd_s = min(100.0, effective_spd * 18)

        # ── Coverage: don't drop below neutral until we have real data ──
        stand_pct = (self.stand_f / max(self.frames, 1)) * 100
        if self.tables_visited > 0:
            # Real data: use actual ratio
            cov_s = (self.tables_visited / self.tables_total) * 100
        else:
            # No visits yet: stay neutral (don't collapse to 0)
            cov_s = 50.0

        # ── Reactivity score: average of real intervals, or neutral 60 ──
        if self._react_times:
            react_s = sum(self._react_times) / len(self._react_times)
        else:
            react_s = 60.0   # neutral while no interval data yet

        a_speed = 0.08   # slower smoothing for speed (less jittery)
        a_cov   = 0.05   # very slow smoothing for coverage (stable)
        a_react = 0.10
        a_stand = 0.10
        self._speed_s = (1-a_speed)*self._speed_s + a_speed*spd_s
        self._react_s = (1-a_react)*self._react_s + a_react*react_s
        self._cov_s   = (1-a_cov  )*self._cov_s   + a_cov  *cov_s
        self._stand_s = (1-a_stand)*self._stand_s + a_stand*stand_pct

        # ── Inactivity tracking ───────────────────────────────────────────
        if effective_spd < 0.5 and not at_table:
            if self._inactive_since is None:
                self._inactive_since = now_t
        else:
            self._inactive_since = None

        # ── Daily active time accumulation ─────────────────────────────
        self._active_sec += now_t - self._active_tick
        self._active_tick = now_t
        self._away_since = None          # reset: server is visible
        self._last_seen_str = datetime.now().strftime("%H:%M:%S")

    def mark_away(self) -> None:
        """Call once per push cycle when this server is NOT in the camera frame."""
        now_t = time.time()
        if self._away_since is None:
            self._away_since = now_t   # just left the frame
        self._active_tick = now_t      # keep tick updated so next visible tick is accurate

    def to_payload(self, video_pos_sec: float = 0.0, current_frame=None,
                   status: str = "active") -> dict:
        score = max(0, min(100, round(
            self._speed_s*0.30 + self._react_s*0.30 +
            self._cov_s*0.25  + self._stand_s*0.15
        )))
        now_str = datetime.now().strftime("%H:%M")
        self.history.append({"time": now_str, "value": score})

        # ── Day score: track all scores; compute daily average ──────────
        self._day_scores.append(score)
        if len(self._day_scores) > 3600:   # keep max last 3600 snapshots (~1h at 1s)
            self._day_scores = self._day_scores[-3600:]
        day_score_avg = round(sum(self._day_scores) / len(self._day_scores)) if self._day_scores else score

        arr_h, arr_m = map(int, self.arrival.split(":"))
        dur_min = max(0, datetime.now().hour*60 + datetime.now().minute - arr_h*60 - arr_m)
        active_min = round(self._active_sec / 60, 1)
        avg_resp = round(max(0.5, (100-self._react_s)/14), 1)
        stand_pct = round((self.stand_f / max(self.frames, 1)) * 100)
        cov_pct   = round((self.tables_visited / self.tables_total) * 100)
        spd_label = "Rapide" if self._speed_s >= 65 else ("Normal" if self._speed_s >= 35 else "Lent")

        # ── Away duration ────────────────────────────────────────────────
        away_min: float = 0.0
        if self._away_since is not None:
            away_min = round((time.time() - self._away_since) / 60, 1)

        alerts = []
        now_t = time.time()
        now_str_full = datetime.now().strftime("%H:%M:%S")

        def _make_alert(alert_id: str, alert_type: str, message: str) -> dict:
            """Create alert enriched with sequence history + video position + screenshot."""
            with _global_lock:
                if alert_id not in _alert_history:
                    _alert_history[alert_id] = {
                        "firstSeen": now_str_full,
                        "count": 0,
                        "sequence": [],
                        "server": self.name,
                        "type": alert_type,
                    }
                hist = _alert_history[alert_id]
                hist["count"] += 1
                hist["lastSeen"] = now_str_full
                # Keep last 10 entries; each = {time, videoSec, frameId}
                last = hist["sequence"][-1] if hist["sequence"] else {}
                if not hist["sequence"] or last.get("time") != now_str_full:
                    fid = f"{alert_id}_{int(now_t)}"
                    hist["sequence"].append({
                        "time":     now_str_full,
                        "videoSec": round(video_pos_sec, 1),
                        "frameId":  fid,
                    })
                    if len(hist["sequence"]) > 10:
                        hist["sequence"].pop(0)
                    # Save screenshot async (avoid blocking main loop)
                    if current_frame is not None:
                        threading.Thread(
                            target=_save_alert_frame,
                            args=(fid, current_frame.copy()),
                            daemon=True
                        ).start()
            return {
                "id":        alert_id,
                "type":      alert_type,
                "message":   message,
                "time":      now_str_full,
                "firstSeen": hist["firstSeen"],
                "count":     hist["count"],
                "sequence":  list(hist["sequence"]),
                "server":    self.name,
            }

        # ── Alerts only fire when server is visible (active) ─────────────
        if status == "active":
            if avg_resp > 6:
                alerts.append(_make_alert(f"a-{self.name}-r",  "critical",
                               f"Réactivité critique : {avg_resp} min"))
            elif avg_resp > 4:
                alerts.append(_make_alert(f"a-{self.name}-rw", "warning",
                               f"Réactivité faible : {avg_resp} min"))

            if cov_pct < 30:
                alerts.append(_make_alert(f"a-{self.name}-c",  "critical",
                               f"Couverture très faible : {cov_pct}%"))
            elif cov_pct < 50 and dur_min > 15:
                alerts.append(_make_alert(f"a-{self.name}-cw", "warning",
                               f"Couverture faible : {cov_pct}% des tables"))

            if score < 35:
                if self._low_score_since is None:
                    self._low_score_since = now_t
                low_for_min = round((now_t - self._low_score_since) / 60)
                if low_for_min >= 5:
                    alerts.append(_make_alert(f"a-{self.name}-s", "critical",
                                   f"Score d'efficacité < 40 depuis {low_for_min} min"))
            else:
                self._low_score_since = None

            INACTIVITY_WARN_S = 3 * 60
            if self._inactive_since is not None:
                idle_s = now_t - self._inactive_since
                if idle_s >= INACTIVITY_WARN_S:
                    idle_min = int(idle_s // 60)
                    idle_sec = int(idle_s % 60)
                    alerts.append(_make_alert(f"a-{self.name}-i", "warning",
                                   f"Inactivité détectée : {idle_min}min {idle_sec:02d}s sans mouvement"))

        return {
            "id":               f"srv-{abs(hash(self.name)) % 999 + 1}",
            "name":             self.name,
            "avatar":           "".join(p[0].upper() for p in self.name.split()[:2]),
            "camera":           self.camera,
            "arrivalTime":      self.arrival,
            "serviceDuration":  f"{dur_min//60}h {dur_min%60:02d}m",
            "score":            score,
            "dayScore":         day_score_avg,       # cumulative day average
            "activeMin":        active_min,          # total minutes visible today
            "awayMin":          away_min,            # minutes since last seen
            "lastSeen":         self._last_seen_str,
            "status":           status,              # "active" | "en_pause"
            "speedScore":       round(self._speed_s),
            "reactivityScore":  round(self._react_s),
            "coverageScore":    round(self._cov_s),
            "standingScore":    round(self._stand_s),
            "speed":            round(self._speed_s / 20, 2),
            "speedLabel":       spd_label,
            "tablesVisited":    self.tables_visited,
            "totalTables":      self.tables_total,
            "avgResponseTime":  avg_resp,
            "standingPercent":  stand_pct,
            "recognitionScore": 90,
            "lastZone":         ZONES[self.zone_idx % len(ZONES)],
            "color":            self.color,
            "alerts":           alerts,
            "speedHistory":     list(self.history),
            "activityBySlot":   list(self.history),
            "position":         {"x": round(self._svgx, 1), "y": round(self._svgy, 1)},
            "source":           "live_detection",
        }


# ══════════════════════════════════════════════════════════════════════════════
# load_models — chargés UNE FOIS dans le thread principal et partagés
# ══════════════════════════════════════════════════════════════════════════════
def load_models(is_live_mode: bool = False, num_cams: int = 1):
    models = {}

    # Load BOTH person models — CameraThread will choose based on source type:
    # - RTSP live cameras → yolov8n (speed for 6 cameras)
    # - Video files → yolov8x (accuracy, same as auto_test_detection.py ✅)
    px_path = PERSON_MODEL_PATH if PERSON_MODEL_PATH.exists() else PERSON_MODEL_FAST
    pn_path = PERSON_MODEL_FAST if PERSON_MODEL_FAST.exists() else PERSON_MODEL_PATH
    models["person_x"] = YOLO(str(px_path))   # accurate
    models["person_n"] = YOLO(str(pn_path))   # fast
    models["person"]   = models["person_x"]   # default = accurate
    print(f"  ✅ Person (video): {px_path.name}")
    print(f"  ✅ Person (live):  {pn_path.name}")

    # Staff v8
    if STAFF_MODEL_PATH.exists():
        models["staff_v8"] = YOLO(str(STAFF_MODEL_PATH))
        print(f"  ✅ Staff v8: {STAFF_MODEL_PATH.name}")
    else:
        models["staff_v8"] = None
        print(f"  ⚠️  Staff v8 non trouvé: {STAFF_MODEL_PATH.name}")

    # Table detector
    if TABLE_MODEL_PATH.exists():
        models["table"] = YOLO(str(TABLE_MODEL_PATH))
        print(f"  ✅ Table: {TABLE_MODEL_PATH.name}")
    else:
        models["table"] = None
        print(f"  ⚠️  Table model non trouvé")

    # Body classifier
    if CLS_YOLO_PATH.exists():
        models["classifier"] = YOLO(str(CLS_YOLO_PATH))
        models["cls_names"]  = models["classifier"].names
        models["cls_type"]   = "yolo"
        print(f"  ✅ Classifier YOLO: {CLS_YOLO_PATH.name}")
    elif CLS_RESNET_PATH.exists():
        import torch, torch.nn as nn
        from torchvision import models as tvm
        import torchvision.transforms as T
        ck = torch.load(str(CLS_RESNET_PATH), map_location="cpu", weights_only=True)
        net = tvm.resnet18(weights=None)
        net.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(net.fc.in_features, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, ck["num_classes"])
        )
        net.load_state_dict(ck["model_state"])
        net.eval()
        models["classifier"] = net
        models["cls_names"]  = {i: n for i, n in enumerate(ck["class_names"])}
        models["cls_type"]   = "resnet"
        models["cls_transform"] = T.Compose([
            T.ToPILImage(), T.Resize((224,224)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        print(f"  ✅ Classifier ResNet: {CLS_RESNET_PATH.name}")
    else:
        models["classifier"] = None
        models["cls_names"]  = {}
        models["cls_type"]   = None
        print(f"  ⚠️  Body classifier non trouvé")

    # Face DB
    if FACE_DB_PATH.exists():
        with open(FACE_DB_PATH, "rb") as f:
            models["face_db"] = pickle.load(f)
        print(f"  ✅ Face DB: {list(models['face_db'].keys())}")
    else:
        models["face_db"] = None
        print(f"  ⚠️  face_db.pkl non trouvé")

    # ArcFace
    try:
        venv_site = BASE / "venv_new/Lib/site-packages"
        sys.path.insert(0, str(venv_site.resolve()))
        from deepface import DeepFace
        models["arcface"] = DeepFace
        print(f"  ✅ ArcFace (DeepFace) chargé")
    except Exception as e:
        models["arcface"] = None
        print(f"  ⚠️  ArcFace: {e}")

    return models


def px_to_svg(cx, cy, fw, fh):
    return (max(3.0, min(97.0, (cx/max(fw,1))*96+2)),
            max(10.0, min(90.0, (cy/max(fh,1))*80+10)))


# ══════════════════════════════════════════════════════════════════════════════
# face_worker — thread asynchrone de reconnaissance faciale
# ══════════════════════════════════════════════════════════════════════════════
def make_face_worker(models, face_thresh):
    req_q = queue.Queue(maxsize=8)
    res_q = queue.Queue()

    def worker():
        arcface = models.get("arcface")
        face_db = models.get("face_db")
        while True:
            item = req_q.get()
            if item is None:
                break
            tid, crop = item
            name_out, conf_out = None, 0.0
            if arcface and face_db:
                try:
                    result = arcface.represent(
                        img_path=crop, model_name="ArcFace",
                        detector_backend="retinaface", enforce_detection=True
                    )
                    if result:
                        raw = np.array(result[0]["embedding"])
                        emb = raw / (np.linalg.norm(raw) + 1e-8)
                        scores = {}
                        for n, data in face_db.items():
                            best = max(
                                (float(np.dot(emb, np.array(e) / (np.linalg.norm(e) + 1e-8)))
                                 for e in data.get("embeddings", [])),
                                default=-1.0
                            )
                            if best < 0:
                                c = np.array(data["centroid"])
                                best = float(np.dot(emb, c / (np.linalg.norm(c)+1e-8)))
                            scores[n] = best
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        if ranked and ranked[0][1] >= face_thresh:
                            if len(ranked) < 2 or (ranked[0][1] - ranked[1][1]) >= 0.05:
                                bk, bs = ranked[0]
                                name_out = WS_NAME_MAP.get(bk, bk.capitalize())
                                conf_out = float(bs)
                except Exception:
                    pass
            res_q.put((tid, name_out, conf_out))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return req_q, res_q, t


def identify_by_body(models, crop):
    """Body classifier — returns (display_name, conf) or (None, 0)."""
    clf = models.get("classifier")
    if clf is None:
        return None, 0.0
    try:
        if models["cls_type"] == "yolo":
            r = clf(crop, verbose=False)[0]
            idx  = r.probs.top1
            conf = float(r.probs.top1conf)
            if conf >= 0.55:
                raw = models["cls_names"][idx]
                return WS_NAME_MAP.get(raw, raw.capitalize()), conf
        else:
            import torch
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = models["cls_transform"](img_rgb).unsqueeze(0)
            with torch.no_grad():
                probs  = torch.softmax(clf(t), dim=1)
                conf, idx = probs.max(1)
            if float(conf) >= 0.55:
                raw = models["cls_names"][int(idx)]
                return WS_NAME_MAP.get(raw, raw.capitalize()), float(conf)
    except Exception:
        pass
    return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CameraThread — one thread per RTSP / video source
# ══════════════════════════════════════════════════════════════════════════════
class CameraThread(threading.Thread):
    def __init__(self, source: str, label: str, models: dict, show_gui: bool = False, force_staff: bool = False):
        super().__init__(daemon=True)
        self.source      = source
        self.label       = label
        # Keep shared models (arcface, classifier, face_db) but NOT the YOLO ones
        # YOLO models are NOT thread-safe (fuse() mutates state on first call)
        # Each thread will load its own person + staff_v8 + table instances
        self._shared_models = models
        self.show_gui    = show_gui
        self.force_staff = force_staff
        self.name        = f"cam-{label}"

    def run(self):
        src = str(self.source)
        is_rtsp = src.startswith("rtsp")

        # ── Per-thread YOLO instances (NOT shared — fuse() is not thread-safe) ─
        thread_models = dict(self._shared_models)  # copy shared refs (arcface, etc.)
        try:
            # ── Choose person model based on source type ────────────────────
            # RTSP (live camera) → yolov8n: fast, handles 6 simultaneous streams
            # Video file        → yolov8x: accurate, same as auto_test_detection.py
            if is_rtsp and PERSON_MODEL_FAST.exists():
                chosen_person = PERSON_MODEL_FAST
            else:
                chosen_person = PERSON_MODEL_PATH if PERSON_MODEL_PATH.exists() else PERSON_MODEL_FAST
            print(f"  📦 {self.label}: person={chosen_person.name} ({'live' if is_rtsp else 'video'})")
            thread_models["person"] = YOLO(str(chosen_person))
            if STAFF_MODEL_PATH.exists():
                thread_models["staff_v8"] = YOLO(str(STAFF_MODEL_PATH))
            else:
                thread_models["staff_v8"] = None
            if TABLE_MODEL_PATH.exists():
                thread_models["table"] = YOLO(str(TABLE_MODEL_PATH))
            else:
                thread_models["table"] = None
        except Exception as e:
            print(f"  ❌ {self.label}: erreur chargement modèles locaux: {e}")
            return
        self.models = thread_models

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"  ❌ {self.label}: impossible d'ouvrir {src}")
            return
        if is_rtsp:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        print(f"  📺 {self.label}: {fw}×{fh} @ {fps_src:.1f}fps")
        # Share video path so frontend can seek to alert positions
        if not is_rtsp:
            WS_SHARED["video_path"] = src

        # Per-camera state
        trackers      = defaultdict(PersonTracker)
        id_to_name    = {}           # tid → (raw_key, conf)
        id_votes      = defaultdict(list)
        id_locked     = set()
        id_reverify   = defaultdict(list)
        id_face_last  = defaultdict(float)
        id_body_last  = defaultdict(float)
        last_pos      = {}           # tid → (cx, cy)
        lost_tracks   = {}           # name → (cx, cy, t, conf, was_crossing)
        crossing_flag = {}           # tid → timestamp
        recov_cool    = {}           # tid → timestamp
        face_pending  = set()
        cached_tables = []           # [x1,y1,x2,y2] in pixels
        server_metrics: dict[str, ServerMetrics] = {}
        name_color_idx: dict[str, int] = {}

        face_req_q, face_res_q, face_t = make_face_worker(self.models, FACE_THRESHOLD)

        cam_cfg_path = BASE / "camera_config.json"
        src_name = self.label
        staff_v8_conf = 0.50  # default
        if cam_cfg_path.exists():
            try:
                d = json.loads(cam_cfg_path.read_text(encoding="utf-8"))
                c = dict(d.get("default", {}))
                # Case-insensitive prefix match:
                # e.g. src_name='IMOU_CAM1_2026' → matches key 'imou_cam1'
                src_lower = src_name.lower().replace("-", "_")
                cam_keys = [k for k in d if not k.startswith("_") and k != "default"]
                # Sort by length descending (most specific match wins)
                for key in sorted(cam_keys, key=len, reverse=True):
                    if src_lower.startswith(key.lower().replace("-", "_")):
                        c.update(d[key])
                        print(f"  ⚙️  Config '{key}' appliquée pour {src_name}")
                        break
                staff_v8_conf = float(c.get("staff_v8_conf", 0.50))
                print(f"  ⚙️  staff_v8_conf={staff_v8_conf:.2f} pour {src_name}")
            except Exception as e:
                print(f"  ⚠️  camera_config.json erreur: {e}")

        # Elimination zones
        elimination_zones = []
        ez_path = BASE / "elimination_zones.json"
        if ez_path.exists():
            try:
                data = json.loads(ez_path.read_text())
                elimination_zones = data.get(src_name, [])
                if elimination_zones:
                    print(f"  🚫 {self.label}: {len(elimination_zones)} zone(s) d'élimination")
            except Exception:
                pass

        def in_elim(cx, cy2):
            return any(z[0] <= cx <= z[2] and z[1] <= cy2 <= z[3] for z in elimination_zones)

        tracker_cfg = str(BYTETRACK_CFG) if BYTETRACK_CFG.exists() else "bytetrack.yaml"
        frame_count = 0
        last_push   = 0.0
        table_client_since: dict[int, float] = {}  # table_idx → timestamp client arrived
        table_person_first: dict[int, float] = {}  # table_idx → first time person seen near
        CLIENT_CONFIRM_S = 10.0  # seconds near table before counting as seated

        if self.show_gui:
            cv2.namedWindow(self.label, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.label, 1280, 720)

        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_rtsp:
                    time.sleep(0.1)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                trackers.clear()
                continue

            frame_count += 1
            now_t = time.time()

            # ── Step 1: Person tracking ───────────────────────────────────
            person_res = self.models["person"].track(
                frame, conf=0.25, persist=True,
                tracker=tracker_cfg, verbose=False, classes=[0]
            )[0]

            # ── Step 2: Staff v8 detection ───────────────────────────────
            v8_boxes = []
            if self.models["staff_v8"] is not None:
                v8_res = self.models["staff_v8"](frame, conf=staff_v8_conf, verbose=False)[0]
                if v8_res.boxes is not None:
                    for b in v8_res.boxes:
                        v8_boxes.append(list(map(int, b.xyxy[0])))

            # ── Step 3: Table detection (every 5 frames) ──────────────────
            if self.models["table"] is not None and frame_count % 5 == 0:
                cached_tables = []
                t_res = self.models["table"](frame, conf=0.50, verbose=False)[0]
                if t_res.boxes is not None:
                    for b in t_res.boxes:
                        cached_tables.append(list(map(int, b.xyxy[0])))

            # ── Step 4: Build person list + assign v8 ────────────────────
            pboxes, tids = [], []
            if person_res.boxes is not None and person_res.boxes.id is not None:
                for box, tid in zip(person_res.boxes, person_res.boxes.id):
                    pboxes.append(list(map(int, box.xyxy[0])))
                    tids.append(int(tid))

            v8_tids = set()
            for sb in v8_boxes:
                best_iou, best_tid = 0.3, None
                for i, pb in enumerate(pboxes):
                    s = iou(pb, sb)
                    if s > best_iou:
                        best_iou, best_tid = s, tids[i]
                if best_tid is not None:
                    v8_tids.add(best_tid)

            # ── Step 5: Consume face results ──────────────────────────────
            while not face_res_q.empty():
                r_tid, r_name, r_conf = face_res_q.get()
                face_pending.discard(r_tid)
                if r_name:
                    raw_key = next((k for k, v in WS_NAME_MAP.items() if v == r_name), r_name)
                    if r_tid not in id_locked:
                        id_votes[r_tid].append(("FACE", raw_key, r_conf))
                    elif r_tid in id_to_name and id_to_name[r_tid][0] != raw_key:
                        id_reverify[r_tid].append(False)
                    elif r_tid in id_to_name:
                        id_reverify[r_tid].append(True)
                    recent = id_reverify.get(r_tid, [])
                    if len(recent) >= 3 and sum(1 for x in recent[-3:] if x is False) >= 2:
                        id_locked.discard(r_tid)
                        id_to_name.pop(r_tid, None)
                        id_votes[r_tid] = []
                        id_reverify[r_tid] = []

            current_staff_boxes = {}
            current_client_count = 0  # clients détectés dans ce frame
            # Table occupancy: track_id persons near each table
            table_person_map: dict[int, list] = {i: [] for i in range(len(cached_tables))}

            # ── Step 6: Per-person logic ──────────────────────────────────
            for pb, tid in zip(pboxes, tids):
                px1, py1, px2, py2 = pb
                cx = (px1+px2)//2
                cy = (py1+py2)//2
                if in_elim(cx, py2):
                    continue
                w, h = px2-px1, py2-py1
                is_v8 = tid in v8_tids
                at_table = any(near_table(pb, tb) for tb in cached_tables)
                trackers[tid].update(cx, cy, w, h, is_v8, at_table, fw, fh)
                label, combined, *_ = trackers[tid].get_label()
                last_pos[tid] = (cx, cy)

                # ── Track table occupancy from person positions ───────────
                for t_idx, t_box in enumerate(cached_tables):
                    if near_table(pb, t_box):
                        table_person_map[t_idx].append((tid, label))

                # ── Count clients (non-staff) ────────────────────────────
                if self.force_staff:
                    # force_staff = ALL detected persons are staff candidates
                    # Skip the is_standing filter entirely — overhead cameras
                    # make people look wide (low h/w ratio) which would wrongly
                    # mark standing staff as "not standing"
                    pass  # everyone goes to staff tracking below
                elif label != "staff":
                    current_client_count += 1
                    # Draw client (thin yellow) in GUI
                    if self.show_gui:
                        col_c = (255, 200, 50)
                        cv2.rectangle(frame, (px1,py1), (px2,py2), col_c, 1)
                        cv2.putText(frame, f"Client",
                                    (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_c, 1)
                    continue

                current_staff_boxes[tid] = pb

                # ArcFace request (async)
                if (self.models["arcface"] and tid not in id_locked
                        and tid not in face_pending
                        and now_t - id_face_last.get(tid, 0) > FACE_INTERVAL
                        and w > 40 and h > 80):
                    fh_crop = int((py2-py1)*FACE_CROP_TOP)
                    crop = frame[py1:py1+fh_crop, px1:px2]
                    if crop.size > 0 and crop.shape[0] > 20:
                        try:
                            face_req_q.put_nowait((tid, crop.copy()))
                            face_pending.add(tid)
                            id_face_last[tid] = now_t
                        except queue.Full:
                            pass

                # Body classifier (sync)
                if (self.models["classifier"] and tid not in id_locked
                        and now_t - id_body_last.get(tid, 0) > BODY_INTERVAL):
                    body_crop = frame[py1:py2, px1:px2]
                    if body_crop.size > 0 and body_crop.shape[0] > 30:
                        taken = {id_to_name[t][0] for t in id_locked if t in id_to_name and t != tid}
                        fname, fsim = identify_by_body(self.models, body_crop)
                        id_body_last[tid] = now_t
                        if fname:
                            raw_key = next((k for k, v in WS_NAME_MAP.items() if v == fname), fname)
                            if raw_key not in taken:
                                id_votes[tid].append(("BODY", raw_key, fsim))

                # Vote processing
                votes = id_votes.get(tid, [])
                if votes and tid not in id_locked:
                    taken = {id_to_name[t][0] for t in id_locked if t in id_to_name and t != tid}
                    weighted = {}
                    last_conf = {}
                    for src, vname, vc in votes:
                        if vname not in taken:
                            weighted[vname] = weighted.get(vname, 0) + (2 if src == "FACE" else 1)
                            last_conf[vname] = vc
                    if weighted:
                        top = max(weighted, key=weighted.get)
                        face_v = sum(1 for s, n, c in votes if s == "FACE" and n == top)
                        body_v = sum(1 for s, n, c in votes if s == "BODY" and n == top)
                        arc_responded = any(s == "FACE" for s, n, c in votes)
                        # Lock after 1 ArcFace vote or 2 Body votes
                        # 2 body votes × 0.5s interval = ~1s to identify
                        should_lock = (face_v >= 1) or (body_v >= 2 and not arc_responded)
                        if should_lock:
                            id_to_name[tid] = (top, last_conf.get(top, 0.5))
                            id_locked.add(tid)
                            display = WS_NAME_MAP.get(top, top.capitalize())
                            print(f"  ✅ {self.label} #{tid} → {display} ({'ArcFace' if face_v>=2 else 'Body'})")

                # Update server metrics — show staff IMMEDIATELY when PersonTracker
                # detects them, even before body classifier assigns a name.
                # This matches old Python test behavior ('STAFF: 1' appears right away).
                if tid in id_locked and tid in id_to_name:
                    # Named staff → use real name
                    raw_key = id_to_name[tid][0]
                    display = WS_NAME_MAP.get(raw_key, raw_key.capitalize())
                elif trackers[tid].locked_as_staff:
                    # Staff detected but not yet named → show as placeholder
                    display = "Serveur"
                else:
                    display = None

                if display:
                    if display not in name_color_idx:
                        name_color_idx[display] = len(name_color_idx)
                    color = COLOR_CYCLE[name_color_idx[display] % len(COLOR_CYCLE)]
                    if display not in server_metrics:
                        server_metrics[display] = ServerMetrics(display, color, self.label)
                    svgx, svgy = px_to_svg(cx, cy, fw, fh)
                    zone_idx = int(cx / fw * len(ZONES)) % len(ZONES)
                    server_metrics[display].tick(
                        svgx, svgy, trackers[tid].speed(),
                        trackers[tid].is_standing(), at_table, zone_idx
                    )

                # Draw GUI
                if self.show_gui:
                    col = (0, 255, 0) if tid in id_locked else (100, 255, 100)
                    cv2.rectangle(frame, (px1,py1), (px2,py2), col, 2)
                    if tid in id_locked and tid in id_to_name:
                        display = WS_NAME_MAP.get(id_to_name[tid][0], id_to_name[tid][0])
                        cv2.putText(frame, f"{display} {combined:.0%}",
                                    (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                    else:
                        cv2.putText(frame, f"STAFF#{tid} ({len(id_votes.get(tid,[]))}/v)",
                                    (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            # ── Track recovery (crossing-aware) ───────────────────────────
            current_tids = set(current_staff_boxes.keys())
            # Detect crossings
            staff_pos = {t: ((b[0]+b[2])//2, (b[1]+b[3])//2)
                         for t, b in current_staff_boxes.items()}
            pairs = [(ta, tb) for ta in staff_pos for tb in staff_pos if ta < tb]
            for ta, tb in pairs:
                if math.hypot(staff_pos[ta][0]-staff_pos[tb][0],
                              staff_pos[ta][1]-staff_pos[tb][1]) < 80:
                    crossing_flag[ta] = now_t
                    crossing_flag[tb] = now_t
            # Release lost
            for lt in list(id_locked):
                if lt not in current_tids and lt in id_to_name:
                    if now_t - recov_cool.get(lt, 0) < 5.0:
                        continue
                    name_l = id_to_name[lt][0]
                    if lt in last_pos and name_l not in lost_tracks:
                        was_x = (now_t - crossing_flag.get(lt, 0)) < 3.0
                        lost_tracks[name_l] = (*last_pos[lt], now_t, id_to_name[lt][1], was_x)
                    id_locked.discard(lt)
                    id_to_name.pop(lt, None)
            # Recover new
            assigned = set()
            for new_tid in sorted(current_tids):
                if new_tid in id_locked or new_tid in id_to_name:
                    continue
                if (now_t - crossing_flag.get(new_tid, 0)) < 2.0:
                    continue
                nb = current_staff_boxes[new_tid]
                ncx, ncy = (nb[0]+nb[2])//2, (nb[1]+nb[3])//2
                best_name, best_dist = None, RECOVER_DIST
                for lname, ldata in lost_tracks.items():
                    if lname in assigned:
                        continue
                    lx, ly, lt2, lc, was_x = ldata
                    if now_t - lt2 > NAME_MEMORY:
                        continue
                    dlim = 40 if was_x else RECOVER_DIST
                    d = math.hypot(ncx-lx, ncy-ly)
                    if d < min(best_dist, dlim):
                        best_dist, best_name = d, (lname, lc)
                if best_name:
                    rname, rconf = best_name
                    taken = any(id_to_name.get(t, (None,))[0] == rname
                                for t in current_tids if t != new_tid)
                    if not taken and rname not in assigned:
                        id_to_name[new_tid] = (rname, rconf)
                        id_locked.add(new_tid)
                        recov_cool[new_tid] = now_t
                        lost_tracks.pop(rname, None)
                        assigned.add(rname)
                        id_face_last[new_tid] = 0
                        id_reverify[new_tid] = []
                        print(f"  🔄 {self.label} #{new_tid} → {WS_NAME_MAP.get(rname, rname)} (récupéré)")
            # Clean expired
            lost_tracks  = {n: v for n, v in lost_tracks.items()  if now_t - v[2] < NAME_MEMORY}
            crossing_flag = {t: ts for t, ts in crossing_flag.items() if now_t - ts < 5.0}

            # ── Push metrics to aggregator ────────────────────────────────
            if now_t - last_push >= PUSH_INTERVAL:
                last_push = now_t
                # ── Build payloads: CURRENTLY VISIBLE tracks only ────────────
                # Trust FAST_LOCK (2 v8 hits at conf≥0.50) directly — no extra
                # stability filter needed. The conf threshold is the real guard.
                active_display_names: set[str] = set()
                for tid in id_locked:
                    if tid in id_to_name and tid in current_staff_boxes:
                        raw = id_to_name[tid][0]
                        active_display_names.add(WS_NAME_MAP.get(raw, raw.capitalize()))
                # NOTE: unnamed locked tracks ("Serveur" placeholder) are intentionally
                # NOT added to active_display_names. Reason: if the system cannot name
                # a person, it cannot compute their efficiency score, reactivity, etc.
                # Only Rami / Mohamed Ali / Sadio have profiles → only they appear as
                # server cards. Unnamed tracks are still counted correctly for client
                # counting (they reduce current_client_count but don't inflate staff count).

                # Track video position for alert sequences
                video_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if not is_rtsp else 0.0

                # Push ALL known servers (active + en_pause) — cards persist all day
                payloads = {}
                for name, m in server_metrics.items():
                    is_active = name in active_display_names
                    if is_active:
                        status = "active"
                        cur_frame = frame
                    else:
                        status = "en_pause"
                        cur_frame = None
                        m.mark_away()   # track time since last seen
                    payloads[name] = m.to_payload(
                        video_pos_sec,
                        current_frame=cur_frame,
                        status=status,
                    )

                # ── Build REAL table data from detected tables ────────────
                table_data = []
                if cached_tables:
                    # Use actual detected table positions
                    for t_idx, (tx1, ty1, tx2, ty2) in enumerate(cached_tables):
                        tcx = (tx1+tx2)//2; tcy = (ty1+ty2)//2
                        svgx_t = max(3.0, min(97.0, (tcx/max(fw,1))*96+2))
                        svgy_t = max(10.0, min(90.0, (tcy/max(fh,1))*80+10))
                        persons_near = table_person_map.get(t_idx, [])
                        staff_near = any(lbl == "staff" for _, lbl in persons_near)
                        has_person  = len(persons_near) > 0
                        if staff_near:
                            st = "visited"; wait_m = 0
                            # Staff arrived → reset all timers for this table
                            table_client_since.pop(t_idx, None)
                            table_person_first.pop(t_idx, None)
                        elif has_person:
                            # Track first time a non-staff person is near this table
                            if t_idx not in table_person_first:
                                table_person_first[t_idx] = now_t
                            time_near = now_t - table_person_first[t_idx]

                            if time_near >= CLIENT_CONFIRM_S:
                                # Person has been near table ≥10s → confirmed seated/waiting
                                st = "occupied"
                                if t_idx not in table_client_since:
                                    table_client_since[t_idx] = table_person_first[t_idx]
                                wait_m = round((now_t - table_client_since[t_idx]) / 60, 1)
                            else:
                                # Still confirming (passerby?) → show as free for now
                                st = "free"; wait_m = 0
                        else:
                            st = "free"; wait_m = 0
                            # Table empty → clear all timers
                            table_client_since.pop(t_idx, None)
                            table_person_first.pop(t_idx, None)
                        table_data.append({
                            "id": t_idx+1,
                            "x": round(svgx_t, 1),
                            "y": round(svgy_t, 1),
                            "status": st,
                            "waitMinutes": wait_m,
                        })
                else:
                    # Fallback: TABLE_GRID with staff proximity
                    for tid2, sx, sy in TABLE_GRID:
                        srv_near = any(
                            math.hypot(m._svgx - sx, m._svgy - sy) < 12
                            for m in server_metrics.values()
                        )
                        table_data.append({
                            "id": tid2, "x": sx, "y": sy,
                            "status": "visited" if srv_near else "free",
                            "waitMinutes": 0,
                        })

                # ── Compute global counters ──────────────────────────────
                tables_occupied = sum(1 for t in table_data
                                      if t["status"] in ("occupied", "waiting"))
                tables_visited  = sum(1 for t in table_data if t["status"] == "visited")
                avg_wait_mins   = 0.0
                waiting = [t for t in table_data if t["status"] == "waiting" and t.get("waitMinutes",0) > 0]
                if waiting:
                    avg_wait_mins = round(sum(t["waitMinutes"] for t in waiting) / len(waiting), 1)

                with _global_lock:
                    # Stamp each server with current time for stale detection
                    now_stamp = time.time()
                    for name, payload in payloads.items():
                        payload["_last_seen"] = now_stamp

                    # ── Remove ghost staff: purge THIS camera's old entries ──
                    # If a person changed identity (e.g., Sadio → Mohamed Ali on
                    # the same track), both names would coexist. We fix this by
                    # removing all entries that were sourced from this camera but
                    # are no longer in the current active payloads.
                    active_names_this_cam = set(payloads.keys())
                    to_remove = [
                        name for name, data in _all_servers.items()
                        if data.get("camera") == self.label        # belongs to this camera
                        and name not in active_names_this_cam      # but no longer active here
                    ]
                    for ghost_name in to_remove:
                        del _all_servers[ghost_name]
                        print(f"  👻 Ghost supprimé: {ghost_name} (remplacé sur {self.label})")

                    _all_servers.update(payloads)
                    if table_data:
                        _all_tables.clear()
                        _all_tables.extend(table_data)
                    _cam_clients[self.label] = current_client_count
                    total_clients = sum(_cam_clients.values())
                    _all_global.update({
                        "totalClients":    total_clients,
                        # Tables with actual clients waiting = truly "occupied"
                        "tablesOccupied":  tables_occupied,     # status == "occupied" or "waiting"
                        # Tables visited by staff (coverage metric, not "occupied")
                        "tablesServed":    tables_visited,      # status == "visited"
                        "totalTables":     max(len(table_data), 15),
                        "avgWaitTime":     avg_wait_mins,
                        "serviceStart":    _service_start,
                        "staffCount":      len(server_metrics),
                        "tableData":       table_data,
                    })

                n = len(payloads)   # only NAMED staff (excludes anonymous Serveur)
                avg = round(sum(p["score"] for p in payloads.values()) / max(n, 1)) if n else 0
                print(f"  📊 {self.label}: {n} staff | {current_client_count} clients | {len(cached_tables)} tables | score={avg} | {datetime.now():%H:%M:%S}", end="\r")

            # ── GUI display ───────────────────────────────────────────────
            if self.show_gui:
                for tb in cached_tables:
                    cv2.rectangle(frame, (tb[0],tb[1]), (tb[2],tb[3]), (0,200,0), 1)
                cv2.putText(frame, f"STAFF:{len(current_staff_boxes)} | Q=quit",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.imshow(self.label, cv2.resize(frame, (1280,720)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    _stop_event.set()
                    break

        cap.release()
        face_req_q.put(None)
        if self.show_gui:
            cv2.destroyWindow(self.label)
        print(f"\n  ⏹  {self.label} terminé.")


# ══════════════════════════════════════════════════════════════════════════════
# Publisher — pushes _all_servers to ws_dashboard_server._shared every 1s
# ══════════════════════════════════════════════════════════════════════════════
def publisher_thread():
    print("📡 Publisher démarré — push vers WS toutes les secondes")
    WS_SHARED["cameras_live"] = True
    WS_SHARED["live_since"]   = datetime.now().strftime("%H:%M:%S")
    TABLE_ALERT_MIN = 5.0   # alert when table unserved for > 5 minutes
    while not _stop_event.is_set():
        time.sleep(PUSH_INTERVAL)

        # ── Purge staff not seen for > STAFF_STALE_TIMEOUT seconds ──────────
        now_ts = time.time()
        with _global_lock:
            stale = [name for name, data in _all_servers.items()
                     if now_ts - data.get("_last_seen", now_ts) > STAFF_STALE_TIMEOUT]
            for name in stale:
                del _all_servers[name]
                print(f"  ⏰ {name} retiré (inactif depuis >{STAFF_STALE_TIMEOUT:.0f}s)")
            servers     = dict(_all_servers)
            tables      = list(_all_tables)
            global_info = dict(_all_global)

        # ── Table alerts: flag tables unserved for too long ──────────────────
        global_alerts = []
        now_str = datetime.now().strftime("%H:%M")
        for tbl in tables:
            wait_m = tbl.get("waitMinutes", 0)
            if tbl.get("status") == "occupied" and wait_m >= TABLE_ALERT_MIN:
                global_alerts.append({
                    "id":      f"tbl-alert-{tbl['id']}",
                    "type":    "critical" if wait_m >= 10 else "warning",
                    "message": f"Table {tbl['id']} non servie depuis {int(wait_m)} min",
                    "time":    now_str,
                })

        WS_SHARED["cameras_live"]   = True
        WS_SHARED["last_update"]    = now_ts
        WS_SHARED["servers"]        = servers
        WS_SHARED["tables"]         = {t["id"]: t for t in tables}
        WS_SHARED["global_alerts"]  = global_alerts   # new: table-level alerts

        ws_global = WS_SHARED.setdefault("global", {})
        ws_global["serviceStart"]   = _service_start
        ws_global["totalClients"]   = global_info.get("totalClients", 0)
        ws_global["tablesOccupied"] = global_info.get("tablesOccupied", 0)
        ws_global["totalTables"]    = global_info.get("totalTables", 15)
        ws_global["avgWaitTime"]    = global_info.get("avgWaitTime", 0)
        ws_global["staffCount"]     = global_info.get("staffCount", len(servers))
    print("📡 Publisher arrêté.")
    WS_SHARED["cameras_live"] = False






# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet de détection → dashboard en temps réel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--live",  action="store_true",   help="Toutes les 6 caméras RTSP en live")
    src_group.add_argument("--cam",   type=str, default=None, help="Une caméra RTSP (ex: imou_cam1)")
    src_group.add_argument("--video", type=str, default=None, help="Fichier vidéo local")
    parser.add_argument("--gui",         action="store_true",  help="Afficher les fenêtres OpenCV")
    parser.add_argument("--cams",        type=str, default=None, help="Sélection caméras ex: 1,3,5")
    parser.add_argument("--force-staff", action="store_true",  help="Traiter toutes les personnes debout comme staff (utile si staff_v8 ne détecte rien)")
    args = parser.parse_args()

    print("=" * 60)
    print("🍴 Kitchen — Pipeline de détection complet")
    print("   Modèles: Person + Staff v8 + Tables + ArcFace + Classifier")
    print("=" * 60)
    print("📦 Chargement des modèles…")

    is_live = bool(args.live or args.cam)
    num_live_cams = len(RTSP_STREAMS) if args.live else (1 if args.cam else 0)
    models = load_models(is_live_mode=is_live, num_cams=num_live_cams)
    print()

    # Build source list
    sources: list[tuple[str, str]] = []

    if args.live:
        cam_keys = list(RTSP_STREAMS.keys())
        if args.cams:
            sel = args.cams.split(",")
            cam_keys = [f"imou_cam{s.strip()}" for s in sel if f"imou_cam{s.strip()}" in RTSP_STREAMS]
        sources = [(RTSP_STREAMS[k], k.upper().replace("_", "-")) for k in cam_keys]
        print(f"🎥 Mode LIVE — {len(sources)} caméra(s): {[s[1] for s in sources]}")

    elif args.cam:
        if args.cam not in RTSP_STREAMS:
            print(f"❌ Caméra inconnue: {args.cam}")
            print(f"   Disponibles: {list(RTSP_STREAMS.keys())}")
            return
        sources = [(RTSP_STREAMS[args.cam], args.cam.upper().replace("_", "-"))]
        print(f"🎥 Caméra: {args.cam}")

    elif args.video:
        sources = [(args.video, Path(args.video).stem[:14].upper())]
        print(f"🎬 Vidéo: {args.video}")

    else:
        # No source → server mode: start WS and wait for dashboard commands
        sources = []
        print("🖥️  Mode serveur — en attente de commandes depuis le dashboard")

    # ── Start embedded WebSocket server (same process = shared _shared dict) ──
    import ws_dashboard_server as _wss
    import asyncio

    def _run_ws_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_wss.main_server())

    ws_thread = threading.Thread(target=_run_ws_server, daemon=True, name="ws-server")
    ws_thread.start()
    print("🌐 Serveur WebSocket démarré en thread (ws://0.0.0.0:8765)")

    force_staff = args.force_staff
    if force_staff:
        print("⚡ Mode --force-staff activé : toutes les personnes debout seront traitées comme staff")

    # NOTE: _force_staff_video was removed — setting force_staff=True on video
    # files from dashboard was causing 0 clients (ALL persons treated as staff
    # candidates, bypassing client counting). Use --force-staff flag explicitly.

    # ── Main restart loop — supports hot video switching from dashboard ──────
    current_sources = sources
    while True:
        # Clear stale data from previous run
        _stop_event.clear()
        with _global_lock:
            _all_servers.clear()
            _all_tables.clear()
            _cam_clients.clear()    # ← clear per-camera client counts (prevents ghost clients)
            _all_global.clear()     # ← clear aggregated global stats
        WS_SHARED["servers"] = {}
        WS_SHARED["tables"]  = {}
        WS_PENDING["path"]   = None   # clear any pending request

        # Start publisher
        pub = threading.Thread(target=publisher_thread, daemon=True)
        pub.start()

        # Start camera/video threads
        threads = []
        for src, lbl in current_sources:
            t = CameraThread(src, lbl, models, show_gui=args.gui, force_staff=force_staff)
            t.start()
            threads.append(t)
            time.sleep(0.5)

        print(f"\n▶  Analyse démarrée sur {[s[1] for s in current_sources]}\n")

        _switch_source = False

        try:
            while any(t.is_alive() for t in threads):
                # Check for video switch request from dashboard
                pending = WS_PENDING.get("path")
                if pending and pending != "__STOP__":
                    print(f"\n🔄 Changement de source → {pending}")
                    _stop_event.set()
                    # Wait up to 20s — GPU inferences can exceed the NMS 2s limit
                    for t in threads:
                        t.join(timeout=20)
                    alive = [t for t in threads if t.is_alive()]
                    if alive:
                        print(f"⚠️  {len(alive)} thread(s) toujours actif(s) après 20s (GPU occupé)")
                    # Free GPU VRAM before starting video thread
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        print("🧹 Cache GPU libéré")
                    except Exception:
                        pass
                    time.sleep(2)   # brief grace period before video thread starts
                    WS_PENDING["path"] = None
                    vid_label = Path(pending).stem[:16].upper()
                    current_sources = [(pending, vid_label)]
                    # ⚠️ Do NOT set force_staff=True here — it causes 0 clients
                    # force_staff=True means ALL persons are staff candidates
                    # which bypasses client counting entirely
                    force_staff = args.force_staff  # respect --force-staff flag only
                    print(f"✅ Analyse vidéo: {vid_label}  (force_staff={force_staff})")
                    _switch_source = True
                    break  # restart outer while loop via continue below
                elif pending == "__STOP__":
                    print("\n⏹  Stop demandé depuis le dashboard")
                    _stop_event.set()
                    for t in threads:
                        t.join(timeout=20)
                    try:
                        import torch; torch.cuda.empty_cache()
                    except Exception:
                        pass
                    WS_PENDING["path"] = None
                    current_sources = sources
                    force_staff = args.force_staff
                    _switch_source = True
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n⏹  Arrêt demandé…")
            _stop_event.set()
            for t in threads:
                t.join(timeout=10)
            break  # exit outer while

        if _switch_source:
            # ← Don't let the else clause below execute for source switches.
            # Just continue the outer while loop to restart with new sources.
            continue

        # ── Threads finished normally (video ended, no switch requested) ──
        pending = WS_PENDING.get("path")
        if pending and pending not in (None, "__STOP__"):
            vid_label = Path(pending).stem[:16].upper()
            current_sources = [(pending, vid_label)]
            # outer while restarts with new video
        else:
            # Video finished — wait for new command (do NOT auto-restart live cameras)
            print("\n📼 Vidéo terminée — en attente d'une nouvelle vidéo ou commande STOP")
            WS_SHARED["servers"] = {}
            WS_SHARED["tables"]  = {}
            while True:
                pending = WS_PENDING.get("path")
                if pending:
                    if pending == "__STOP__":
                        WS_PENDING["path"] = None
                        current_sources = sources
                        force_staff = args.force_staff
                    else:
                        vid_label = Path(pending).stem[:16].upper()
                        current_sources = [(pending, vid_label)]
                        force_staff = True
                    WS_PENDING["path"] = None
                    break
                time.sleep(1)



    # Final cleanup
    WS_SHARED["servers"] = {}
    WS_SHARED["tables"]  = {}
    print("✅ Terminé.")


if __name__ == "__main__":
    main()
