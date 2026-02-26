"""
pipeline_dashboard.py
=====================
Pipeline unifié : Détection Staff → Identification → Dashboard

Combine :
  - ByteTrack + heuristiques (posture, mouvement, uniforme noir)  ← staff_tracker_pro.py
  - ArcFace (DeepFace) + body classifier YOLO                     ← video_to_dashboard.py
  - Push WebSocket vers le dashboard React                        ← ws_dashboard_server.py

Usage :
  # Vidéo fichier
  .\\venv_new\\Scripts\\python.exe pipeline_dashboard.py --video monfichier.mp4

  # Caméra live unique
  .\\venv_new\\Scripts\\python.exe pipeline_dashboard.py --cam imou_cam1

  # Toutes les 6 caméras live
  .\\venv_new\\Scripts\\python.exe pipeline_dashboard.py --live

  # Avec fenêtre OpenCV de debug
  .\\venv_new\\Scripts\\python.exe pipeline_dashboard.py --video monfichier.mp4 --gui
"""

import cv2
import math
import time
import json
import queue
import random
import threading
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

# ─── Chargement modèles YOLO ────────────────────────────────────────────────
from ultralytics import YOLO

# ─── WebSocket shared state ──────────────────────────────────────────────────
try:
    from ws_dashboard_server import _shared as WS_SHARED
    print("✅ ws_dashboard_server importé — push direct vers _shared")
except Exception as e:
    print(f"⚠️  ws_dashboard_server non importé ({e}) — données perdues si serveur absent")
    WS_SHARED = {"servers": {}, "tables": {}, "global": {}, "last_update": 0}

# ─── Répertoire de base ───────────────────────────────────────────────────────
# Pointe vers le dossier contenant les modèles (.pt), face_db.pkl, vidéos, etc.
_SCRIPT_DIR = Path(__file__).parent
BASE = _SCRIPT_DIR.parent / "staff detection images"
if not BASE.exists():
    BASE = _SCRIPT_DIR  # Fallback si lancé depuis le dossier original

# ─── Flux RTSP ───────────────────────────────────────────────────────────────
RTSP_STREAMS = {
    "imou_cam1": "rtsp://admin:admin@100.96.105.67:554/cam/realmonitor?channel=1&subtype=0",
    "imou_cam2": "rtsp://admin:admin@100.96.105.68:554/cam/realmonitor?channel=1&subtype=0",
    "imou_cam3": "rtsp://admin:admin@100.96.105.69:554/cam/realmonitor?channel=1&subtype=0",
    "imou_cam4": "rtsp://admin:admin@100.96.105.70:554/cam/realmonitor?channel=1&subtype=0",
    "imou_cam5": "rtsp://admin:admin@100.96.105.71:554/cam/realmonitor?channel=1&subtype=0",
    "imou_cam6": "rtsp://admin:admin@100.96.105.72:554/cam/realmonitor?channel=1&subtype=0",
}

# ─── Mapping noms ─────────────────────────────────────────────────────────────
WS_NAME_MAP = {"x": "Mohamed Ali", "y": "Rami", "z": "Sadio"}
COLOR_CYCLE  = ["#f97316", "#3b82f6", "#22c55e", "#a855f7", "#ec4899", "#14b8a6"]

# ─── Zones ────────────────────────────────────────────────────────────────────
ZONES = ["Salle principale", "Bar", "Entrée", "Cuisine", "Terrasse", "Caisse"]

# ─── Constantes de détection ─────────────────────────────────────────────────
STANDING_RATIO   = 1.35   # h/w du bounding box → debout si > ce seuil
DARK_THRESH      = 100    # luminosité moyenne uniforme → uniforme sombre si <
MOVEMENT_THRESH  = 40     # px/sec → staff se déplace
MIN_TRACK_TIME   = 3.0    # sec avant de classifier
STAFF_CONF_LOCK  = 0.60   # score lissé pour verrouiller définitivement
FACE_INTERVAL    = 3.0    # sec entre deux tentatives ArcFace par personne
BODY_INTERVAL    = 1.5    # sec entre deux tentatives body-classifier
VOTES_TO_LOCK    = 3      # nombre de votes corps pour verrouiller
PUSH_INTERVAL    = 1.0    # sec entre deux pushs WebSocket
FACE_CROP_TOP    = 0.35   # fraction haute du bounding box pour crop visage

# ─── Partage inter-threads ────────────────────────────────────────────────────
_global_lock  = threading.Lock()
_all_servers : dict[str, dict] = {}
_all_tables  : list[dict]       = []
_service_start = datetime.now().strftime("%H:%M")
_stop_event    = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
# Chargement des modèles
# ══════════════════════════════════════════════════════════════════════════════
def load_models() -> dict:
    models: dict = {}

    # Person tracker (yolov8x → meilleure précision)
    for name in ["yolov8x.pt", "yolov8n.pt"]:
        p = BASE / name
        try:
            models["person"] = YOLO(str(p) if p.exists() else name)
            print(f"  ✅ Person: {name}")
            break
        except Exception:
            pass

    # Staff detector spécialisé
    for name in ["staff_detector_v8.pt", "staff_classifier_yolo.pt"]:
        p = BASE / name
        if p.exists():
            try:
                models["staff_v8"] = YOLO(str(p))
                print(f"  ✅ Staff detector: {name}")
                break
            except Exception:
                pass
    if "staff_v8" not in models:
        models["staff_v8"] = None
        print("  ⚠️  Pas de modèle staff spécialisé — mode heuristiques pur")

    # Classificateur corps
    for name in ["staff_classifier_yolo.pt", "best.pt"]:
        p = BASE / name
        if p.exists():
            try:
                models["classifier"] = YOLO(str(p))
                print(f"  ✅ Body classifier: {name}")
                break
            except Exception:
                pass
    if "classifier" not in models:
        models["classifier"] = None

    # Table detector — même chemin que video_to_dashboard.py
    table_candidates = [
        BASE.parent / "vision-ia-restaurant/runs/detect/kitchen_table_specialist_yolo11x/weights/best.pt",
        BASE / "best.pt",
    ]
    models["table"] = None
    for table_p in table_candidates:
        if table_p.exists():
            try:
                models["table"] = YOLO(str(table_p))
                print(f"  ✅ Table detector: {table_p.name} ({table_p.parent.parent.name})")
                break
            except Exception as e:
                print(f"  ⚠️  Table detector ({table_p}): {e}")
    if models["table"] is None:
        print("  ⚠️  Table detector non trouvé — tables ignorées")

    # Face DB (ArcFace via DeepFace)
    face_db_path = BASE / "face_db.pkl"
    models["face_db"] = None
    models["arcface"] = False
    if face_db_path.exists():
        try:
            import pickle
            with open(face_db_path, "rb") as f:
                db = pickle.load(f)
            models["face_db"] = db
            raw_names = list(db.keys())
            display  = [WS_NAME_MAP.get(k, k.capitalize()) for k in raw_names]
            print(f"  ✅ Face DB: {raw_names} → {display}")

            # Test DeepFace dispo
            from deepface import DeepFace  # noqa: F401
            models["arcface"] = True
            print("  ✅ ArcFace (DeepFace) disponible")
        except Exception as e:
            print(f"  ⚠️  ArcFace désactivé: {e}")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# Identification par corps (body classifier YOLO)
# ══════════════════════════════════════════════════════════════════════════════
def identify_by_body(models: dict, crop: np.ndarray) -> tuple[str | None, float]:
    clf = models.get("classifier")
    if clf is None:
        return None, 0.0
    try:
        res = clf(crop, verbose=False)[0]
        if res.probs is not None:
            idx  = int(res.probs.top1)
            conf = float(res.probs.top1conf)
            name = res.names.get(idx, "")
            if conf > 0.55 and name:
                return name, conf
    except Exception:
        pass
    return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Identification par visage (ArcFace via DeepFace)
# ══════════════════════════════════════════════════════════════════════════════
def face_worker(models: dict, req_q: queue.Queue, res_q: queue.Queue):
    """Thread dédié à l'identification par visage (évite les blocages GPU)"""
    if not models.get("arcface") or models.get("face_db") is None:
        return
    from deepface import DeepFace
    db = models["face_db"]
    while not _stop_event.is_set():
        try:
            tid, crop = req_q.get(timeout=1.0)
        except queue.Empty:
            continue
        best_name, best_sim = None, 0.0
        try:
            emb_obj = DeepFace.represent(crop, model_name="ArcFace",
                                         enforce_detection=False, detector_backend="opencv")
            if emb_obj:
                q_vec = np.array(emb_obj[0]["embedding"])
                for raw_key, ref_embs in db.items():
                    for ref in (ref_embs if isinstance(ref_embs, list) else [ref_embs]):
                        r_vec = np.array(ref)
                        sim = float(np.dot(q_vec, r_vec) /
                                    (np.linalg.norm(q_vec) * np.linalg.norm(r_vec) + 1e-9))
                        if sim > best_sim:
                            best_sim, best_name = sim, raw_key
        except Exception:
            pass
        if best_sim < 0.35:
            best_name = None
        res_q.put((tid, best_name, best_sim))


# ══════════════════════════════════════════════════════════════════════════════
# ServerMetrics — calcul des métriques réelles
# ══════════════════════════════════════════════════════════════════════════════
class ServerMetrics:
    def __init__(self, name: str, color: str, camera: str):
        self.name    = name
        self.color   = color
        self.camera  = camera
        self.arrival = datetime.now().strftime("%H:%M")
        self.history : deque = deque(maxlen=20)

        # Compteurs réels depuis la vidéo
        self.frames       = 0
        self.stand_frames = 0    # Frames debout
        self.move_total   = 0.0  # Déplacement total en px
        self.tables_visited = 0
        self.tables_total   = 15
        self.zone_idx   = 0
        self._svgx      = 50.0
        self._svgy      = 50.0

        # Scores lissés (filtre exponentiel alpha=0.10)
        self._spd_s   = 40.0
        self._react_s = 55.0
        self._cov_s   = 20.0
        self._stand_s = 60.0

        # Historique de positions pour calcul vitesse
        self._pos_hist: deque = deque(maxlen=30)

    def tick(self, cx: float, cy: float, svgx: float, svgy: float,
             is_standing: bool, at_table: bool, speed_px: float, zone_idx: int):
        self.frames += 1
        self._svgx = svgx
        self._svgy = svgy
        self.zone_idx = zone_idx
        self._pos_hist.append((cx, cy))

        if is_standing:
            self.stand_frames += 1
        if at_table:
            self.tables_visited = min(self.tables_visited + 1, self.tables_total)

        # Calcul vitesse réelle depuis l'historique de positions
        if len(self._pos_hist) >= 5:
            pts = list(self._pos_hist)
            speed_px = sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                           for i in range(1, len(pts))) / len(pts)

        # Conversion px/frame → score 0-100 (8px/frame ≈ score 100)
        spd_s     = min(100.0, speed_px * 12.5)
        stand_pct = (self.stand_frames / max(self.frames, 1)) * 100
        cov_s     = (self.tables_visited / self.tables_total) * 100
        # Réactivité : basée sur la régularité du mouvement (-1 = stationnaire = mauvais)
        react_s   = min(100.0, 50 + (spd_s - 40) * 0.5)

        a = 0.10
        self._spd_s   = (1-a)*self._spd_s   + a*spd_s
        self._react_s = (1-a)*self._react_s + a*react_s
        self._cov_s   = (1-a)*self._cov_s   + a*cov_s
        self._stand_s = (1-a)*self._stand_s + a*stand_pct

    def to_payload(self) -> dict:
        score = max(0, min(100, round(
            self._spd_s   * 0.30 +
            self._react_s * 0.30 +
            self._cov_s   * 0.25 +
            self._stand_s * 0.15
        )))
        now_str = datetime.now().strftime("%H:%M")
        self.history.append({"time": now_str, "value": score})

        arr_h, arr_m = map(int, self.arrival.split(":"))
        dur_min  = max(0, datetime.now().hour*60 + datetime.now().minute - arr_h*60 - arr_m)
        avg_resp = round(max(0.5, (100 - self._react_s) / 14), 1)
        stand_pct = round((self.stand_frames / max(self.frames, 1)) * 100)
        cov_pct   = round((self.tables_visited / self.tables_total) * 100)
        spd_label = "Rapide" if self._spd_s >= 60 else ("Normal" if self._spd_s >= 30 else "Lent")

        alerts = []
        if avg_resp > 6:
            alerts.append({"id": f"a-{self.name}-r", "type": "critical",
                           "message": f"Réactivité critique : {avg_resp} min", "time": now_str})
        elif avg_resp > 4:
            alerts.append({"id": f"a-{self.name}-rw", "type": "warning",
                           "message": f"Réactivité faible : {avg_resp} min", "time": now_str})
        if cov_pct < 30:
            alerts.append({"id": f"a-{self.name}-c", "type": "warning",
                           "message": f"Couverture faible : {cov_pct}%", "time": now_str})
        if score < 35:
            alerts.append({"id": f"a-{self.name}-s", "type": "critical",
                           "message": f"Score critique : {score}/100", "time": now_str})

        return {
            "id":              f"srv-{abs(hash(self.name)) % 999 + 1}",
            "name":            self.name,
            "avatar":          "".join(p[0].upper() for p in self.name.split()[:2]),
            "camera":          self.camera,
            "arrivalTime":     self.arrival,
            "serviceDuration": f"{dur_min//60}h {dur_min%60:02d}m",
            "score":           score,
            "speedScore":      round(self._spd_s),
            "reactivityScore": round(self._react_s),
            "coverageScore":   round(self._cov_s),
            "standingScore":   round(self._stand_s),
            "speed":           round(self._spd_s / 20, 2),
            "speedLabel":      spd_label,
            "tablesVisited":   self.tables_visited,
            "totalTables":     self.tables_total,
            "avgResponseTime": avg_resp,
            "standingPercent": stand_pct,
            "recognitionScore": 90,
            "lastZone":        ZONES[self.zone_idx % len(ZONES)],
            "color":           self.color,
            "alerts":          alerts,
            "speedHistory":    list(self.history),
            "activityBySlot":  list(self.history),
            "position":        {"x": round(self._svgx, 1), "y": round(self._svgy, 1)},
            "source":          "live_detection",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers géométriques
# ══════════════════════════════════════════════════════════════════════════════
def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / max(a1 + a2 - inter, 1)

def near_table(pbox, tbox, expand=1.4):
    pcx = (pbox[0]+pbox[2])/2; pcy = (pbox[1]+pbox[3])/2
    tcx = (tbox[0]+tbox[2])/2; tcy = (tbox[1]+tbox[3])/2
    tw  = (tbox[2]-tbox[0]) * expand
    th  = (tbox[3]-tbox[1]) * expand
    return abs(pcx-tcx) < tw/2 and abs(pcy-tcy) < th/2

def px_to_svg(cx, cy, fw, fh) -> tuple[float, float]:
    return round(cx / fw * 100, 1), round(cy / fh * 100, 1)

def is_standing(box) -> bool:
    x1, y1, x2, y2 = box
    h, w = y2-y1, x2-x1
    return (h / max(w, 1)) >= STANDING_RATIO

def has_dark_uniform(frame, box) -> bool:
    """Vérifie si la zone t-shirt de la personne est sombre (uniforme noir)"""
    x1, y1, x2, y2 = box
    h = y2 - y1
    # Zone corps milieu-haut (t-shirt)
    cy1 = int(y1 + h * 0.12)
    cy2 = int(y1 + h * 0.50)
    cx1 = int(x1 + (x2-x1) * 0.15)
    cx2 = int(x1 + (x2-x1) * 0.85)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < DARK_THRESH

def movement_speed(pos_hist: deque) -> float:
    """px/sec depuis l'historique de positions (x, y, timestamp)"""
    if len(pos_hist) < 5:
        return 0.0
    pts = list(pos_hist)
    dist = sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
               for i in range(1, len(pts)))
    dt = pts[-1][2] - pts[0][2]
    return dist / dt if dt > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PersonState — historique complet par track_id
# ══════════════════════════════════════════════════════════════════════════════
class PersonState:
    def __init__(self, tid: int, ts: float):
        self.tid          = tid
        self.first_seen   = ts
        self.last_seen    = ts
        self.total_frames = 0
        self.stand_frames = 0
        self.dark_frames  = 0
        self.positions    : deque = deque(maxlen=60)   # (cx, cy, t)
        self.at_table_hist: deque = deque(maxlen=30)
        self.v8_detections = 0
        self.is_locked    = False
        self.smoothed_score = 0.3

    def duration(self) -> float:
        return self.last_seen - self.first_seen

    def standing_ratio(self) -> float:
        return self.stand_frames / max(self.total_frames, 1)

    def speed(self) -> float:
        return movement_speed(self.positions)

    def staff_score(self) -> float:
        """Score combiné 0→1 pour être staff"""
        dur   = self.duration()
        stand = self.standing_ratio()
        spd   = self.speed()
        dark  = self.dark_frames / max(self.total_frames, 1)

        # Personnes assises > 3s = client
        if dur > 3 and stand < 0.45:
            return 0.05

        s = 0.0
        s += min(1.0, stand * 1.2)       * 0.35   # posture debout → 35%
        s += min(1.0, spd / 80.0)        * 0.30   # vitesse → 30%
        s += min(1.0, dark * 1.5)        * 0.15   # uniforme sombre → 15%
        s += min(1.0, self.v8_detections / 10) * 0.20   # staff_v8 detections → 20%

        return max(0.0, min(1.0, s))

    def update(self, ts: float, box, frame, is_v8: bool, at_table: bool):
        x1, y1, x2, y2 = box
        cx, cy = (x1+x2)/2, (y1+y2)/2
        self.last_seen = ts
        self.total_frames += 1
        self.positions.append((cx, cy, ts))
        if is_v8:
            self.v8_detections += 1
        if is_standing(box):
            self.stand_frames += 1
        if has_dark_uniform(frame, box):
            self.dark_frames += 1
        self.at_table_hist.append(1 if at_table else 0)

        # Score lissé
        raw = self.staff_score()
        alpha = 0.15
        self.smoothed_score = alpha * raw + (1 - alpha) * self.smoothed_score

        # Verrouillage progressif
        if not self.is_locked:
            if self.duration() > 8 and self.smoothed_score >= STAFF_CONF_LOCK:
                self.is_locked = True
            elif self.v8_detections >= 6:
                self.is_locked = True

    def is_staff(self) -> bool:
        dur = self.duration()
        stand = self.standing_ratio()
        # Annuler le lock si la personne s'assoit
        if self.is_locked and dur > 3 and stand < 0.40:
            self.is_locked = False
        if self.is_locked:
            return True
        if dur < MIN_TRACK_TIME:
            return False
        return self.smoothed_score >= 0.45


# ══════════════════════════════════════════════════════════════════════════════
# Publisher thread — pousse vers WS_SHARED chaque seconde
# ══════════════════════════════════════════════════════════════════════════════
def publisher_thread():
    print("📡 Publisher démarré — push vers WS toutes les secondes")
    while not _stop_event.is_set():
        time.sleep(PUSH_INTERVAL)
        with _global_lock:
            servers = dict(_all_servers)
            tables  = list(_all_tables)
        if servers:
            WS_SHARED["servers"]                   = servers
            WS_SHARED["tables"]                    = {t["id"]: t for t in tables}
            WS_SHARED["global"]["serviceStart"]    = _service_start
            WS_SHARED["last_update"]               = time.time()
        else:
            WS_SHARED["servers"] = {}
    print("📡 Publisher arrêté.")


# ══════════════════════════════════════════════════════════════════════════════
# CameraThread — traitement d'une source vidéo
# ══════════════════════════════════════════════════════════════════════════════
TRACKER_CFG = str(BASE / "bytetrack.yaml") if (BASE / "bytetrack.yaml").exists() else "bytetrack.yaml"

class CameraThread(threading.Thread):
    def __init__(self, source: str, label: str, models: dict, show_gui: bool = False):
        super().__init__(daemon=True)
        self.source   = source
        self.label    = label
        self.models   = models
        self.show_gui = show_gui
        self.name     = f"cam-{label}"

    def run(self):
        src = str(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"  ❌ {self.label}: Impossible d'ouvrir {src}")
            return

        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        print(f"  📺 {self.label}: {fw}×{fh} @ {fps:.1f}fps")

        # State par track_id
        persons      : dict[int, PersonState]  = {}
        id_to_name   : dict[int, tuple]        = {}   # tid → (raw_key, conf)
        id_locked    : set[int]                = set()
        id_votes     : dict[int, list]         = defaultdict(list)
        server_metrics: dict[str, ServerMetrics] = {}
        name_color_idx : dict[str, int]         = {}

        # Queues ArcFace async
        face_req_q : queue.Queue = queue.Queue(maxsize=4)
        face_res_q : queue.Queue = queue.Queue()
        face_pending: set[int]   = set()
        id_face_last: dict[int, float] = {}
        id_body_last: dict[int, float] = {}

        # Démarrer thread ArcFace
        ft = threading.Thread(target=face_worker,
                              args=(self.models, face_req_q, face_res_q),
                              daemon=True)
        ft.start()

        frame_count    = 0
        cached_tables  : list = []
        last_log       = time.time()

        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"  ⏹  {self.label}: fin de source")
                break
            frame_count += 1
            now_t = time.time()

            # ── Détection personnes (ByteTrack) ───────────────────────────
            try:
                person_res = self.models["person"].track(
                    frame, conf=0.25, persist=True,
                    tracker=TRACKER_CFG, verbose=False, classes=[0]
                )[0]
            except Exception:
                continue

            # ── Détection staff_v8 (toutes les frames) ───────────────────
            v8_tids: set[int] = set()
            if self.models.get("staff_v8") is not None:
                try:
                    v8_res = self.models["staff_v8"](frame, conf=0.25, verbose=False)[0]
                    if v8_res.boxes is not None:
                        v8_boxes = [list(map(int, b.xyxy[0])) for b in v8_res.boxes]
                        # Associer boxes staff_v8 aux track_ids
                        if person_res.boxes is not None and person_res.boxes.id is not None:
                            pboxes = [list(map(int, b.xyxy[0])) for b in person_res.boxes]
                            ptids  = [int(tid) for tid in person_res.boxes.id]
                            for sb in v8_boxes:
                                best_iou, best_tid = 0.25, None
                                for pb, tid in zip(pboxes, ptids):
                                    s = iou(pb, sb)
                                    if s > best_iou:
                                        best_iou, best_tid = s, tid
                                if best_tid is not None:
                                    v8_tids.add(best_tid)
                except Exception:
                    pass

            # ── Détection tables (1 frame / 5) ────────────────────────────
            if self.models.get("table") is not None and frame_count % 5 == 0:
                try:
                    t_res = self.models["table"](frame, conf=0.45, verbose=False)[0]
                    if t_res.boxes is not None:
                        cached_tables = [list(map(int, b.xyxy[0])) for b in t_res.boxes]
                except Exception:
                    pass

            # ── Consommer résultats ArcFace ───────────────────────────────
            while not face_res_q.empty():
                r_tid, r_name, r_conf = face_res_q.get()
                face_pending.discard(r_tid)
                if r_name and r_tid not in id_locked:
                    id_votes[r_tid].append(("FACE", r_name, r_conf))

            # ── Traitement par personne ───────────────────────────────────
            if person_res.boxes is None or person_res.boxes.id is None:
                continue

            active_tids: set[int] = set()
            for box_obj, tid_t in zip(person_res.boxes, person_res.boxes.id):
                tid   = int(tid_t)
                pb    = list(map(int, box_obj.xyxy[0]))
                px1, py1, px2, py2 = pb
                cx, cy = (px1+px2)//2, (py1+py2)//2
                w, h   = px2-px1, py2-py1

                active_tids.add(tid)

                # Créer ou mettre à jour l'état de la personne
                if tid not in persons:
                    persons[tid] = PersonState(tid, now_t)

                at_table = any(near_table(pb, tb) for tb in cached_tables)
                persons[tid].update(now_t, pb, frame, tid in v8_tids, at_table)

                # ── N'identifier que les STAFF confirmés ─────────────────
                if not persons[tid].is_staff():
                    continue

                # ── ArcFace (async, toutes les FACE_INTERVAL sec) ─────────
                if (self.models.get("arcface") and tid not in id_locked
                        and tid not in face_pending
                        and now_t - id_face_last.get(tid, 0) > FACE_INTERVAL
                        and w > 40 and h > 80):
                    fh_crop = int(h * FACE_CROP_TOP)
                    crop = frame[py1:py1+fh_crop, px1:px2]
                    if crop.size > 0 and crop.shape[0] > 20:
                        try:
                            face_req_q.put_nowait((tid, crop.copy()))
                            face_pending.add(tid)
                            id_face_last[tid] = now_t
                        except queue.Full:
                            pass

                # ── Body classifier (sync, toutes les BODY_INTERVAL sec) ──
                if (self.models.get("classifier") and tid not in id_locked
                        and now_t - id_body_last.get(tid, 0) > BODY_INTERVAL):
                    body_crop = frame[py1:py2, px1:px2]
                    if body_crop.size > 0 and body_crop.shape[0] > 30:
                        taken = {id_to_name[t][0] for t in id_locked
                                 if t in id_to_name and t != tid}
                        fname, fsim = identify_by_body(self.models, body_crop)
                        id_body_last[tid] = now_t
                        if fname and fname not in taken:
                            id_votes[tid].append(("BODY", fname, fsim))

                # ── Vote pour verrouiller un nom ──────────────────────────
                votes = id_votes.get(tid, [])
                if votes and tid not in id_locked:
                    taken = {id_to_name[t][0] for t in id_locked
                             if t in id_to_name and t != tid}
                    weighted: dict[str, float] = {}
                    last_conf: dict[str, float] = {}
                    for src, vname, vc in votes:
                        if vname not in taken:
                            weighted[vname] = weighted.get(vname, 0) + (2 if src == "FACE" else 1)
                            last_conf[vname] = vc
                    if weighted:
                        top    = max(weighted, key=weighted.get)
                        face_v = sum(1 for s, n, _ in votes if s == "FACE" and n == top)
                        body_v = sum(1 for s, n, _ in votes if s == "BODY" and n == top)
                        arc    = any(s == "FACE" for s, n, _ in votes)
                        if face_v >= 2 or (body_v >= VOTES_TO_LOCK and not arc):
                            id_to_name[tid] = (top, last_conf.get(top, 0.5))
                            id_locked.add(tid)
                            disp = WS_NAME_MAP.get(top, top.capitalize())
                            method = "ArcFace" if face_v >= 2 else "Body"
                            print(f"  ✅ {self.label} #{tid} → {disp} ({method}, conf={last_conf.get(top,0):.2f})")

                # ── Nom d'affichage ───────────────────────────────────────
                if tid in id_locked and tid in id_to_name:
                    raw_key = id_to_name[tid][0]
                    display = WS_NAME_MAP.get(raw_key, raw_key.capitalize())
                elif tid not in id_locked:
                    # Nom temporaire basé sur l'ordre d'apparition
                    display = f"Staff #{tid}"
                else:
                    continue

                # ── Mise à jour des métriques ─────────────────────────────
                if display not in name_color_idx:
                    name_color_idx[display] = len(name_color_idx)
                color = COLOR_CYCLE[name_color_idx[display] % len(COLOR_CYCLE)]
                if display not in server_metrics:
                    server_metrics[display] = ServerMetrics(display, color, self.label)

                svgx, svgy = px_to_svg(cx, cy, fw, fh)
                zone_idx   = int(cx / fw * len(ZONES)) % len(ZONES)
                speed_px   = persons[tid].speed()
                standing   = persons[tid].standing_ratio() > 0.5

                server_metrics[display].tick(
                    cx, cy, svgx, svgy, standing, at_table, speed_px, zone_idx
                )

                # ── Dessin debug GUI ──────────────────────────────────────
                if self.show_gui:
                    col = (0, 215, 255)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), col, 2)
                    cv2.putText(frame, f"{display} ({persons[tid].smoothed_score:.0%})",
                                (px1, py1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

            # ── Nettoyage des tracks inactifs (>10s) ──────────────────────
            old = [tid for tid, p in persons.items()
                   if tid not in active_tids and now_t - p.last_seen > 10]
            for tid in old:
                del persons[tid]

            # ── Push vers _all_servers ────────────────────────────────────
            payloads = {disp: sm.to_payload() for disp, sm in server_metrics.items()}
            with _global_lock:
                _all_servers.update(payloads)
                # Tables
                _all_tables.clear()
                for i, tb in enumerate(cached_tables):
                    tx1, ty1, tx2, ty2 = tb
                    svgx_, svgy_ = px_to_svg((tx1+tx2)/2, (ty1+ty2)/2, fw, fh)
                    # Vérifier si une personne est près de cette table
                    is_occ = False
                    if person_res.boxes is not None:
                        for b in person_res.boxes:
                            if near_table(list(map(int, b.xyxy[0])), tb):
                                is_occ = True
                                break
                    _all_tables.append({
                        "id": i+1, "x": svgx_, "y": svgy_,
                        "status": "occupied" if is_occ else "free"
                    })

            # ── Log périodique ────────────────────────────────────────────
            if now_t - last_log >= 1.0:
                staff_count = sum(1 for p in persons.values() if p.is_staff())
                avg_score = (sum(sm.to_payload()["score"] for sm in server_metrics.values())
                             / max(len(server_metrics), 1))
                print(f"  📊 {self.label}: {staff_count} staff | "
                      f"score moy={round(avg_score)} | {datetime.now().strftime('%H:%M:%S')}")
                last_log = now_t

            # ── Affichage GUI ─────────────────────────────────────────────
            if self.show_gui:
                cv2.imshow(f"Kitchen — {self.label}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    _stop_event.set()
                    break

        cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        print(f"  ⏹  {self.label} terminé.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Kitchen : Détection staff → Métriques réelles → Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python pipeline_dashboard.py --video mafichier.mp4
  python pipeline_dashboard.py --video mafichier.mp4 --gui
  python pipeline_dashboard.py --live
  python pipeline_dashboard.py --cam imou_cam1
        """
    )
    src_grp = parser.add_mutually_exclusive_group()
    src_grp.add_argument("--live",  action="store_true", help="6 caméras RTSP live")
    src_grp.add_argument("--cam",   type=str, default=None, help="Une caméra (ex: imou_cam1)")
    src_grp.add_argument("--video", type=str, default=None, help="Fichier vidéo")
    parser.add_argument("--gui",  action="store_true", help="Fenêtres OpenCV de debug")
    parser.add_argument("--cams", type=str, default=None, help="Sélection caméras: 1,3,5")
    args = parser.parse_args()

    print("=" * 62)
    print("🍴 Kitchen — Pipeline Complet de Détection + Dashboard")
    print("=" * 62)
    print("📦 Chargement des modèles…")
    models = load_models()
    print()

    # ── Chargement des vrais noms depuis face_db ─────────────────────────
    db_path = BASE / "face_db.pkl"
    if db_path.exists():
        try:
            import pickle
            db = pickle.load(open(db_path, "rb"))
            real_names = [WS_NAME_MAP.get(k, k) for k in db.keys()]
            print(f"✅ Vrais noms : {real_names}")
        except Exception:
            pass

    # ── Construction de la liste de sources ──────────────────────────────
    sources: list[tuple[str, str]] = []

    if args.live:
        cam_keys = list(RTSP_STREAMS.keys())
        if args.cams:
            sel = args.cams.split(",")
            cam_keys = [f"imou_cam{s.strip()}" for s in sel
                        if f"imou_cam{s.strip()}" in RTSP_STREAMS]
        sources = [(RTSP_STREAMS[k], k.upper().replace("_", "-")) for k in cam_keys]
        print(f"🎥 Mode LIVE — {len(sources)} caméra(s) : {[s[1] for s in sources]}")

    elif args.cam:
        if args.cam not in RTSP_STREAMS:
            print(f"❌ Caméra inconnue: {args.cam}")
            return
        sources = [(RTSP_STREAMS[args.cam], args.cam.upper().replace("_", "-"))]
        print(f"🎥 Caméra : {args.cam}")

    elif args.video:
        p = Path(args.video)
        if not p.exists():
            # Chercher dans le répertoire courant
            p = BASE / args.video
        if not p.exists():
            print(f"❌ Fichier vidéo introuvable : {args.video}")
            return
        sources = [(str(p), p.stem[:16].upper())]
        print(f"🎬 Vidéo : {p}")

    else:
        # Dialogue de sélection de fichier
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                title="Choisir une vidéo",
                filetypes=[("Vidéos", "*.mp4 *.avi *.mkv *.mov *.dav"), ("Tous", "*.*")]
            )
            root.destroy()
        except Exception:
            path = input("Chemin de la vidéo : ").strip()
        if not path:
            print("Aucune source sélectionnée.")
            return
        p = Path(path)
        sources = [(str(p), p.stem[:16].upper())]
        print(f"🎬 Vidéo : {p}")

    # ── Démarrage publisher ───────────────────────────────────────────────
    pub = threading.Thread(target=publisher_thread, daemon=True)
    pub.start()

    # ── Démarrage threads caméra ──────────────────────────────────────────
    threads = []
    for src, lbl in sources:
        t = CameraThread(src, lbl, models, show_gui=args.gui)
        t.start()
        threads.append(t)
        time.sleep(0.5)

    print(f"\n▶  Analyse démarrée — Ctrl+C pour quitter\n")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n⏹  Arrêt demandé…")
        _stop_event.set()

    for t in threads:
        t.join(timeout=5)

    WS_SHARED["servers"] = {}
    WS_SHARED["tables"]  = {}
    print("✅ Pipeline terminé.")


if __name__ == "__main__":
    main()
