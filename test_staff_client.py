"""
V8 Model: 3 Models - Person + Staff(v8) + Tables
- YOLO person detector (yolov8x) + tracking
- Staff v8 (YOLOv8m, staff-only, 99.0% mAP50, 90.2% mAP50-95)
- Table model (yolo11x) = table positions
"""
import cv2
import math
import time
import csv
import numpy as np


from datetime import datetime
import json
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, deque
import tkinter as tk
from tkinter import filedialog



# Elimination zones (areas to ignore)
elimination_zones = []  # list of [x1, y1, x2, y2]

def select_elimination_zones(frame, source_name):
    """Let user draw elimination zones on the first frame."""
    global elimination_zones
    
    # Try to load saved zones (source-specific only)
    zones_file = Path("elimination_zones.json")
    if zones_file.exists():
        data = json.loads(zones_file.read_text())
        if source_name in data:
            elimination_zones = data[source_name]
            print(f"  ✅ {len(elimination_zones)} zone(s) chargée(s) pour {source_name}")
            return
    
    print("\n  📌 DESSINEZ LES ZONES A ELIMINER:")
    print("  - Cliquez-glissez pour dessiner un rectangle")
    print("  - Dessinez plusieurs zones si nécessaire")
    print("  - ENTER = confirmer | Z = annuler dernière zone")
    
    clone = frame.copy()
    img_h, img_w = clone.shape[:2]
    win_w, win_h = 1280, 720
    scale_x = img_w / win_w
    scale_y = img_h / win_h
    display_img = cv2.resize(clone, (win_w, win_h))
    zones = []
    drawing = {"active": False, "x1": 0, "y1": 0, "x2": 0, "y2": 0}
    
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            drawing["x1"], drawing["y1"] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            drawing["x2"], drawing["y2"] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["active"] = False
            x1 = min(drawing["x1"], x)
            y1 = min(drawing["y1"], y)
            x2 = max(drawing["x1"], x)
            y2 = max(drawing["y1"], y)
            if abs(x2-x1) > 20 and abs(y2-y1) > 20:
                # Scale to original frame coordinates
                rx1 = int(x1 * scale_x)
                ry1 = int(y1 * scale_y)
                rx2 = int(x2 * scale_x)
                ry2 = int(y2 * scale_y)
                zones.append([rx1, ry1, rx2, ry2])
                print(f"  🚫 Zone #{len(zones)} ajoutée: ({rx1},{ry1})-({rx2},{ry2})")
    
    win = "Zones d'elimination"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, win_w, win_h)
    cv2.setMouseCallback(win, mouse_cb)
    
    while True:
        display = display_img.copy()
        cv2.putText(display, "Dessinez zones a eliminer | ENTER=confirmer | S=skip",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for z in zones:
            # Draw in window coords for visual feedback
            dz = [int(z[0]/scale_x), int(z[1]/scale_y), int(z[2]/scale_x), int(z[3]/scale_y)]
            cv2.rectangle(display, (dz[0], dz[1]), (dz[2], dz[3]), (0, 0, 255), 2)
            cv2.putText(display, "ELIMINER", (dz[0]+5, dz[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if drawing["active"]:
            cv2.rectangle(display, (drawing["x1"], drawing["y1"]),
                         (drawing["x2"], drawing["y2"]), (0, 0, 255), 2)
        cv2.imshow(win, display)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == ord('q'):  # ENTER or Q
            elimination_zones = zones
            if zones:
                data = {}
                if zones_file.exists():
                    data = json.loads(zones_file.read_text())
                data[source_name] = zones
                zones_file.write_text(json.dumps(data, indent=2))
                print(f"  💾 {len(zones)} zone(s) sauvegardée(s) pour {source_name}")
            else:
                print("  Pas de zone - détection partout")
            break
        elif key == ord('z'):  # Undo last zone
            if zones:
                zones.pop()
                print("  ↩ Dernière zone annulée")
    
    cv2.destroyWindow(win)

def in_elimination_zone(cx, cy):
    """Check if point is inside any elimination zone."""
    for z in elimination_zones:
        if z[0] <= cx <= z[2] and z[1] <= cy <= z[3]:
            return True
    return False

STAFF_MODEL_V8 = Path("staff_detector_v8.pt")
PERSON_MODEL = "yolov8x.pt"  # extra-large = best person detection
TABLE_MODEL = Path("../vision-ia-restaurant/runs/detect/kitchen_table_specialist_yolo11x/weights/best.pt")

WINDOW_SIZE = 30
STAFF_THRESHOLD = 0.45

# Correspondance labels internes → vrais noms de serveurs
NAME_MAP = {
    "x": "Mohamed Ali",
    "y": "Rami",
    "z": "Sadio",
}

class PersonTracker:
    def __init__(self):
        self.positions = deque(maxlen=60)
        self.box_ratios = deque(maxlen=WINDOW_SIZE)
        self.at_table_history = deque(maxlen=WINDOW_SIZE)
        self.total_frames = 0
        self.v8_detections = 0
        self.position_changes = 0
        self.last_significant_pos = None
        self.locked_as_staff = False
        self.staff_frames_count = 0
    
    def update(self, cx, cy, w, h, staff_v8, at_table, fw, fh):
        self.total_frames += 1
        self.positions.append((cx, cy))
        
        if staff_v8:
            self.v8_detections += 1
        
        # Table proximity
        self.at_table_history.append(1 if at_table else 0)
        
        # Track position
        if self.last_significant_pos is None:
            self.last_significant_pos = (cx, cy)
        
        # Track posture
        self.box_ratios.append(h / max(w, 1))
        
        # Position changes
        dist = math.sqrt((cx - self.last_significant_pos[0])**2 + 
                         (cy - self.last_significant_pos[1])**2)
        if dist > 50:
            self.position_changes += 1
            self.last_significant_pos = (cx, cy)
    
    def get_movement_speed(self):
        if len(self.positions) < 5:
            return 0
        total_dist = 0
        pos_list = list(self.positions)
        for i in range(1, len(pos_list)):
            dx = pos_list[i][0] - pos_list[i-1][0]
            dy = pos_list[i][1] - pos_list[i-1][1]
            total_dist += math.sqrt(dx*dx + dy*dy)
        return total_dist / len(pos_list)
    
    def is_standing(self):
        if len(self.box_ratios) < 3:
            return True
        # Lower ratio for overhead cameras (sitting still looks tall from above)
        return sum(self.box_ratios) / len(self.box_ratios) > 1.8
    
    def is_at_table(self):
        if len(self.at_table_history) < 3:
            return False
        return sum(self.at_table_history) / len(self.at_table_history) > 0.5
    
    def get_combined_score(self):
        """
        Combined score: 0.0 = client, 1.0 = staff
        Visual model is the GATEKEEPER — if visual < 0.30, always client.
        Visual: 70%, Table: 15%, Behavior: 15%
        """
        if self.total_frames == 0:
            return 0, 0, 0, 0
        
        # Visual score: v8 model
        v8_ratio = self.v8_detections / self.total_frames
        visual_score = min(v8_ratio * 1.2, 1.0)  # Slight boost since v8 is very precise
        
        # GATEKEEPER: Visual must be at least 0.20
        if self.total_frames >= 10 and visual_score < 0.20:
            table_score = 0.5
            behavior_score = 0.5
            return 0.0, visual_score, table_score, behavior_score
        
        # Table proximity score (15%)
        table_score = 0.5  # neutral default
        if len(self.at_table_history) >= 3:
            at_table_ratio = sum(self.at_table_history) / len(self.at_table_history)
            if at_table_ratio > 0.7 and not self.is_standing():
                table_score = 0.05
            elif at_table_ratio > 0.5 and not self.is_standing():
                table_score = 0.15
            elif at_table_ratio > 0.5 and self.is_standing():
                table_score = 0.4
            elif at_table_ratio < 0.2:
                table_score = 0.7
        
        # Behavioral score (15%)
        behavior_score = 0.5
        
        if self.total_frames >= 5:
            if self.is_standing():
                behavior_score += 0.3
            else:
                behavior_score -= 0.2
        
        if self.total_frames >= 10:
            speed = self.get_movement_speed()
            if speed > 5.0:
                behavior_score += 0.3
            elif speed < 1.0:
                behavior_score -= 0.2
        
        if self.total_frames >= 20:
            if self.position_changes >= 3:
                behavior_score += 0.2
            elif self.position_changes == 0:
                behavior_score -= 0.2
        
        behavior_score = max(0, min(1, behavior_score))
        
        # Combined: 70% visual + 15% table + 15% behavior
        combined = visual_score * 0.70 + table_score * 0.15 + behavior_score * 0.15
        
        # At table with very weak visual → likely client
        # Only override if v8 barely detects this person at all
        if self.total_frames >= 15 and self.is_at_table():
            if not self.is_standing() and visual_score < 0.25:
                combined = min(combined, 0.25)
            elif self.is_standing() and visual_score < 0.15:
                combined = min(combined, 0.30)
        
        # FAST LOCK: if v8 detected staff at least 5 times on 10+ frames → force staff
        # (Raised from 3/5 to avoid false positives from a single bad model detection)
        if self.v8_detections >= 5 and self.total_frames >= 10:
            combined = max(combined, 0.60)
            self.locked_as_staff = True
        
        return combined, visual_score, table_score, behavior_score
    
    def get_label(self):
        if self.total_frames < 2:
            return None, 0, 0, 0, 0
        combined, visual, table, behavior = self.get_combined_score()
        
        # Once locked as staff, stay staff
        if self.locked_as_staff:
            return "staff", combined, visual, table, behavior
        
        # Hysteresis: prevent flickering between staff/client
        # To BECOME staff: need >= 0.45
        # To DROP BACK to client: need < 0.30 (sticky once classified)
        if not hasattr(self, '_was_staff'):
            self._was_staff = False
        
        if self._was_staff:
            # Already classified as staff — stay staff unless score drops significantly
            if combined < 0.30:
                self._was_staff = False
                self.staff_frames_count = max(0, self.staff_frames_count - 1)
                return "client", combined, visual, table, behavior
            else:
                self.staff_frames_count += 1
                if self.staff_frames_count >= 20:
                    self.locked_as_staff = True
                return "staff", combined, visual, table, behavior
        else:
            # Not yet classified as staff — need high score to become staff
            if combined >= STAFF_THRESHOLD:
                self._was_staff = True
                self.staff_frames_count += 1
                return "staff", combined, visual, table, behavior
            else:
                self.staff_frames_count = 0
                return "client", combined, visual, table, behavior
    
    def get_debug_info(self):
        s = "S" if self.is_standing() else "s"
        m = "M" if self.get_movement_speed() > 3.0 else "m"
        t = "T" if self.is_at_table() else "."
        return "{}{}{}".format(s, m, t)


# Global state for staff tracking
current_staff_boxes = {}  # {track_id: [x1,y1,x2,y2]}



def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1)

def person_near_table(person_box, table_box, expand=1.3):
    """Check if person center is near table area"""
    pcx = (person_box[0] + person_box[2]) / 2
    pcy = (person_box[1] + person_box[3]) / 2
    tcx = (table_box[0] + table_box[2]) / 2
    tcy = (table_box[1] + table_box[3]) / 2
    tw = (table_box[2] - table_box[0]) * expand
    th = (table_box[3] - table_box[1]) * expand
    return abs(pcx - tcx) < tw/2 and abs(pcy - tcy) < th/2

def select_video():
    root = tk.Tk()
    root.withdraw()
    f = filedialog.askopenfilename(
        title="Select video",
        filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.dav"), ("All", "*.*")]
    )
    root.destroy()
    return f

def load_camera_config(source_name: str) -> dict:
    """Load per-camera parameters from camera_config.json.
    Falls back to 'default' if the source is not found."""
    config_path = Path("camera_config.json")
    defaults = {
        "face_threshold":      0.45,
        "face_crop_top_pct":   0.30,
        "staff_v8_conf":       0.50,
        "person_conf":         0.25,
        "recover_dist_px":     120,
        "name_memory_sec":     20,
        "votes_to_lock":       3,
        "face_check_interval": 3.0,
        "body_check_interval": 1.5,
    }
    if not config_path.exists():
        print("  ⚠️ camera_config.json not found — using defaults")
        return defaults
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        # Merge default → global default → source-specific
        cfg = {**defaults}
        if "default" in data:
            cfg.update({k: v for k, v in data["default"].items() if not k.startswith("_")})
        if source_name in data:
            cfg.update({k: v for k, v in data[source_name].items() if not k.startswith("_")})
            print(f"  ⚙️  Config loaded for '{source_name}'")
        else:
            print(f"  ⚙️  No specific config for '{source_name}' — using defaults")
        print(f"     face_threshold={cfg['face_threshold']}  crop_top={cfg['face_crop_top_pct']:.0%}  "
              f"v8_conf={cfg['staff_v8_conf']}  recover={cfg['recover_dist_px']}px")
        return cfg
    except Exception as e:
        print(f"  ⚠️ camera_config.json parse error: {e} — using defaults")
        return defaults

def test_hybrid(video_path):
    print("Loading 3 models (v8 + person + tables)...")
    staff_v8 = YOLO(str(STAFF_MODEL_V8))
    person_model = YOLO(PERSON_MODEL)
    
    use_tables = TABLE_MODEL.exists()
    table_model = None
    if use_tables:
        table_model = YOLO(str(TABLE_MODEL))
        print("  Table model loaded!")
    else:
        print("  Table model not found, skipping table detection")
    

    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Fallback: OpenCV can't handle Unicode paths on Windows
        import shutil, tempfile
        tmp = Path(tempfile.gettempdir()) / "kitchen_video.mp4"
        print(f"  ⚠️ Chemin Unicode, copie temporaire vers {tmp}...")
        shutil.copy2(video_path, str(tmp))
        cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        print("Cannot open video")
        return
    
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 20
    print("Video: {}x{}, Press Q to quit".format(fw, fh))

    # Load per-source configuration
    if isinstance(video_path, str) and video_path.startswith("rtsp"):
        source_name = video_path.rstrip("/").split("/")[-1]
    else:
        source_name = Path(str(video_path)).stem
    cam_cfg = load_camera_config(source_name)
    
    # Apply configuration values
    FACE_CFG_THRESHOLD    = cam_cfg["face_threshold"]
    FACE_CROP_TOP_PCT     = cam_cfg["face_crop_top_pct"]
    STAFF_V8_CONF         = cam_cfg["staff_v8_conf"]
    RECOVER_DIST_PX       = int(cam_cfg["recover_dist_px"])
    NAME_MEMORY_SEC       = float(cam_cfg["name_memory_sec"])
    VOTES_TO_LOCK_CFG     = int(cam_cfg["votes_to_lock"])
    FACE_CHECK_INTERVAL   = float(cam_cfg["face_check_interval"])
    BODY_CHECK_INTERVAL   = float(cam_cfg["body_check_interval"])
    
    # Video recording disabled
    # out_name = "detection_output_{}.mp4".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(out_name, fourcc, fps_video, (fw, fh))
    # print("📹 Recording to: {}".format(out_name))

    
    win_name = "Server Efficiency Tracker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    trackers = defaultdict(PersonTracker)
    cached_table_boxes = []  # Persist tables between frames
    current_staff_boxes.clear()

    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    # --- Staff Classifier (YOLOv8 classification) ---
    staff_classifier = None
    staff_class_names = None
    yolo_cls_path = Path("staff_classifier_yolo.pt")
    resnet_path = Path("staff_classifier.pt")
    
    if yolo_cls_path.exists():
        staff_classifier = YOLO(str(yolo_cls_path))
        staff_class_names = staff_classifier.names
        print(f"  \U0001f9d1 Staff classifier (YOLO): {staff_class_names}")
        print("  \u2705 Staff classifier loaded")
    elif resnet_path.exists():
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision import models as tv_models
        checkpoint = torch.load(str(resnet_path), map_location="cpu", weights_only=True)
        staff_class_names = {i: n for i, n in enumerate(checkpoint["class_names"])}
        staff_classifier = tv_models.resnet18(weights=None)
        in_features = staff_classifier.fc.in_features
        staff_classifier.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_features, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, checkpoint["num_classes"])
        )
        staff_classifier.load_state_dict(checkpoint["model_state"])
        staff_classifier.eval()
        print(f"  \U0001f9d1 Staff classifier (ResNet): {staff_class_names}")
    else:
        print("  \u26a0\ufe0f No classifier - run: python train_staff_yolo_cls.py")
    
    # --- ArcFace Face DB ---
    face_db = None
    arcface_model = None
    face_db_path = Path("face_db.pkl")
    if face_db_path.exists():
        import pickle
        import sys
        # Use venv_new python for deepface
        venv_python = Path("venv_new/Scripts/python.exe")
        if venv_python.exists():
            try:
                with open(face_db_path, "rb") as f:
                    face_db = pickle.load(f)
                print(f"  \U0001f9d1 Face DB loaded: {list(face_db.keys())} ({sum(v['count'] for v in face_db.values())} embeddings)")
                # Load ArcFace via deepface in subprocess mode — or import directly
                sys.path.insert(0, str(Path("venv_new/Lib/site-packages").resolve()))
                try:
                    from deepface import DeepFace
                    arcface_model = DeepFace
                    print("  \u2705 ArcFace (DeepFace) loaded")
                except Exception as e:
                    print(f"  \u26a0\ufe0f DeepFace import failed: {e}")
            except Exception as e:
                print(f"  \u26a0\ufe0f Face DB load failed: {e}")
    else:
        print("  \u26a0\ufe0f No face DB - run: venv_new\\Scripts\\python build_face_db.py")
    
    # Use per-camera thresholds loaded from camera_config.json
    FACE_SIM_THRESHOLD  = FACE_CFG_THRESHOLD   # set earlier from cam_cfg
    VOTES_TO_LOCK       = VOTES_TO_LOCK_CFG

    id_to_name = {}
    id_votes = defaultdict(list)  # [(src, name, conf), ...]
    id_locked = set()
    # Separate timers for ArcFace (async) and body classifier (sync) — from cam_cfg
    id_face_last_check = defaultdict(float)
    id_body_last_check = defaultdict(float)

    
    # Track recovery: save position of lost locked tracks
    lost_tracks = {}   # {name: (cx, cy, time_lost, conf, was_crossing)}
    last_positions = {}  # {tid: (cx, cy)} — last known position per track
    id_reverify = defaultdict(list)  # {tid: [True/False]}
    crossing_flag = {}   # {tid: timestamp} — recently involved in a crossing event
    recovery_cooldown = {}  # {tid: timestamp} — prevents recover→release→recover loop

    
    # --- Async face recognition thread ---
    import threading, queue
    face_request_queue = queue.Queue(maxsize=6)   # (tid, crop_img) — élargi pour 3+ serveurs simultanés
    face_result_queue  = queue.Queue()             # (tid, name, conf)
    face_pending = set()  # tids currently being processed
    
    def face_worker():
        """Background thread: runs ArcFace without blocking video.

        IMPORTANT: enforce_detection=False is DISABLED.
        When enabled, ArcFace extracts an embedding from ANY image (even no face),
        producing garbage vectors that score 0.97-0.99 against ALL database entries.
        -> Only use strict detection (retinaface, opencv with enforce=True).
        -> If no face detected -> return None -> rely on body classifier.
        """
        MARGIN = 0.05  # 5% gap required between 1st and 2nd best

        while True:
            item = face_request_queue.get()
            if item is None:
                break
            tid_f, crop_f = item
            name_f, conf_f = None, 0
            if arcface_model is not None and face_db is not None:
                try:
                    # STRICT detection only - no enforce_detection=False fallback
                    # (False produces garbage embeddings from non-face crops)
                    emb = None
                    for backend in ["retinaface", "opencv"]:
                        try:
                            result = arcface_model.represent(
                                img_path=crop_f,
                                model_name="ArcFace",
                                detector_backend=backend,
                                enforce_detection=True
                            )
                            if result:
                                raw = np.array(result[0]["embedding"])
                                emb = raw / (np.linalg.norm(raw) + 1e-8)
                                break
                        except Exception:
                            continue
                    # emb is None if no face found -> no vote, body classifier takes over

                    if emb is not None:
                        # Compare against ALL individual embeddings per person
                        # Use best single match (max sim) per person
                        scores = {}
                        for n, data in face_db.items():
                            best = -1.0
                            for db_emb in data.get("embeddings", []):
                                db_vec = np.array(db_emb)
                                norm = np.linalg.norm(db_vec)
                                if norm < 1e-8:
                                    continue
                                s = float(np.dot(emb, db_vec / norm))
                                if s > best:
                                    best = s
                            if best < 0:  # fallback to centroid
                                c = np.array(data["centroid"])
                                best = float(np.dot(emb, c / (np.linalg.norm(c) + 1e-8)))
                            scores[n] = best

                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        if ranked:
                            best_name, best_sim = ranked[0]
                            # MARGIN CHECK: reject if 2nd is too close
                            if len(ranked) >= 2:
                                gap = best_sim - ranked[1][1]
                                if gap < MARGIN:
                                    # Too ambiguous — don't assign any name
                                    best_name = None
                            if best_name and best_sim >= FACE_SIM_THRESHOLD:
                                name_f, conf_f = best_name, best_sim
                except Exception:
                    pass
            face_result_queue.put((tid_f, name_f, conf_f))
            face_pending.discard(tid_f)
            face_request_queue.task_done()
    
    face_thread = threading.Thread(target=face_worker, daemon=True)
    face_thread.start()
    
    def identify_by_face(crop_img):
        """Try to identify staff by face using ArcFace. Returns (name, confidence) or (None, 0)."""
        if arcface_model is None or face_db is None:
            return None, 0
        try:
            # Get face embedding from crop
            result = arcface_model.represent(
                img_path=crop_img,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True
            )
            if not result:
                return None, 0
            emb = np.array(result[0]["embedding"])
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            # Compare to all known faces
            best_name, best_sim = None, -1
            for name, data in face_db.items():
                sim = float(np.dot(emb, data["centroid"]))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            
            if best_sim >= FACE_SIM_THRESHOLD:
                return best_name, best_sim
        except:
            pass
        return None, 0
    
    def identify_by_body(crop_img):
        """Classify staff by body/appearance only (NO face call here — face is async only).
        Returns (name, confidence) or (None, 0)."""
        if staff_classifier is None:
            return None, 0
        try:
            if isinstance(staff_classifier, YOLO):
                results = staff_classifier(crop_img, verbose=False)[0]
                top1_idx = results.probs.top1
                top1_conf = results.probs.top1conf.item()
                # Restored to 0.55 (yesterday's working value — 0.65 was too strict for 55% accuracy classifier)
                if top1_conf >= 0.55:
                    return staff_class_names[top1_idx], top1_conf
            else:
                img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                tensor = T.Compose([
                    T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(staff_classifier(tensor), dim=1)
                    conf, idx = probs.max(1)
                if conf.item() >= 0.55:
                    return staff_class_names[idx.item()], conf.item()
        except:
            pass
        return None, 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # For live streams, just retry; for files, loop
            if isinstance(video_path, str) and video_path.startswith("rtsp"):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            trackers.clear()
            continue
        
        frame_count += 1
        current_staff_boxes.clear()  # Refresh click targets each frame
        
        # Step 1: Detect all people with tracking
        person_results = person_model.track(
            frame, conf=0.25, persist=True,
            tracker="bytetrack_kitchen.yaml", verbose=False, classes=[0]
        )[0]
        
        # Step 2: Detect staff visually (V8) — conf from camera_config.json
        v8_results = staff_v8(frame, conf=STAFF_V8_CONF, verbose=False)[0]
        v8_boxes = []
        if v8_results.boxes is not None:
            for box in v8_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                v8_boxes.append([x1, y1, x2, y2])
        staff_boxes = v8_boxes
        
        # Step 3: Detect tables (every 5 frames, cached between frames)
        if use_tables and frame_count % 5 == 0:
            cached_table_boxes = []
            table_results = table_model(frame, conf=0.50, verbose=False)[0]
            if table_results.boxes is not None:
                for box in table_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cached_table_boxes.append([x1, y1, x2, y2])
        table_boxes = cached_table_boxes
        
        # Draw tables (green)
        for tb in table_boxes:
            cv2.rectangle(frame, (tb[0], tb[1]), (tb[2], tb[3]), (0, 200, 0), 1)
            cv2.putText(frame, "TABLE", (tb[0], tb[1]-3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        
        frame_staff = 0
        frame_client = 0
        
        # Step 4: Pre-compute best staff match per person (EXCLUSIVE assignment)
        # Each staff box is assigned to the person with highest IoU only
        person_boxes_list = []
        person_tids = []
        if person_results.boxes is not None and person_results.boxes.id is not None:
            for box, track_id in zip(person_results.boxes, person_results.boxes.id):
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                person_boxes_list.append([px1, py1, px2, py2])
                person_tids.append(int(track_id))
        
        # For each staff box, find the BEST matching person (highest IoU)
        v8_assigned_tids = set()
        for sb in v8_boxes:
            best_iou = 0.3
            best_tid = None
            for i, pb in enumerate(person_boxes_list):
                score = iou(pb, sb)
                if score > best_iou:
                    best_iou = score
                    best_tid = person_tids[i]
            if best_tid is not None:
                v8_assigned_tids.add(best_tid)
        staff_assigned_tids = v8_assigned_tids
        
        # Step 5: Match people to staff + tables
        for i, (pb, tid) in enumerate(zip(person_boxes_list, person_tids)):
                px1, py1, px2, py2 = pb
                person_box = pb
                cx = (px1 + px2) // 2
                cy = (py1 + py2) // 2
                
                # Skip if FEET are in elimination zone (bottom-center of box)
                # This ensures staff in front of counter aren't excluded
                if in_elimination_zone(cx, py2):
                    continue
                
                w = px2 - px1
                h = py2 - py1
                
                is_v8 = tid in v8_assigned_tids
                
                # Check table proximity
                is_at_table = False
                for tb in table_boxes:
                    if person_near_table(person_box, tb):
                        is_at_table = True
                        break
                
                trackers[tid].update(cx, cy, w, h, is_v8, is_at_table, fw, fh)
                
                # Ghost filter disabled - was removing seated clients
                # t = trackers[tid]
                # if t.total_frames >= 60 and t.get_movement_speed() < 0.3:
                #     continue
                
                label, combined, visual, table, behavior = trackers[tid].get_label()
                debug = trackers[tid].get_debug_info()
                speed = trackers[tid].get_movement_speed()
                
                # Find which table this person is near
                near_table_idx = None
                if is_at_table:
                    for ti, tb in enumerate(table_boxes):
                        if person_near_table(person_box, tb):
                            near_table_idx = ti
                            break
                
                if label == "staff":
                    frame_staff += 1
                    current_staff_boxes[tid] = [px1, py1, px2, py2]
                    cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
                    last_positions[tid] = (cx, cy)

                    
                    # --- Staff identification (vote + lock) ---
                    staff_display_name = None
                    now = time.time()
                    
                    # Consume any ready face results (non-blocking)
                    while not face_result_queue.empty():
                        try:
                            r_tid, r_name, r_conf = face_result_queue.get_nowait()
                            if r_name and r_tid not in id_locked:
                                id_votes[r_tid].append(("FACE", r_name, r_conf))
                            elif r_tid in id_locked and r_tid in id_to_name:
                                if r_name and id_to_name[r_tid][0] != r_name:
                                    # ArcFace dit un NOM DIFFÉRENT du nom verrouillé
                                    id_reverify[r_tid].append(False)
                                    locked_display = NAME_MAP.get(id_to_name[r_tid][0], id_to_name[r_tid][0])
                                    face_display   = NAME_MAP.get(r_name, r_name)
                                    print(f"  ⚠️  Reverify FAIL #{r_tid}: locké={locked_display}, face={face_display}")
                                elif r_name:
                                    id_reverify[r_tid].append(True)
                            # --- Utiliser id_reverify pour invalider un mauvais lock ---
                            recent = id_reverify.get(r_tid, [])
                            if len(recent) >= 3:
                                fails = sum(1 for x in recent[-3:] if x is False)
                                if fails >= 2:  # 2 désaccords sur les 3 dernières verifs
                                    unlocked_name = NAME_MAP.get(id_to_name.get(r_tid, ('?',))[0], id_to_name.get(r_tid, ('?',))[0])
                                    print(f"  🔓 Unlock #{r_tid} ({unlocked_name}): ArcFace contredit 2/3 fois")
                                    id_locked.discard(r_tid)
                                    if r_tid in id_to_name: del id_to_name[r_tid]
                                    if r_tid in id_votes:   id_votes[r_tid] = []
                                    id_reverify[r_tid] = []
                        except:
                            pass
                    
                    # Submit ArcFace recognition request (async, non-blocking)
                    # Send FACE CROP only (top FACE_CROP_TOP_PCT of bbox) — not full body
                    if arcface_model and tid not in id_locked and tid not in face_pending:
                        if now - id_face_last_check.get(tid, 0) > FACE_CHECK_INTERVAL:
                            fh_crop = max(1, int((py2 - py1) * FACE_CROP_TOP_PCT))
                            face_crop = frame[py1:py1 + fh_crop, px1:px2]
                            if face_crop.size > 0 and face_crop.shape[0] > 20 and face_crop.shape[1] > 20:
                                try:
                                    face_request_queue.put_nowait((tid, face_crop.copy()))
                                    face_pending.add(tid)
                                    id_face_last_check[tid] = now
                                except:
                                    pass  # Queue full, skip
                    
                    # Body classifier (sync, every BODY_CHECK_INTERVAL)
                    if staff_classifier and tid not in id_locked and (now - id_body_last_check.get(tid, 0) > BODY_CHECK_INTERVAL):
                        body_crop = frame[py1:py2, px1:px2]
                        if body_crop.size > 0 and body_crop.shape[0] > 30 and body_crop.shape[1] > 30:
                            taken_names = {id_to_name[t][0] for t in id_locked if t in id_to_name and t != tid}
                            fname, fsim = identify_by_body(body_crop)
                            id_body_last_check[tid] = now
                            if fname and fname in taken_names:
                                if isinstance(staff_classifier, YOLO):
                                    try:
                                        results = staff_classifier(body_crop, verbose=False)[0]
                                        probs_list = results.probs.data.tolist()
                                        ranked = sorted(enumerate(probs_list), key=lambda x: -x[1])
                                        for idx2, conf2 in ranked:
                                            name_candidate = staff_class_names[idx2]
                                            if name_candidate not in taken_names and conf2 >= 0.40:
                                                fname, fsim = name_candidate, conf2
                                                break
                                        else:
                                            fname = None
                                    except:
                                        fname = None
                                else:
                                    fname = None
                            if fname:
                                id_votes[tid].append(("BODY", fname, fsim))
                    
                    # Process votes (face votes count double)
                    from collections import Counter
                    all_votes = id_votes.get(tid, [])
                    if all_votes and tid not in id_locked:
                        taken_names = {id_to_name[t][0] for t in id_locked if t in id_to_name and t != tid}
                        # Weight: face=2, body=1
                        weighted = {}
                        last_conf = {}
                        for src, vname, vconf in all_votes:
                            if vname not in taken_names:
                                w = 2 if src == "FACE" else 1
                                weighted[vname] = weighted.get(vname, 0) + w
                                last_conf[vname] = vconf
                        if weighted:
                            top_name = max(weighted, key=weighted.get)
                            face_votes_for_top = sum(1 for s, n, c in all_votes if s == "FACE" and n == top_name)
                            body_votes_for_top = sum(1 for s, n, c in all_votes if s == "BODY" and n == top_name)
                            arcface_has_responded = any(s == "FACE" for s, n, c in all_votes)
                            
                            # --- RÈGLE DE LOCK (caméras surveillance basse qualité) ---
                            # Cas 1 : ArcFace a confirmé → 2 votes FACE suffisent
                            # Cas 2 : ArcFace n'a PAS encore répondu (visage non visible)
                            #         → body seul peut locker MAIS seulement avec 6 votes
                            #           (3 vote × 2 checks/intervalle = ~9 secondes de cohérence)
                            # Cas 3 : ArcFace a répondu MAIS pour un autre nom
                            #         → body ne peut PAS locker (conflit → attendre)
                            should_lock = False
                            if face_votes_for_top >= 2:
                                # ArcFace confirme → lock immédiat
                                should_lock = True
                                print(f"  ✅ Lock #{tid} → {NAME_MAP.get(top_name, top_name)} (ArcFace: {face_votes_for_top} votes, sim={last_conf.get(top_name, 0):.2f})")
                            elif body_votes_for_top >= 6 and not arcface_has_responded:
                                # Body seul ET ArcFace n'a jamais répondu (visage jamais visible)
                                # → on accepte après beaucoup de votes body cohérents
                                should_lock = True
                                print(f"  ⚠️  Lock #{tid} → {NAME_MAP.get(top_name, top_name)} (Body seulement: {body_votes_for_top} votes — aucune réponse ArcFace)")
                            elif body_votes_for_top >= 3 and arcface_has_responded:
                                # Body + ArcFace a répondu (peut-être un autre nom) → ne pas locker
                                # ArcFace a la priorité — attendre qu'ArcFace confirme
                                pass
                            
                            if should_lock:
                                id_to_name[tid] = (top_name, last_conf.get(top_name, 0.5))
                                id_locked.add(tid)
                    
                    # Display: only show name if LOCKED
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                    if tid in id_locked and tid in id_to_name:
                        name_info = id_to_name[tid]
                        # Traduire x/y/z → vrai nom
                        display_name = NAME_MAP.get(name_info[0], name_info[0])
                        # Show source: face (F) or body (B)
                        src = "F" if name_info[1] >= FACE_SIM_THRESHOLD else "B"
                        cv2.putText(frame, "{} {:.0f}% [{}]".format(display_name.upper(), combined*100, src),
                                   (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        vote_count = len(id_votes.get(tid, []))
                        cv2.putText(frame, "STAFF #{} {:.0f}% ({}/5)".format(tid, combined*100, vote_count),
                                   (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Debug scores
                    dbg = f"V:{visual:.0%} T:{table:.0%} B:{behavior:.0%}"
                    cv2.putText(frame, dbg, (px1, py2+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
                elif label == "client":
                    frame_client += 1
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                    cv2.putText(frame, "CLIENT #{}".format(tid),
                               (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (128, 128, 128), 1)
        
        # ══════════════════════════════════════════════════════════════════
        # CROSSING-AWARE TRACK RECOVERY
        # ══════════════════════════════════════════════════════════════════
        current_tids = set(current_staff_boxes.keys())

        # ── Step A: Detect crossing events ────────────────────────────────
        # If two staff are within CROSS_DIST px of each other → flag both as "crossing"
        CROSS_DIST = 80  # pixels — adjust for your camera resolution
        staff_positions = {}  # tid → (cx, cy)
        for tid_c in current_tids:
            if tid_c in current_staff_boxes:
                b = current_staff_boxes[tid_c]
                staff_positions[tid_c] = ((b[0]+b[2])//2, (b[1]+b[3])//2)

        for ta, (ax, ay) in staff_positions.items():
            for tb, (bx, by) in staff_positions.items():
                if ta >= tb:
                    continue
                dist = math.sqrt((ax-bx)**2 + (ay-by)**2)
                if dist < CROSS_DIST:
                    # Mark both as recently crossing — block proximity recovery
                    crossing_flag[ta] = time.time()
                    crossing_flag[tb] = time.time()

        # ── Step B: Release lost locked tracks ────────────────────────────
        RECOVERY_COOLDOWN = 5.0  # seconds — a recovered track can't be re-released this fast
        for locked_tid in list(id_locked):
            if locked_tid not in current_tids and locked_tid in id_to_name:
                # COOLDOWN CHECK: if recently recovered, don't release yet
                # This prevents the recover → release → recover loop
                if time.time() - recovery_cooldown.get(locked_tid, 0) < RECOVERY_COOLDOWN:
                    continue  # Still in cooldown, keep the lock
                name = id_to_name[locked_tid][0]
                conf = id_to_name[locked_tid][1]
                if locked_tid in last_positions and name not in lost_tracks:
                    pos = last_positions[locked_tid]
                    was_crossing = (time.time() - crossing_flag.get(locked_tid, 0)) < 3.0
                    lost_tracks[name] = (pos[0], pos[1], time.time(), conf, was_crossing)
                id_locked.discard(locked_tid)
                if locked_tid in id_to_name:
                    del id_to_name[locked_tid]

        # ── Step C: Auto-recover with crossing awareness ───────────────────
        # Track names assigned in THIS frame to prevent double-assignment
        assigned_this_frame = set()

        for new_tid in sorted(current_tids):  # sorted for determinism
            if new_tid in id_locked or new_tid in id_to_name:
                continue
            if new_tid not in current_staff_boxes:
                continue

            # Skip if this track itself is currently crossing
            if (time.time() - crossing_flag.get(new_tid, 0)) < 2.0:
                continue  # Wait for ArcFace to identify it properly

            nb = current_staff_boxes[new_tid]
            ncx = (nb[0] + nb[2]) // 2
            ncy = (nb[1] + nb[3]) // 2

            best_name_recover = None
            best_dist_recover = RECOVER_DIST_PX

            for lname, ldata in lost_tracks.items():
                if lname in assigned_this_frame:
                    continue  # Already claimed this frame
                lx, ly, lt, lconf, was_cross = ldata[0], ldata[1], ldata[2], ldata[3], ldata[4]
                if time.time() - lt > NAME_MEMORY_SEC:
                    continue
                # If the lost track was crossing → require MUCH shorter distance
                # to avoid assigning the wrong name after position swap
                dist_limit = 40 if was_cross else RECOVER_DIST_PX
                dist = math.sqrt((ncx - lx)**2 + (ncy - ly)**2)
                if dist < min(best_dist_recover, dist_limit):
                    best_dist_recover = dist
                    best_name_recover = (lname, lconf)

            if best_name_recover:
                rname, rconf = best_name_recover
                # Final safety: not already taken by a VISIBLE locked track
                already_taken = any(
                    id_to_name.get(t, (None,))[0] == rname
                    for t in current_tids if t != new_tid
                )
                if not already_taken and rname not in assigned_this_frame:
                    id_to_name[new_tid] = (rname, rconf)
                    id_locked.add(new_tid)
                    recovery_cooldown[new_tid] = time.time()  # start cooldown
                    del lost_tracks[rname]
                    assigned_this_frame.add(rname)
                    # ← NOUVEAU : forcer une re-vérification ArcFace immédiate
                    # (reset du timer pour que le prochain frame envoie un crop)
                    id_face_last_check[new_tid] = 0
                    id_reverify[new_tid] = []  # reset l'historique de vérif
                    print(f"  🔄 Track #{new_tid} recovered → {NAME_MAP.get(rname, rname)} (dist={best_dist_recover:.0f}px) — ArcFace re-check forcé")

        # ── Step D: Enforce uniqueness ─────────────────────────────────────
        # If somehow 2 visible tracks have same name, keep highest confidence
        visible_names = {}
        for tid_v in current_tids:
            if tid_v in id_locked and tid_v in id_to_name:
                name_v = id_to_name[tid_v][0]
                conf_v = id_to_name[tid_v][1]
                if name_v in visible_names:
                    prev_tid, prev_conf = visible_names[name_v]
                    if conf_v > prev_conf:
                        id_locked.discard(prev_tid)
                        if prev_tid in id_to_name: del id_to_name[prev_tid]
                        if prev_tid in id_votes:   id_votes[prev_tid] = []
                        visible_names[name_v] = (tid_v, conf_v)
                    else:
                        id_locked.discard(tid_v)
                        if tid_v in id_to_name: del id_to_name[tid_v]
                        if tid_v in id_votes:   id_votes[tid_v] = []
                else:
                    visible_names[name_v] = (tid_v, conf_v)

        # ── Step E: Clean expired entries ──────────────────────────────────
        now_clean = time.time()
        lost_tracks  = {n: v for n, v in lost_tracks.items()  if now_clean - v[2] < NAME_MEMORY_SEC}
        crossing_flag = {t: ts for t, ts in crossing_flag.items() if now_clean - ts < 5.0}
        
        # HUD - top left
        cv2.putText(frame, "STAFF: {}".format(frame_staff), (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "CLIENT: {}".format(frame_client), (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        

        

        
        # Draw elimination zones (semi-transparent)
        for z in elimination_zones:
            overlay = frame.copy()
            cv2.rectangle(overlay, (z[0], z[1]), (z[2], z[3]), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), (0, 0, 200), 1)
        
        cv2.putText(frame, "R=zones | F=+5s | B=-5s | Q=quitter",
                   (10, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # video_writer.write(frame)  # recording disabled
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f') or key == 83:  # 'f' or right arrow
            # Skip forward 5 seconds
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current + fps * 5)
            print(f"⏩ +5s (frame {int(current)} → {int(current + fps*5)})")
        elif key == ord('b') or key == 81:  # 'b' or left arrow
            # Skip backward 5 seconds
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - fps * 5))
            print(f"⏪ -5s (frame {int(current)} → {int(max(0, current - fps*5))})")
        elif key == ord('r'):
            # Redraw zones
            ret2, fresh = cap.read()
            if ret2:
                source_name = Path(str(video_path)).stem if not str(video_path).startswith("rtsp") else str(video_path).split("/")[-1]
                # Delete saved zones
                zones_file = Path("elimination_zones.json")
                if zones_file.exists():
                    data = json.loads(zones_file.read_text())
                    data.pop(source_name, None)
                    zones_file.write_text(json.dumps(data, indent=2))
                elimination_zones.clear()
                select_elimination_zones(fresh, source_name)
    
    cap.release()
    # video_writer.release()  # recording disabled
    cv2.destroyAllWindows()
    
    print("Session terminée.")
# Live camera streams via go2rtc
CAMERAS = {
    "1": ("imou_cam1", "rtsp://100.96.105.67:8554/imou_cam1"),
    "2": ("imou_cam2", "rtsp://100.96.105.67:8554/imou_cam2"),
    "3": ("imou_cam3", "rtsp://100.96.105.67:8554/imou_cam3"),
    "4": ("imou_cam4", "rtsp://100.96.105.67:8554/imou_cam4"),
    "5": ("imou_cam5", "rtsp://100.96.105.67:8554/imou_cam5"),
    "6": ("imou_cam6", "rtsp://100.96.105.67:8554/imou_cam6"),
}

def main():
    print("=" * 50)
    print("SERVER EFFICIENCY TRACKER")
    print("  3 Models: Person + Staff(v8) + Tables")
    print("  📊 Staff/Client tracking en temps réel")
    print("=" * 50)
    
    if not STAFF_MODEL_V8.exists():
        print(f"Staff model not found: {STAFF_MODEL_V8}")
        return
    
    print("\n📹 SOURCES DISPONIBLES:")
    for k, (name, _) in CAMERAS.items():
        print(f"  {k} = {name} (LIVE)")
    print("  f = Ouvrir un fichier vidéo")
    
    choice = input("\nChoisir (1-6 ou f): ").strip()
    
    if choice == "f":
        video = select_video()
        if not video:
            print("Aucun fichier sélectionné")
            return
    elif choice in CAMERAS:
        _, video = CAMERAS[choice]
    else:
        print("Choix invalide")
        return
    
    test_hybrid(video)

if __name__ == "__main__":
    main()
