"""
ws_dashboard_server.py
======================
Serveur WebSocket temps réel pour Kitchen Sparkle Desk Dashboard.

Lance ce serveur EN PARALLÈLE de test_staff_client.py :
    python ws_dashboard_server.py

Le dashboard React se connecte à ws://localhost:8765 et reçoit
les métriques dès qu'elles changent (max 1 push/seconde).

Données exposées :
  - Serveurs identifiés (nom ArcFace/body, caméra, score, vitesse, zone, etc.)
  - Tables (statut : libre / occupée / en attente / visitée)
  - Statistiques globales (clients, alertes, attente moyenne)
  - Historique des scores (dernier 8 points toutes les 15 sec)

Usage autonome (sans detect. active) :
  Le serveur simule des données réalistes si aucun tracker n'est connecté,
  pour permettre de tester le dashboard seul.
"""

import asyncio
import json
import math
import random
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import websockets

# ─── Video analysis state ────────────────────────────────────────────────────
_analysis: dict = {
    "running": False,
    "progress": 0.0,   # 0.0 → 1.0
    "mode": "idle",    # idle | loading | running | done | error
    "error": "",
    "video_name": "",
}
_analysis_stop = threading.Event()
_analysis_thread: threading.Thread | None = None

# ─── Shared state (written by detector, read by WS server) ──────────────────
# These dicts are populated by the detection pipeline when run in the same
# process, OR by the standalone simulation below.
_shared: dict = {
    "servers": {},       # {name: ServerData dict}
    "tables": {},        # {table_id: TableData dict}
    "global": {},        # global stats dict
    "last_update": 0.0,
}

COLORS = {
    0: "#F59E0B",   # amber
    1: "#60A5FA",   # blue
    2: "#F87171",   # red
    3: "#A78BFA",   # purple
    4: "#34D399",   # green
    5: "#FB923C",   # orange
}

ZONES = [
    "Zone A - Entrée",
    "Zone B - Centre",
    "Zone C - Terrasse",
    "Zone D - Bar",
    "Zone E - Cuisine",
]


# ─── Helper: compute score ────────────────────────────────────────────────────
def compute_score(speed_score, reactivity_score, coverage_score, standing_score):
    return round(
        speed_score * 0.30
        + reactivity_score * 0.30
        + coverage_score * 0.25
        + standing_score * 0.15
    )


# ─── Simulation engine (used when no detector is connected) ──────────────────
class ServerSim:
    """Simulates a single named server with realistic metric drift."""

    def __init__(self, idx: int, name: str):
        self.idx = idx
        self.name = name
        self.color = COLORS[idx % len(COLORS)]
        self.camera = f"CAM-{idx + 1:02d}"
        self.arrival = "18:30"
        self.base_speed = random.uniform(55, 92)
        self.base_react = random.uniform(55, 92)
        self.base_cov   = random.uniform(50, 90)
        self.base_stand = random.uniform(65, 95)
        self.tables_total = 15
        self.tables_visited = random.randint(4, 14)
        self.recognition_score = random.randint(88, 99)
        self.zone_idx = idx % len(ZONES)
        self.history: deque = deque(maxlen=8)  # (timestamp_str, score)
        self._t = 0

    def tick(self) -> dict:
        self._t += 1
        jitter = lambda v: max(0, min(100, v + random.gauss(0, 3)))

        speed_s  = jitter(self.base_speed  + math.sin(self._t * 0.05) * 6)
        react_s  = jitter(self.base_react  + math.sin(self._t * 0.07) * 5)
        cov_s    = jitter(self.base_cov    + math.sin(self._t * 0.03) * 4)
        stand_s  = jitter(self.base_stand  + math.sin(self._t * 0.04) * 3)

        score = compute_score(speed_s, react_s, cov_s, stand_s)
        speed_px = round(max(0.3, (speed_s / 100) * 5.0), 1)

        avg_response = round(max(0.5, (100 - react_s) / 12), 1)
        standing_pct = int(stand_s)

        # Drift tables visited slowly
        if random.random() < 0.08 and self.tables_visited < self.tables_total:
            self.tables_visited += 1
        if random.random() < 0.01 and self.tables_visited > 0:
            self.tables_visited -= 1

        # Zone drift
        if random.random() < 0.04:
            self.zone_idx = (self.zone_idx + random.choice([-1, 1])) % len(ZONES)

        now_str = datetime.now().strftime("%H:%M")
        self.history.append({"time": now_str, "value": score})

        # Alerts
        alerts = []
        if avg_response > 6:
            alerts.append({
                "id": f"a-{self.name}-react",
                "type": "critical",
                "message": f"Réactivité critique : {avg_response} min en moyenne",
                "time": now_str,
            })
        elif avg_response > 4:
            alerts.append({
                "id": f"a-{self.name}-react-w",
                "type": "warning",
                "message": f"Réactivité faible : {avg_response} min en moyenne",
                "time": now_str,
            })
        cov_pct = round(self.tables_visited / self.tables_total * 100)
        if cov_pct < 35:
            alerts.append({
                "id": f"a-{self.name}-cov",
                "type": "critical",
                "message": f"Couverture très faible : {cov_pct}% des tables",
                "time": now_str,
            })
        elif cov_pct < 50:
            alerts.append({
                "id": f"a-{self.name}-cov-w",
                "type": "warning",
                "message": f"Couverture faible : {cov_pct}% des tables",
                "time": now_str,
            })
        if score < 35:
            alerts.append({
                "id": f"a-{self.name}-score",
                "type": "critical",
                "message": f"Score d'efficacité critique : {score}/100",
                "time": now_str,
            })

        speed_label = "Rapide" if speed_s >= 70 else ("Normal" if speed_s >= 45 else "Lent")

        service_start_min = 18 * 60 + 30
        now_min = datetime.now().hour * 60 + datetime.now().minute
        dur_min = max(0, now_min - service_start_min)
        dur_str = f"{dur_min // 60}h {dur_min % 60:02d}m"

        return {
            "id": f"srv-{self.idx + 1}",
            "name": self.name,
            "avatar": "".join([p[0].upper() for p in self.name.split()[:2]]),
            "camera": self.camera,
            "arrivalTime": self.arrival,
            "serviceDuration": dur_str,
            "score": score,
            "speedScore": round(speed_s),
            "reactivityScore": round(react_s),
            "coverageScore": round(cov_s),
            "standingScore": round(stand_s),
            "speed": speed_px,
            "speedLabel": speed_label,
            "tablesVisited": self.tables_visited,
            "totalTables": self.tables_total,
            "avgResponseTime": avg_response,
            "standingPercent": standing_pct,
            "recognitionScore": self.recognition_score,
            "lastZone": ZONES[self.zone_idx],
            "color": self.color,
            "alerts": alerts,
            "speedHistory": list(self.history),
            "activityBySlot": list(self.history),
            "position": {
                "x": 15 + (self.idx % 3) * 30 + random.uniform(-3, 3),
                "y": 25 + (self.idx // 3) * 35 + random.uniform(-3, 3),
            },
            "source": "simulation",
        }


class TableSim:
    STATUSES = ["free", "occupied", "waiting", "visited"]

    def __init__(self, table_id: int, x: int, y: int):
        self.table_id = table_id
        self.x = x
        self.y = y
        self.status = random.choice(self.STATUSES)
        self.wait = random.randint(0, 14) if self.status in ("occupied", "waiting") else 0
        self._timer = 0

    def tick(self) -> dict:
        self._timer += 1
        # Slow random state changes
        if random.random() < 0.015:
            self.status = random.choice(self.STATUSES)
            self.wait = random.randint(0, 14) if self.status in ("occupied", "waiting") else 0
        elif self.status == "waiting":
            self.wait = min(self.wait + 1, 30)
        elif self.status == "occupied" and self._timer % 20 == 0:
            self.wait = max(0, self.wait - 1)
        return {
            "id": self.table_id,
            "x": self.x,
            "y": self.y,
            "status": self.status,
            "waitMinutes": self.wait,
        }


# ─── Build simulation data ────────────────────────────────────────────────────

# Real name mapping: face_db keys → display names
NAME_MAP: dict[str, str] = {
    "x": "Mohamed Ali",
    "y": "Rami",
    "z": "Sadio",
    # Add more if needed: "a": "Khaled"
}

# Camera streams available on Go2RTC server
CAMERAS = [
    {"id": "imou_cam1", "label": "CAM-01", "stream_url": "http://100.96.105.67:1984/stream.html?src=imou_cam1", "api": "http://100.96.105.67:1984"},
    {"id": "imou_cam2", "label": "CAM-02", "stream_url": "http://100.96.105.67:1984/stream.html?src=imou_cam2", "api": "http://100.96.105.67:1984"},
    {"id": "imou_cam3", "label": "CAM-03", "stream_url": "http://100.96.105.67:1984/stream.html?src=imou_cam3", "api": "http://100.96.105.67:1984"},
    {"id": "imou_cam4", "label": "CAM-04", "stream_url": "http://100.96.105.67:1984/stream.html?src=imou_cam4", "api": "http://100.96.105.67:1984"},
    {"id": "imou_cam5", "label": "CAM-05", "stream_url": "http://100.96.105.67:1984/stream.html?src=imou_cam5", "api": "http://100.96.105.67:1984"},
    {"id": "imou_cam6", "label": "CAM-06", "stream_url": "http://100.96.105.67:1984/stream.html?src=imou_cam6", "api": "http://100.96.105.67:1984"},
]

# ─── Data directory (models, face_db, videos) ────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_DATA_DIR = _SCRIPT_DIR.parent / "staff detection images"
if not _DATA_DIR.exists():
    _DATA_DIR = _SCRIPT_DIR  # Fallback si lancé depuis le dossier original

# Try to read real server names from face_db.pkl
def load_server_names() -> list[str]:
    face_db_path = _DATA_DIR / "face_db.pkl"
    if face_db_path.exists():
        try:
            import pickle
            with open(face_db_path, "rb") as f:
                db = pickle.load(f)
            # Map internal keys → display names
            raw_keys = list(db.keys())
            names = [NAME_MAP.get(k, k.capitalize()) for k in raw_keys]
            if names:
                print(f"✅ Vrais noms chargés depuis face_db: {raw_keys} → {names}")
                return names
        except Exception as e:
            print(f"⚠️  face_db.pkl illisible: {e}")
    # Fallback
    fallback = ["Mohamed Ali", "Rami", "Sadio"]
    print(f"ℹ️  Noms par défaut: {fallback}")
    return fallback


SERVER_NAMES = load_server_names()

_sims: list[ServerSim] = [ServerSim(i, n) for i, n in enumerate(SERVER_NAMES)]

TABLE_POSITIONS = [
    (1, 12, 15), (2, 12, 38), (3, 12, 60), (4, 12, 80),
    (5, 38, 15), (6, 38, 38), (7, 38, 60), (8, 38, 80),
    (9, 63, 15), (10, 63, 38), (11, 63, 60), (12, 63, 80),
    (13, 85, 25), (14, 85, 55), (15, 85, 78),
]
_table_sims: list[TableSim] = [TableSim(tid, x, y) for tid, x, y in TABLE_POSITIONS]


# ─── Build payload ────────────────────────────────────────────────────────────
def build_payload() -> dict:
    """
    If _shared["servers"] is populated by the live detector, use that.
    Otherwise fall back to simulation.
    """
    now = datetime.now()
    cameras_live = _shared.get("cameras_live", False)

    # --- SERVERS ---
    if _shared["servers"]:
        servers_list = list(_shared["servers"].values())
    elif cameras_live:
        servers_list = []          # cameras live, no staff yet → empty (real)
    else:
        servers_list = [s.tick() for s in _sims]   # no cameras → simulation

    # --- CAMERAS --- always included
    cameras_list = CAMERAS

    # --- TABLES ---
    if _shared["tables"]:
        tables_list = list(_shared["tables"].values())
    elif cameras_live:
        tables_list = []           # cameras live, no tables detected → empty
    else:
        tables_list = [t.tick() for t in _table_sims]

    # --- GLOBAL ---
    active      = len(servers_list)
    g           = _shared["global"]
    # cameras_live=True as soon as video_to_dashboard.py publisher starts
    cameras_live = _shared.get("cameras_live", False)
    is_live      = cameras_live   # live = cameras running, regardless of staff count
    alert_count = sum(len(s.get("alerts", [])) for s in servers_list)
    avg_score   = round(sum(s["score"] for s in servers_list) / max(len(servers_list), 1))

    if cameras_live:
        # ── Real camera values (staff may be 0 for now) ───────────────
        total_clients   = int(g.get("totalClients", 0))
        tables_occupied = int(g.get("tablesOccupied",
                            sum(1 for t in tables_list if t["status"] in ("occupied","waiting"))))
        total_tables    = int(g.get("totalTables", max(len(tables_list), 15)))
        avg_wait        = float(g.get("avgWaitTime", 0.0))
    else:
        # ── Simulation estimates (no cameras at all) ──────────────────
        tables_occupied = sum(1 for t in tables_list if t["status"] in ("occupied", "waiting"))
        total_tables    = max(len(tables_list), 15)
        total_clients   = tables_occupied * random.randint(2, 4)
        waiting_tables  = [t for t in tables_list if t["status"] == "waiting" and t.get("waitMinutes", 0) > 0]
        avg_wait = (
            round(sum(t["waitMinutes"] for t in waiting_tables) / len(waiting_tables), 1)
            if waiting_tables else round(random.uniform(1.5, 4.5), 1)
        )

    global_stats = {
        "activeServers":  active,
        "totalClients":   total_clients,
        "tablesOccupied": tables_occupied,
        "totalTables":    total_tables,
        "avgWaitTime":    avg_wait,
        "serviceStart":   g.get("serviceStart", "18:30"),
        "currentTime":    now.strftime("%H:%M:%S"),
        "alertCount":     alert_count,
        "avgScore":       avg_score,
        "timestamp":      now.isoformat(),
        "mode":           "live" if is_live else "simulation",
        "camerasLive":    cameras_live,
        "staffDetected":  active > 0,
        "videoPath":      _shared.get("video_path", ""),
    }


    return {
        "type": "dashboard_update",
        "servers": servers_list,
        "tables": tables_list,
        "global": global_stats,
        "cameras": cameras_list,
    }


# ─── WebSocket handler ────────────────────────────────────────────────────────
CLIENTS: set = set()


async def handler(websocket):
    CLIENTS.add(websocket)
    client_ip = websocket.remote_address[0] if websocket.remote_address else "?"
    print(f"🔌 Client connecté: {client_ip}  (total: {len(CLIENTS)})")
    try:
        # Send first payload immediately on connect
        payload = build_payload()
        await websocket.send(json.dumps(payload, ensure_ascii=False))

        async for message in websocket:
            # Accept control messages from dashboard
            try:
                msg = json.loads(message)
                if msg.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                elif msg.get("type") == "get_snapshot":
                    await websocket.send(json.dumps(build_payload(), ensure_ascii=False))
            except Exception:
                pass

    except (websockets.exceptions.ConnectionClosed, Exception) as e:
        if not isinstance(e, websockets.exceptions.ConnectionClosed):
            print(f"⚠️  Erreur client: {e}")
    finally:
        CLIENTS.discard(websocket)
        print(f"❌ Client déconnecté. (restants: {len(CLIENTS)})")


async def broadcaster():
    """Push updates to all connected clients every second."""
    global CLIENTS
    print("📡 Broadcaster démarré (1 push/sec)")
    while True:
        await asyncio.sleep(1.0)
        if not CLIENTS:
            continue
        try:
            payload = json.dumps(build_payload(), ensure_ascii=False)
        except Exception as exc:
            print(f"⚠️  Broadcaster build_payload error: {exc}")
            continue   # skip this tick, don't disconnect clients
        dead: set = set()
        for ws in list(CLIENTS):
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        CLIENTS -= dead


# ─── Control API (HTTP port 8766) ─────────────────────────────────────────────
_pending_video: dict = {"path": None}
_BASE_DIR = _DATA_DIR


class _ControlHandler(BaseHTTPRequestHandler):
    def _cors(self, status: int = 200):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_OPTIONS(self):
        self._cors()

    def do_GET(self):
        # ── Video streaming with HTTP Range support ──────────────────────────
        if self.path.startswith("/api/serve-video"):
            from urllib.parse import urlparse, parse_qs, unquote
            qs = parse_qs(urlparse(self.path).query)
            raw = (qs.get("path", [""])[0]).strip()
            if not raw:
                self._cors(400)
                self.wfile.write(b'{"error":"missing path"}')
                return
            fpath = Path(unquote(raw))
            if not fpath.exists():
                self._cors(404)
                self.wfile.write(b'{"error":"file not found"}')
                return
            size = fpath.stat().st_size
            range_header = self.headers.get("Range", "")
            ext = fpath.suffix.lower()
            mime = {"mp4":"video/mp4","avi":"video/x-msvideo","mkv":"video/webm",
                    "mov":"video/quicktime","dav":"video/mp4"}.get(ext.lstrip("."), "video/mp4")
            if range_header:
                # e.g.  bytes=0-1023
                byte_range = range_header.replace("bytes=", "").split("-")
                start = int(byte_range[0]) if byte_range[0] else 0
                end   = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else size - 1
                end   = min(end, size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", mime)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fpath, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            return
        elif self.path.startswith("/api/alert-frame"):
            # Serve a JPEG screenshot captured at alert time
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            fid = (qs.get("id", [""])[0]).strip()
            # Import the global _alert_frames from video_to_dashboard (if embedded)
            try:
                from video_to_dashboard import _alert_frames
            except ImportError:
                _alert_frames = {}
            jpeg_bytes = _alert_frames.get(fid)
            if jpeg_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(jpeg_bytes)
            else:
                self._cors(404)
                self.wfile.write(b'{"error":"frame not found"}')
            return
        elif self.path == "/api/current-video":
            self._cors()
            self.wfile.write(json.dumps({
                "path": _pending_video.get("_running", ""),
            }).encode())
            return
        elif self.path == "/api/videos":
            import glob
            patterns = ["*.mp4", "*.avi", "*.mkv", "*.dav",
                        "azzabivids/*.mp4", "azzabivids/*.avi"]
            found = []
            for p in patterns:
                found.extend(glob.glob(str(_BASE_DIR / p)))
            videos = sorted(
                [str(Path(v).relative_to(_BASE_DIR)).replace("\\", "/") for v in found],
                key=lambda x: x.lower()
            )
            self._cors()
            self.wfile.write(json.dumps({"videos": videos}).encode())

        elif self.path == "/api/browse":
            # Open native Windows file picker
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                chosen = filedialog.askopenfilename(
                    title="Choisir une vidéo",
                    filetypes=[
                        ("Vidéos", "*.mp4 *.avi *.mkv *.mov *.dav"),
                        ("Tous les fichiers", "*.*"),
                    ],
                )
                root.destroy()
                self._cors()
                self.wfile.write(json.dumps({"path": chosen or ""}).encode())
            except Exception as e:
                self._cors(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/api/status":
            self._cors()
            self.wfile.write(json.dumps({
                "cameras_live": _shared.get("cameras_live", False),
                "pending": _pending_video.get("path"),
            }).encode())
        else:
            self._cors(404)
            self.wfile.write(b'{"error":"not found"}')

    def do_POST(self):
        if self.path == "/api/run-video":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            abs_path = body.get("absolute_path", "")
            rel_path = body.get("path", "")
            resolved = abs_path if abs_path else (str(_BASE_DIR / rel_path) if rel_path else "")
            if resolved:
                _pending_video["path"] = resolved
                # Build a file:// URL for the static file server (port 8767)
                try:
                    rel = str(Path(resolved).relative_to(_BASE_DIR)).replace("\\", "/")
                    file_url = f"http://localhost:8767/{rel}"
                except ValueError:
                    file_url = f"http://localhost:8767/{Path(resolved).name}"
                self._cors()
                self.wfile.write(json.dumps({
                    "ok": True,
                    "path": resolved,
                    "file_url": file_url,
                }).encode())
            else:
                self._cors(400)
                self.wfile.write(b'{"error":"missing path"}')
        elif self.path == "/api/stop":
            _pending_video["path"] = "__STOP__"
            self._cors()
            self.wfile.write(b'{"ok":true}')
        else:
            self._cors(404)
            self.wfile.write(b'{"error":"not found"}')

    def log_message(self, *args):
        pass


def start_control_api(port: int = 8766):
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("0.0.0.0", port), _ControlHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="control-api")
    t.start()
    print(f"🎮 API contrôle → http://localhost:{port}/api/videos")
    return srv


def start_file_server(port: int = 8767):
    """Dedicated static file server for video streaming — serves the detection folder."""
    import os
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class CORSFileHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()
        def log_message(self, *args): pass  # silence logs

    base = str(_BASE_DIR)

    class DirectoryHandler(CORSFileHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=base, **kwargs)

    srv = ThreadingHTTPServer(("0.0.0.0", port), DirectoryHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="file-server")
    t.start()
    print(f"📂 Serveur fichiers → http://localhost:{port}/  (base: {base})")
    return srv



async def main():
    host = "0.0.0.0"
    port = 8765
    print("=" * 60)
    print("🍴 Kitchen Sparkle Desk — Serveur WebSocket")
    print(f"   ws://{host}:{port}")
    print(f"   Serveurs simulés: {[s.name for s in _sims]}")
    print("=" * 60)
    start_control_api()
    start_file_server()

    async with websockets.serve(handler, host, port):
        await broadcaster()

# Alias for embedding in video_to_dashboard.py
main_server = main

if __name__ == "__main__":
    asyncio.run(main())
