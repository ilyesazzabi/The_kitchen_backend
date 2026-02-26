# 🍴 The Kitchen — Backend

Pipeline de détection IA pour restaurant : tracking staff, identification par visage (ArcFace), analyse d'efficacité en temps réel.

## 🚀 Installation

```bash
# 1. Cloner le repo (avec Git LFS pour les modèles)
git lfs install
git clone https://github.com/ilyesazzabi/The_kitchen_backend.git
cd The_kitchen_backend

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Démarrage

```bash
# Mode serveur (attend les commandes du dashboard)
python video_to_dashboard.py

# Analyser une vidéo
python video_to_dashboard.py --video chemin/video.mp4

# Caméras live (6 RTSP)
python video_to_dashboard.py --live

# Une seule caméra
python video_to_dashboard.py --cam imou_cam1
```

## 📡 Ports

| Port | Service |
|------|---------|
| 8765 | WebSocket (données temps réel) |
| 8766 | API HTTP (contrôle vidéo, parcourir fichiers) |
| 8767 | Serveur de fichiers (vidéos) |

## 🧠 Modèles inclus (Git LFS)

| Modèle | Taille | Usage |
|--------|--------|-------|
| `yolov8x.pt` | 130 MB | Détection personnes (précis) |
| `yolov8n.pt` | 6 MB | Détection personnes (rapide, mode live) |
| `staff_detector_v8.pt` | 50 MB | Détection visuelle du staff |
| `staff_classifier_yolo.pt` | 10 MB | Classification corps (identification) |
| `face_db.pkl` | ~1 MB | Base de données visages (ArcFace) |

## 🔗 Frontend

Le dashboard React se connecte à `ws://localhost:8765`.
→ [The-kitchen-Frontend](https://github.com/BramaSquare360/The-kitchen-Fronted)

## 📂 Fichiers

| Fichier | Rôle |
|---------|------|
| `video_to_dashboard.py` | Pipeline principal (vidéo/caméra → dashboard) |
| `ws_dashboard_server.py` | Serveur WebSocket + API HTTP |
| `pipeline_dashboard.py` | Pipeline unifié (alternative) |
| `staff_tracker_pro.py` | Tracking staff avec ByteTrack |
| `detect_staff.py` | Détection YOLO-World |
| `camera_config.json` | Configuration par caméra |
| `bytetrack_kitchen.yaml` | Config tracker |
| `requirements.txt` | Dépendances Python |
