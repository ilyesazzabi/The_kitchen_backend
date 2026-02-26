# The Kitchen — AI Restaurant Monitoring System

> Système de surveillance IA en temps réel pour restaurant, avec détection de personnel, tracking client, et dashboard de performance.

---

## 🏗️ Architecture

```
The_kitchen_camera_detection/
│
├── 🐍 backend/                 ← Ce dossier (Python)
│   ├── video_to_dashboard.py   ← Pipeline de détection principal
│   ├── ws_dashboard_server.py  ← Serveur WebSocket + API HTTP  
│   └── ...                     ← Scripts utilitaires
│
└── ⚛️  frontend/               ← kitchen-sparkle-desk/ (React)
    └── src/
        ├── components/dashboard/
        └── hooks/useWebSocketDashboard.ts
```

---

## 🚀 Démarrage rapide

### Prérequis
- Python 3.10+
- CUDA (recommandé pour la détection temps réel)
- Node.js 18+

### 1. Backend

```bash
# Créer l'environnement virtuel
python -m venv venv_new
venv_new\Scripts\activate   # Windows

# Installer les dépendances
pip install ultralytics deepface opencv-python websockets

# Lancer l'analyse vidéo
python video_to_dashboard.py --video chemin/vers/video.mp4

# Ou en mode live (caméras RTSP)
python video_to_dashboard.py --live
python video_to_dashboard.py --cam imou_cam1
```

### 2. Frontend

```bash
cd kitchen-sparkle-desk
npm install
npm run dev
# → http://localhost:5173
```

---

## 📡 Ports & APIs

| Port | Protocole | Usage |
|------|-----------|-------|
| **:8765** | WebSocket | Flux temps réel (métriques, alertes) |
| **:8766** | HTTP | API : `/api/alert-frame`, `/api/serve-video`, `/api/videos` |
| **:8767** | HTTP | Serveur de fichiers statiques |
| **:5173** | HTTP | Dashboard React (dev) |

---

## 🔍 Fonctionnalités

- **Détection personnes** : YOLOv8x (vidéo) / YOLOv8n (live)
- **Identification staff** : ArcFace (deepface) + classificateur YOLO custom
- **Tracking** : ByteTrack multi-caméra
- **Métriques temps réel** :
  - Score d'efficacité (vitesse × 30% + réactivité × 30% + couverture × 25% + temps debout × 15%)
  - Score journalier cumulé (persistant toute la journée)
  - Tables visitées, temps de réponse moyen
- **Alertes automatiques** :
  - Réactivité critique (> 6 min entre tables)
  - Couverture faible (< 30% tables)
  - Inactivité (> 3 min sans mouvement)
  - Score bas (< 35/100 pendant 5 min)
- **Preuve vidéo** : Screenshot au moment de l'alerte + clip vidéo seeké (±45s)

---

## 📷 Caméras supportées

Configuration RTSP dans `video_to_dashboard.py` :
```python
RTSP_STREAMS = {
    "imou_cam1": "rtsp://...",
    "imou_cam2": "rtsp://...",
    # ...
}
```

---

## ⚠️ Fichiers non inclus dans le dépôt

Ces fichiers sont trop lourds ou sensibles pour git :
- `*.pt` — Modèles YOLO entraînés (contacter pour accès)
- `face_db/`, `x/`, `y/`, `z/` — Bases de données de visages (privées)
- `venv_new/` — Environnement virtuel Python
- `*.mp4`, `*.dav` — Vidéos de caméra

---

## 🛠️ Scripts principaux

| Script | Usage |
|--------|-------|
| `video_to_dashboard.py` | Pipeline principal (détection + métriques) |
| `ws_dashboard_server.py` | Serveur WebSocket + HTTP |
| `add_more_staff.py` | Ajouter un nouveau serveur à la face_db |
| `auto_test_detection.py` | Tester la détection sur une vidéo |
| `train_staff_classifier.py` | Réentraîner le classificateur |
