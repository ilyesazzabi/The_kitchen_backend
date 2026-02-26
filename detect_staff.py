"""
THE KITCHEN - Staff Detector avec YOLO-World

Détecte le staff et les clients en utilisant YOLO-World (zero-shot detection)
pour identifier automatiquement les logos, uniformes, et comportements du staff.

Usage:
    python detect_staff.py                    # Détecter sur une image
    python detect_staff.py --video video.mp4  # Détecter sur une vidéo
    python detect_staff.py --camera 0         # Détecter en temps réel
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import argparse
import time


class StaffDetectorYOLOWorld:
    """
    Détecteur de staff utilisant YOLO-World pour la détection zero-shot.
    
    YOLO-World peut détecter des objets par description textuelle,
    permettant de chercher des logos, uniformes, tabliers, etc.
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence: float = 0.2,
        staff_threshold: float = 0.4  # Seuil plus bas car YOLO-World est fiable
    ):
        """
        Args:
            model_path: Chemin vers yolov8x-worldv2.pt (ou None pour télécharger)
            confidence: Seuil de confiance YOLO
            staff_threshold: Seuil pour classifier comme staff
        """
        from ultralytics import YOLO
        
        # Chercher le modèle YOLO-World
        if model_path is None:
            # Chercher dans les emplacements courants
            possible_paths = [
                Path(__file__).parent.parent / "vision-ia-restaurant" / "yolov8x-worldv2.pt",
                Path("yolov8x-worldv2.pt"),
                Path.home() / ".cache" / "yolov8x-worldv2.pt"
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = str(p)
                    break
            
            if model_path is None:
                print("📥 Téléchargement de YOLO-World...")
                model_path = "yolov8x-worldv2.pt"
        
        print(f"🧠 Chargement de YOLO-World: {model_path}")
        self.world_model = YOLO(model_path)
        
        # Configuration des classes à détecter pour le staff
        self.staff_classes = [
            "person wearing apron",
            "person wearing uniform",
            "waiter",
            "waitress", 
            "server",
            "restaurant staff",
            "chef",
            "bartender",
            "logo on shirt",
            "name badge",
            "person carrying tray",
            "person serving food"
        ]
        
        # Classes pour les clients
        self.customer_classes = [
            "person sitting at table",
            "customer",
            "diner",
            "person eating",
            "person drinking"
        ]
        
        # Configurer YOLO-World avec nos classes
        all_classes = self.staff_classes + self.customer_classes + ["person"]
        self.world_model.set_classes(all_classes)
        
        self.confidence = confidence
        self.staff_threshold = staff_threshold
        
        # Modèle standard pour tracking des personnes
        print("🧠 Chargement du modèle de tracking...")
        self.person_model = YOLO("yolo11n.pt")  # Léger pour le tracking
        
        # Tracking state
        self.person_scores: Dict[int, float] = defaultdict(float)
        self.person_detections: Dict[int, int] = defaultdict(int)
        
        print("✅ Staff Detector YOLO-World initialisé")
        print(f"   Classes staff: {len(self.staff_classes)}")
        print(f"   Classes client: {len(self.customer_classes)}")
    
    def detect_staff_indicators(self, image: np.ndarray) -> Dict[str, List[dict]]:
        """
        Utilise YOLO-World pour détecter les indicateurs de staff.
        
        Returns:
            Dict avec 'staff_indicators' et 'customer_indicators'
        """
        results = self.world_model.predict(
            source=image,
            conf=self.confidence,
            verbose=False
        )[0]
        
        indicators = {
            'staff': [],
            'customer': [],
            'person': []
        }
        
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                detection = {
                    'class': cls_name,
                    'confidence': conf,
                    'box': xyxy.tolist()
                }
                
                if cls_name in self.staff_classes:
                    indicators['staff'].append(detection)
                elif cls_name in self.customer_classes:
                    indicators['customer'].append(detection)
                elif cls_name == "person":
                    indicators['person'].append(detection)
        
        return indicators
    
    def detect_and_classify(
        self,
        image: np.ndarray,
        use_tracking: bool = True
    ) -> List[dict]:
        """
        Détecte et classifie les personnes comme staff ou client.
        
        Args:
            image: Image BGR
            use_tracking: Utiliser le tracking pour améliorer la classification
            
        Returns:
            Liste de détections avec is_staff, staff_score, etc.
        """
        # 1. Détecter les indicateurs avec YOLO-World
        indicators = self.detect_staff_indicators(image)
        
        # 2. Tracker les personnes
        if use_tracking:
            results = self.person_model.track(
                source=image,
                classes=[0],  # person only
                conf=self.confidence,
                persist=True,
                verbose=False
            )[0]
        else:
            results = self.person_model.predict(
                source=image,
                classes=[0],
                conf=self.confidence,
                verbose=False
            )[0]
        
        detections = []
        
        if results.boxes is not None:
            for box_data in results.boxes:
                box = box_data.xyxy[0].cpu().numpy().astype(int)
                person_id = int(box_data.id[0]) if box_data.id is not None else -1
                
                # Calculer le score staff basé sur les indicateurs + heuristiques
                staff_score = self._calculate_staff_score(
                    box, indicators, person_id, image
                )
                
                is_staff = staff_score >= self.staff_threshold
                
                detections.append({
                    'box': box.tolist(),
                    'id': person_id,
                    'is_staff': is_staff,
                    'staff_score': round(staff_score, 2),
                    'label': 'STAFF' if is_staff else 'Client',
                    'indicators': self._get_matching_indicators(box, indicators)
                })
        
        return detections
    
    def _calculate_staff_score(
        self,
        person_box: np.ndarray,
        indicators: Dict[str, List[dict]],
        person_id: int,
        image: np.ndarray = None
    ) -> float:
        """Calcule le score de probabilité staff basé sur les indicateurs ET heuristiques."""
        score = 0.0
        staff_matches = []
        customer_matches = []
        
        # Vérifier les indicateurs de staff qui se chevauchent avec la personne
        for ind in indicators['staff']:
            if self._boxes_overlap(person_box, ind['box'], threshold=0.2):
                staff_matches.append(ind)
                # Score élevé pour chaque indicateur staff détecté
                score += 0.5 + (0.3 * ind['confidence'])
        
        # Vérifier les indicateurs de client
        for ind in indicators['customer']:
            if self._boxes_overlap(person_box, ind['box'], threshold=0.2):
                customer_matches.append(ind)
                score -= 0.3 * ind['confidence']
        
        # Si au moins 1 indicateur staff détecté = très probablement staff
        if len(staff_matches) >= 1:
            score = max(score, 0.6)  # Score minimum de 60%
        
        # Bonus si plusieurs indicateurs staff
        if len(staff_matches) >= 2:
            score += 0.25
        
        # Indicateurs forts (servir, plateau, tablier)
        strong_indicators = ['person serving food', 'person carrying tray', 
                           'waiter', 'waitress', 'server', 'person wearing apron']
        for ind in staff_matches:
            if ind['class'] in strong_indicators:
                score += 0.15
        
        # ========== HEURISTIQUES VISUELLES (TOUJOURS APPLIQUÉES) ==========
        if image is not None:
            # 1. Vérifier uniforme sombre (noir) - THE KITCHEN = t-shirt noir
            is_dark, dark_conf = self._check_dark_uniform(image, person_box)
            if is_dark:
                score += 0.45 * dark_conf
            
            # 2. Vérifier posture debout (staff = généralement debout, clients = assis)
            is_standing, stand_conf = self._check_standing(person_box)
            if is_standing:
                score += 0.40 * stand_conf  # Poids élevé car clients sont assis
            
            # 3. Si debout = probablement staff (dans un restaurant)
            if is_standing and stand_conf > 0.6:
                score = max(score, 0.50)  # Score minimum garanti pour personne debout
            
            # 4. Si debout ET uniforme sombre = définitivement staff
            if is_standing and is_dark:
                score = max(score, 0.65)
        
        # Pénalité pour indicateurs client (assis à table, mange, boit)
        if len(customer_matches) > 0 and len(staff_matches) == 0:
            score = min(score, 0.30)
        
        # Historique du tracking (moyenne mobile)
        if person_id >= 0:
            self.person_detections[person_id] += 1
            prev_score = self.person_scores.get(person_id, score)
            score = 0.8 * score + 0.2 * prev_score
            self.person_scores[person_id] = score
        
        return max(0.0, min(1.0, score))
    
    def _check_dark_uniform(self, image: np.ndarray, box: np.ndarray) -> tuple:
        """Vérifie si la personne porte un uniforme noir (The Kitchen)."""
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        
        # Analyser la partie centrale du corps (zone du t-shirt)
        body_top = int(y1 + (y2 - y1) * 0.12)
        body_bottom = int(y1 + (y2 - y1) * 0.50)
        body_left = int(x1 + (x2 - x1) * 0.15)
        body_right = int(x1 + (x2 - x1) * 0.85)
        
        body_region = image[body_top:body_bottom, body_left:body_right]
        
        if body_region.size == 0:
            return False, 0.0
        
        # Méthode 1: Détecter le noir via HSV
        hsv = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 120, 100])  # Augmenté les seuils
        
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        black_percentage = np.sum(black_mask > 0) / black_mask.size
        
        # Méthode 2: Luminosité moyenne (fallback)
        gray = cv2.cvtColor(body_region, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Combiner les deux méthodes
        # Si noir détecté OU luminosité faible = uniforme sombre
        is_dark_hsv = black_percentage > 0.25
        is_dark_brightness = avg_brightness < 100
        
        is_dark = is_dark_hsv or is_dark_brightness
        
        if is_dark:
            # Prendre le meilleur score des deux méthodes
            hsv_conf = min(1.0, black_percentage / 0.5) if is_dark_hsv else 0
            bright_conf = max(0.4, 1.0 - (avg_brightness / 100)) if is_dark_brightness else 0
            confidence = max(hsv_conf, bright_conf)
        else:
            confidence = max(black_percentage, (100 - avg_brightness) / 100) * 0.3
        
        return is_dark, confidence
    
    def _check_standing(self, box: np.ndarray) -> tuple:
        """Vérifie si une personne est debout basé sur le ratio du bounding box."""
        x1, y1, x2, y2 = box
        height = y2 - y1
        width = x2 - x1
        
        if width <= 0:
            return False, 0.0
        
        ratio = height / width
        # Ratio abaissé à 1.3 car une personne debout a généralement un ratio > 1.3
        # Une personne assise a un ratio plus proche de 1.0 (carré)
        standing_ratio = 1.3
        
        is_standing = ratio >= standing_ratio
        
        if is_standing:
            confidence = min(1.0, (ratio - standing_ratio) / 0.7 + 0.5)
        else:
            confidence = max(0.0, ratio / standing_ratio)
        
        return is_standing, confidence
    
    def _boxes_overlap(
        self,
        box1: np.ndarray,
        box2: List[int],
        threshold: float = 0.3
    ) -> bool:
        """Vérifie si deux boxes se chevauchent."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / min(area1, area2)
        return iou >= threshold
    
    def _get_matching_indicators(
        self,
        person_box: np.ndarray,
        indicators: Dict[str, List[dict]]
    ) -> List[str]:
        """Retourne les indicateurs qui matchent avec la personne."""
        matches = []
        for category in ['staff', 'customer']:
            for ind in indicators[category]:
                if self._boxes_overlap(person_box, ind['box']):
                    matches.append(f"{category}:{ind['class']}")
        return matches
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[dict],
        show_indicators: bool = True
    ) -> np.ndarray:
        """Dessine les détections sur l'image."""
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            is_staff = det['is_staff']
            score = det['staff_score']
            
            # Couleurs
            if is_staff:
                color = (0, 255, 255)  # Jaune pour staff
            else:
                color = (255, 100, 100)  # Bleu pour clients
            
            # Box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det['label']} ({score:.0%})"
            if det['id'] >= 0:
                label = f"#{det['id']} {label}"
            
            # Background du label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1-th-10), (x1+tw+5, y1), color, -1)
            cv2.putText(output, label, (x1+2, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Afficher les indicateurs
            if show_indicators and det['indicators']:
                for i, ind in enumerate(det['indicators'][:3]):
                    cv2.putText(output, f"• {ind}", (x1, y2 + 15 + i*15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Stats
        staff_count = sum(1 for d in detections if d['is_staff'])
        client_count = len(detections) - staff_count
        stats = f"Staff: {staff_count} | Clients: {client_count}"
        cv2.putText(output, stats, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = True
    ):
        """Traite une vidéo complète."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"🎬 Traitement de: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Détecter
            detections = self.detect_and_classify(frame)
            
            # Annoter
            output = self.draw_detections(frame, detections)
            
            if writer:
                writer.write(output)
            
            if show:
                cv2.imshow("Staff Detection", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Stats
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"   Frame {frame_count} | FPS: {fps_actual:.1f}")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Terminé! {frame_count} frames traités")
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = True
    ):
        """Traite une image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Impossible de lire: {image_path}")
            return
        
        print(f"📸 Analyse de: {image_path}")
        
        detections = self.detect_and_classify(image, use_tracking=False)
        output = self.draw_detections(image, detections)
        
        # Stats
        staff_count = sum(1 for d in detections if d['is_staff'])
        client_count = len(detections) - staff_count
        print(f"   Staff détectés: {staff_count}")
        print(f"   Clients détectés: {client_count}")
        
        for det in detections:
            print(f"   - {det['label']} (score: {det['staff_score']:.0%})")
            if det['indicators']:
                print(f"     Indicateurs: {', '.join(det['indicators'])}")
        
        if output_path:
            cv2.imwrite(output_path, output)
            print(f"💾 Sauvegardé: {output_path}")
        
        if show:
            cv2.imshow("Staff Detection", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections


def main():
    parser = argparse.ArgumentParser(
        description="Détection Staff/Client avec YOLO-World"
    )
    parser.add_argument(
        "--image", "-i",
        help="Chemin vers une image à analyser"
    )
    parser.add_argument(
        "--video", "-v",
        help="Chemin vers une vidéo à analyser"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=None,
        help="Index de la caméra (ex: 0)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Chemin de sortie (image ou vidéo)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Ne pas afficher la fenêtre"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Seuil de confiance (défaut: 0.25)"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎯 THE KITCHEN - Staff Detection (YOLO-World)")
    print("=" * 50)
    
    detector = StaffDetectorYOLOWorld(confidence=args.confidence)
    
    if args.image:
        detector.process_image(
            args.image,
            output_path=args.output,
            show=not args.no_show
        )
    elif args.video:
        detector.process_video(
            args.video,
            output_path=args.output,
            show=not args.no_show
        )
    elif args.camera is not None:
        detector.process_video(
            args.camera,
            output_path=args.output,
            show=not args.no_show
        )
    else:
        # Par défaut, analyser les images augmentées
        augmented_dir = Path(__file__).parent / "augmented"
        if augmented_dir.exists():
            images = list(augmented_dir.glob("*_original.*"))[:3]
            for img in images:
                detector.process_image(str(img), show=not args.no_show)
        else:
            print("Usage: python detect_staff.py --image photo.jpg")
            print("       python detect_staff.py --video video.mp4")
            print("       python detect_staff.py --camera 0")


if __name__ == "__main__":
    main()
