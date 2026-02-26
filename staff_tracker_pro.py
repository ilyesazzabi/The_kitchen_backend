"""
THE KITCHEN - Staff Tracker Pro
Professional tracking with ByteTrack + Weighted Scoring
Temporal analysis for accurate staff/client classification
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class PersonHistory:
    """Track historical data for a person"""
    track_id: int
    first_seen: float = 0
    last_seen: float = 0
    total_frames: int = 0
    
    # Position history
    positions: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    
    # Detection history
    logo_detections: int = 0
    standing_frames: int = 0
    dark_uniform_frames: int = 0
    near_table_frames: int = 0
    counter_zone_frames: int = 0  # Time spent behind counter = STAFF
    
    # Staff lock mechanism - prevents flickering
    is_locked_staff: bool = False  # Once confirmed, stays staff
    lock_confidence: float = 0.0   # How confident we are in the lock
    
    # Scores
    current_score: float = 0.5
    smoothed_score: float = 0.5
    
    def get_duration(self) -> float:
        """Get how long this person has been tracked (seconds)"""
        return self.last_seen - self.first_seen
    
    def get_movement_speed(self) -> float:
        """Calculate average movement speed (pixels/second)"""
        if len(self.positions) < 5:
            return 0
        
        positions = self.positions[-30:]  # Last 30 positions
        total_distance = 0
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        time_span = positions[-1][2] - positions[0][2]
        return total_distance / time_span if time_span > 0 else 0
    
    def get_standing_ratio(self) -> float:
        """Ratio of frames standing vs total"""
        return self.standing_frames / self.total_frames if self.total_frames > 0 else 0
    
    def get_logo_ratio(self) -> float:
        """Ratio of frames with logo detected"""
        return self.logo_detections / self.total_frames if self.total_frames > 0 else 0


class StaffTrackerPro:
    """
    Professional staff detection with temporal tracking.
    Uses ByteTrack (built into YOLO) + weighted scoring.
    """
    
    def __init__(self, logo_model_path: str = None, fast_mode: bool = True):
        print("🎥 Initializing Staff Tracker Pro...")
        print("   with ByteTrack temporal tracking")
        
        # Load segmentation model - use smaller model for speed
        model_name = "yolo11s-seg.pt" if fast_mode else "yolo11x-seg.pt"
        self.seg_model = YOLO(model_name)
        print(f"✅ Segmentation model: {model_name} ({'FAST' if fast_mode else 'ACCURATE'})")
        
        # Load logo detector if available
        self.logo_model = None
        if logo_model_path and Path(logo_model_path).exists():
            self.logo_model = YOLO(logo_model_path)
            print(f"✅ Logo detector: {logo_model_path}")
        elif Path("kitchen_logo_detector.pt").exists():
            self.logo_model = YOLO("kitchen_logo_detector.pt")
            print("✅ Logo detector: kitchen_logo_detector.pt")
        else:
            print("ℹ️ No logo model - using heuristics")
        
        # Person tracking history
        self.persons: Dict[int, PersonHistory] = {}
        
        # Scoring weights - BALANCED APPROACH
        # Logo is important but combined with behavior
        self.weights = {
            'logo': 0.25,           # Logo on uniform
            'zone': 0.15,           # ENABLED - Counter zone
            'movement': 0.25,       # Staff moves around
            'posture': 0.20,        # Standing vs sitting
            'duration': 0.10,       # Time present
            'uniform': 0.05,        # Dark uniform
        }
        
        # Counter zone - area behind the counter (RIGHT side of frame)
        # Format: (x1%, y1%, x2%, y2%) - percentage of frame
        # Based on camera view: counter is on the right side (40% to 90% width, 10% to 60% height)
        self.counter_zone = (0.40, 0.10, 0.90, 0.60)  # Right side where counter is
        
        # Thresholds
        self.staff_threshold = 0.45  # Balanced threshold
        self.standing_ratio = 1.4
        self.dark_threshold = 100
        self.movement_threshold = 40  # Catch more movement
        self.min_tracking_time = 3.0  # Wait 3s before confident classification
        
        print(f"📊 Scoring weights: {self.weights}")
        print(f"🎯 Staff threshold: {self.staff_threshold}")
        print(f"📍 Counter zone: {self.counter_zone}")
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> dict:
        """Process a video frame with tracking"""
        if timestamp is None:
            timestamp = time.time()
        
        results = {
            'staff': [],
            'clients': [],
            'tables': [],
            'persons_tracked': 0,
            'annotated_frame': None
        }
        
        # Track people and tables with ByteTrack
        detections = self.seg_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",  # Use default ByteTrack
            classes=[0, 60],  # person, dining table
            conf=0.25,        # Low enough to catch far people
            verbose=False
        )[0]
        
        # Collect person positions first (for targeted logo detection)
        person_boxes = []
        table_boxes = []
        if detections.boxes is not None:
            for box in detections.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                if int(box.cls) == 0:  # Person
                    person_boxes.append(xyxy.tolist())
                elif int(box.cls) == 60:  # Table
                    table_boxes.append(xyxy.tolist())
        
        # Detect logos ONLY inside person regions (eliminates false positives)
        logo_boxes = self._detect_logos(frame, person_boxes)
        
        # Process each tracked person
        output_frame = frame.copy()
        active_ids = set()
        
        if detections.boxes is not None:
            for i, box in enumerate(detections.boxes):
                cls = int(box.cls)
                if cls != 0:  # Not a person
                    continue
                
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                
                
                # Get tracking ID (ByteTrack assigns this)
                track_id = int(box.id) if box.id is not None else -1
                if track_id < 0:
                    continue
                
                active_ids.add(track_id)
                
                # Get or create person history
                if track_id not in self.persons:
                    self.persons[track_id] = PersonHistory(
                        track_id=track_id,
                        first_seen=timestamp
                    )
                
                person = self.persons[track_id]
                person.last_seen = timestamp
                person.total_frames += 1
                
                # Get mask
                mask = None
                if detections.masks is not None and i < len(detections.masks.data):
                    mask_data = detections.masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask_data, (frame.shape[1], frame.shape[0])) > 0.5
                
                # Update person history with current frame data
                self._update_person_history(
                    person, frame, xyxy, mask, timestamp,
                    logo_boxes, table_boxes
                )
                
                # Calculate weighted score
                scores = self._calculate_scores(person)
                final_score = sum(scores[k] * self.weights[k] for k in self.weights)
                
                # Smooth the score (exponential moving average)
                alpha = 0.15
                person.smoothed_score = alpha * final_score + (1 - alpha) * person.smoothed_score
                person.current_score = final_score
                
                # STAFF LOCK MECHANISM - prevents flickering
                # Lock as staff based on multiple criteria
                standing_ratio_lock = person.standing_frames / max(1, person.total_frames)
                is_mostly_sitting = (
                    person.get_duration() > 3 and  # Only 3 seconds needed
                    standing_ratio_lock < 0.50  # Less than 50% standing = sitting
                )
                
                # UNLOCK if person is clearly sitting (override false lock)
                if person.is_locked_staff and is_mostly_sitting:
                    person.is_locked_staff = False
                    print(f"🔓 Person #{track_id} UNLOCKED (sitting detected)")
                
                if not person.is_locked_staff:
                    if is_mostly_sitting:
                        # Don't lock as staff - sitting people are clients
                        pass
                    # Lock by logo - require MORE evidence: 3+ logos + 5s + mostly standing
                    elif person.logo_detections >= 3 and person.get_duration() > 5 and standing_ratio_lock > 0.60:
                        person.is_locked_staff = True
                        person.lock_confidence = 1.0
                        print(f"🔒 Person #{track_id} LOCKED as STAFF (logo:{person.logo_detections} + standing:{standing_ratio_lock:.0%})")
                    
                    # Zone lock DISABLED - causes false positives on seated clients
                    # Need to calibrate zone coordinates for specific camera angle first
                    # elif person.get_duration() > 5:
                    #     zone_ratio = person.counter_zone_frames / max(1, person.total_frames)
                    #     if zone_ratio > 0.5:
                    #         person.is_locked_staff = True
                    #         person.lock_confidence = 0.9
                    #         print(f"🔒 Person #{track_id} LOCKED as STAFF (in counter zone)")
                    
                    # Lock by movement pattern (staff moves a lot)
                    elif person.get_duration() > 8 and person.get_movement_speed() > 60:
                        person.is_locked_staff = True
                        person.lock_confidence = 0.7
                        print(f"🔒 Person #{track_id} LOCKED as STAFF (movement {person.get_movement_speed():.0f}px/s over 8s)")
                
                # Determine classification
                tracking_time = person.get_duration()
                is_confident = tracking_time >= self.min_tracking_time
                
                # Override: Person who is mostly SITTING = CLIENT
                # Staff is almost always standing or walking
                standing_ratio = person.standing_frames / max(1, person.total_frames)
                is_mostly_sitting = (
                    tracking_time > 3 and  # Only need 3 seconds
                    standing_ratio < 0.50  # Less than 50% of time standing = sitting
                )
                
                if is_mostly_sitting:
                    is_staff = False  # Override - sitting = client
                elif person.is_locked_staff:
                    is_staff = True  # Locked staff stays staff!
                else:
                    is_staff = person.smoothed_score >= self.staff_threshold
                
                # Store result
                person_data = {
                    'id': track_id,
                    'box': xyxy.tolist(),
                    'score': person.smoothed_score,
                    'scores': scores,
                    'is_staff': is_staff,
                    'tracking_time': tracking_time,
                    'confident': is_confident
                }
                
                if is_staff:
                    results['staff'].append(person_data)
                    color = (0, 215, 255)  # Orange for STAFF
                else:
                    results['clients'].append(person_data)
                    color = (100, 100, 255)  # Red for Client
                
                # Draw on frame
                lock_icon = "[L]" if person.is_locked_staff else ""
                label = f"{'STAFF' if is_staff else 'Client'} #{track_id}{lock_icon}"
                score_str = f"{person.smoothed_score:.0%}"
                time_str = f"{tracking_time:.1f}s"
                
                # Box color intensity based on confidence
                if not is_confident:
                    color = tuple(int(c * 0.5) for c in color)  # Dimmer when not confident
                
                # Thicker border for locked staff
                thickness = 3 if person.is_locked_staff else 2
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(output_frame, f"{label} ({score_str})", (x1, y1-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(output_frame, time_str, (x1, y1-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw tables
        for tx1, ty1, tx2, ty2 in table_boxes:
            cv2.rectangle(output_frame, (tx1, ty1), (tx2, ty2), (100, 255, 100), 2)
            results['tables'].append([tx1, ty1, tx2, ty2])
        
        # Draw legend
        self._draw_legend(output_frame, len(results['staff']), len(results['clients']))
        
        # Cleanup old tracks
        self._cleanup_old_tracks(timestamp, active_ids)
        
        results['persons_tracked'] = len(self.persons)
        results['annotated_frame'] = output_frame
        return results
    
    def _detect_logos(self, frame: np.ndarray, person_boxes: list = None) -> list:
        """Detect logos using custom model - only inside person regions"""
        if self.logo_model is None:
            return []
        
        logo_boxes = []
        
        if person_boxes and len(person_boxes) > 0:
            # Only search for logos INSIDE detected person bounding boxes
            for px1, py1, px2, py2 in person_boxes:
                # Crop the person region
                person_crop = frame[py1:py2, px1:px2]
                if person_crop.size == 0:
                    continue
                
                # Detect logos in this person crop - HIGH threshold to avoid false positives
                results = self.logo_model.predict(person_crop, conf=0.60, verbose=False)
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # Convert coordinates back to full frame
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        logo_boxes.append([
                            xyxy[0] + px1, xyxy[1] + py1,
                            xyxy[2] + px1, xyxy[3] + py1
                        ])
        else:
            # Fallback: search whole frame (less reliable)
            results = self.logo_model.predict(frame, conf=0.25, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    logo_boxes.append(xyxy.tolist())
        
        return logo_boxes
    
    def _update_person_history(self, person: PersonHistory, frame, box, mask,
                               timestamp, logo_boxes, table_boxes):
        """Update person's historical data"""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Add position
        person.positions.append((cx, cy, timestamp))
        if len(person.positions) > 60:  # Keep last 60 positions
            person.positions = person.positions[-60:]
        
        # Check logo
        for lx1, ly1, lx2, ly2 in logo_boxes:
            lcx = (lx1 + lx2) / 2
            lcy = (ly1 + ly2) / 2
            if x1 - 50 < lcx < x2 + 50 and y1 - 50 < lcy < y2 + 50:
                person.logo_detections += 1
                break
        
        # Check posture
        h = y2 - y1
        w = x2 - x1
        if h / w >= self.standing_ratio if w > 0 else False:
            person.standing_frames += 1
        
        # Check uniform
        if mask is not None and mask.sum() > 100:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg = np.mean(gray[mask])
            if avg < self.dark_threshold:
                person.dark_uniform_frames += 1
        
        # Check near table - simplified: if person overlaps with table area = near table
        for tx1, ty1, tx2, ty2 in table_boxes:
            # Check if person center is within expanded table area
            table_center_x = (tx1 + tx2) / 2
            table_center_y = (ty1 + ty2) / 2
            dist_x = abs(cx - table_center_x)
            dist_y = abs(cy - table_center_y)
            table_width = tx2 - tx1
            table_height = ty2 - ty1
            
            # If person is close to table (within 1.5x table size)
            if dist_x < table_width * 1.5 and dist_y < table_height * 1.5:
                person.near_table_frames += 1
                break
        
        # Check if in counter zone (staff only area)
        frame_h, frame_w = frame.shape[:2]
        zone_x1 = int(self.counter_zone[0] * frame_w)
        zone_y1 = int(self.counter_zone[1] * frame_h)
        zone_x2 = int(self.counter_zone[2] * frame_w)
        zone_y2 = int(self.counter_zone[3] * frame_h)
        
        if zone_x1 < cx < zone_x2 and zone_y1 < cy < zone_y2:
            person.counter_zone_frames += 1
    
    def _calculate_scores(self, person: PersonHistory) -> dict:
        """Calculate individual criterion scores"""
        scores = {}
        
        # Logo score (most important)
        logo_ratio = person.get_logo_ratio()
        scores['logo'] = min(logo_ratio * 3, 1.0)  # Boost if logo seen
        
        # Movement score
        speed = person.get_movement_speed()
        if speed > 100:
            scores['movement'] = 1.0
        elif speed > 50:
            scores['movement'] = 0.7
        elif speed > 20:
            scores['movement'] = 0.4
        else:
            scores['movement'] = 0.1
        
        # Posture score
        standing_ratio = person.get_standing_ratio()
        scores['posture'] = min(standing_ratio * 1.2, 1.0)
        
        # Duration score (staff are present longer, not at tables)
        duration = person.get_duration()
        near_table_ratio = person.near_table_frames / person.total_frames if person.total_frames > 0 else 0
        
        if duration > 60 and near_table_ratio < 0.3:
            scores['duration'] = 1.0  # Long time, not at table = staff
        elif duration > 30 and near_table_ratio < 0.5:
            scores['duration'] = 0.6
        elif near_table_ratio > 0.7:
            scores['duration'] = 0.1  # Mostly at table = client
        else:
            scores['duration'] = 0.3
        
        # Uniform score
        uniform_ratio = person.dark_uniform_frames / person.total_frames if person.total_frames > 0 else 0
        scores['uniform'] = min(uniform_ratio * 1.5, 1.0)
        
        # Counter zone score - being in staff-only area is strong signal
        zone_ratio = person.counter_zone_frames / person.total_frames if person.total_frames > 0 else 0
        if zone_ratio > 0.5:
            scores['zone'] = 1.0  # Mostly in counter area = staff
        elif zone_ratio > 0.2:
            scores['zone'] = 0.6
        else:
            scores['zone'] = 0.0  # Not in zone, neutral
        
        return scores
    
    def _draw_legend(self, frame, staff_count, client_count):
        """Draw legend on frame"""
        cv2.rectangle(frame, (10, 10), (220, 110), (20, 20, 20), -1)
        
        cv2.circle(frame, (25, 30), 8, (0, 215, 255), -1)
        cv2.putText(frame, f"STAFF ({staff_count})", (45, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(frame, (25, 55), 8, (100, 100, 255), -1)
        cv2.putText(frame, f"Clients ({client_count})", (45, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Tracked: {len(self.persons)}", (25, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        mode = "ByteTrack + Logo" if self.logo_model else "ByteTrack + Heuristics"
        cv2.putText(frame, mode, (25, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
    
    def _cleanup_old_tracks(self, current_time: float, active_ids: set, max_age: float = 5.0):
        """Remove tracks not seen recently"""
        to_remove = []
        for track_id, person in self.persons.items():
            if track_id not in active_ids and current_time - person.last_seen > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.persons[track_id]


def run_on_video(video_path: str, output_path: str = None):
    """Process a video file"""
    tracker = StaffTrackerPro()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    print(f"🎬 Processing: {video_path}")
    print(f"   Frames: {total_frames}, FPS: {fps:.1f}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / fps
        results = tracker.process_frame(frame, timestamp)
        
        if out:
            out.write(results['annotated_frame'])
        
        # Show preview
        preview = cv2.resize(results['annotated_frame'], (960, 540))
        cv2.imshow("Staff Tracker Pro", preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        # Progress
        if frame_count % 100 == 0:
            pct = frame_count / total_frames * 100
            print(f"   Progress: {pct:.1f}% ({frame_count}/{total_frames})")
    
    elapsed = time.time() - start_time
    print(f"✅ Done! {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


def run_on_camera(camera_id: int = 0):
    """Run on live camera"""
    tracker = StaffTrackerPro()
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return
    
    print(f"📹 Live camera {camera_id}")
    print("   Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = tracker.process_frame(frame, time.time())
        
        cv2.imshow("THE KITCHEN - Staff Tracker Pro", results['annotated_frame'])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "camera":
            cam_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            run_on_camera(cam_id)
        else:
            video_path = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else None
            run_on_video(video_path, output_path)
    else:
        print("=" * 55)
        print("THE KITCHEN - Staff Tracker Pro")
        print("with ByteTrack Temporal Tracking + Weighted Scoring")
        print("=" * 55)
        print("\nUsage:")
        print("  Camera:  python staff_tracker_pro.py camera [id]")
        print("  Video:   python staff_tracker_pro.py video.mp4 [output.mp4]")
        print("\nScoring System:")
        print("  🏷️ Logo detection:    40%")
        print("  🚶 Movement patterns: 20%")
        print("  🧍 Standing posture:  15%")
        print("  ⏱️ Duration/position: 15%")
        print("  👕 Dark uniform:      10%")
        print("\n  Staff threshold: 55%")
