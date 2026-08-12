# engine/tracker.py
from ultralytics import YOLO
from .person import Person
import time
import config

class PoseTracker:
    def __init__(self, model_path, pose_conf):
        print(f"[PoseTracker] Loading model to device: {config.AI_DEVICE}")
        self.model = YOLO(model_path)
        self.pose_conf = pose_conf

    def process_frame(self, frame, tracked_persons):
        # Run on GPU
        results = self.model.track(frame, persist=True, conf=self.pose_conf, verbose=False, device=config.AI_DEVICE)
        current_ts = time.time()
        
        updated_tids = set()
        
        try:
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                keypoints = results[0].keypoints.xy.cpu().numpy()

                for box, track_id, kps in zip(boxes, track_ids, keypoints):
                    if track_id not in tracked_persons:
                        tracked_persons[track_id] = Person(track_id)
                    
                    tracked_persons[track_id].update_pose(kps.tolist(), box.tolist(), current_ts)
                    updated_tids.add(track_id)
            
            stale_tids = set(tracked_persons.keys()) - updated_tids
            for tid in stale_tids:
                del tracked_persons[tid]

        except Exception:
            pass
        
        return tracked_persons