# engine/person.py
from collections import deque
import numpy as np
from config import HISTORY_MAXLEN, AGGRESSIVE_POSE_FRAMES, SUSPICIOUS_EMOTIONS

class Person:
    def __init__(self, track_id):
        self.track_id = track_id
        self.keypoints_history = deque(maxlen=HISTORY_MAXLEN)
        self.bbox_history = deque(maxlen=2)
        self.state = "NEUTRAL"
        self.emotion = "neutral"
        self.behavior_intent = "Normal" # <-- NEW
        self.aggression_counter = 0

    def update_pose(self, keypoints, bbox, timestamp):
        self.keypoints_history.append({'kps': keypoints, 'ts': timestamp})
        self.bbox_history.append(bbox)
        self._update_state()

    def update_emotion(self, emotion_str):
        self.emotion = emotion_str
        self._update_state()

    def update_behavior(self, intent_str): # <-- NEW METHOD
        self.behavior_intent = intent_str
        self._update_state()

    def _update_state(self):
        is_suspicious_emotion = self.emotion in SUSPICIOUS_EMOTIONS
        has_aggressive_pose = False
        
        # ... (Pose analysis logic remains the same) ...
        if self.keypoints_history:
            kps = self.keypoints_history[-1]['kps']
            if len(kps) > 10:
                try:
                    l_sh_y, r_sh_y = kps[5][1], kps[6][1]
                    avg_sh_y = (l_sh_y + r_sh_y) / 2
                    l_wr_y, r_wr_y = kps[9][1], kps[10][1]
                    if (l_wr_y < avg_sh_y) or (r_wr_y < avg_sh_y):
                        has_aggressive_pose = True
                except: pass
        
        if has_aggressive_pose:
            self.aggression_counter = min(AGGRESSIVE_POSE_FRAMES, self.aggression_counter + 1)
        else:
            self.aggression_counter = max(0, self.aggression_counter - 1)
        
        # --- PRIORITY LOGIC ---
        # 1. Stealing/Sneaking is the highest priority threat
        if self.behavior_intent in ["Stealing", "Sneaking", "Peaking"]:
            self.state = self.behavior_intent.upper()
        # 2. Otherwise, check for aggression
        elif self.aggression_counter >= AGGRESSIVE_POSE_FRAMES or is_suspicious_emotion:
            self.state = "SUSPICIOUS"
        # 3. Default
        elif self.aggression_counter == 0:
            self.state = "NEUTRAL"

    def get_last_keypoints(self):
        return self.keypoints_history[-1] if self.keypoints_history else None

    def get_current_bbox(self):
        return self.bbox_history[-1] if self.bbox_history else None