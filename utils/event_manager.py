# utils/event_manager.py
import time
import os
import cv2
import json
import threading
from collections import deque
from config import (EVENT_COOLDOWN_SECONDS, ACTION_CLIP_DURATION_SECONDS,
                    ACTION_CONFIDENCE_THRESHOLD, ACTION_LABELS_OF_INTEREST, OUTPUT_FOLDER)
from . import drawing

class EventManager:
    """Manages heuristic triggers, (optional) action classification, and event saving."""
    def __init__(self, action_classifier):
        self.action_classifier = action_classifier 
        self.last_event_times = {} 
        self.classification_queue = deque()
        self.confirmed_events = deque(maxlen=10) 
        
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    def process_triggers(self, triggers, frame_buffer):
        """Receives triggers from heuristics and queues them."""
        current_ts = time.time()
        for trigger in triggers:
            attacker_id = trigger['attacker_id']
            if current_ts - self.last_event_times.get(attacker_id, 0) < EVENT_COOLDOWN_SECONDS:
                continue
            
            print(f"[INFO] Heuristic trigger: {trigger['type']} by ID {attacker_id}. Queuing for processing.")
            clip = self.get_clip_from_buffer(frame_buffer)
            if clip:
                self.classification_queue.append((clip, trigger, current_ts))
            self.last_event_times[attacker_id] = current_ts

    def _process_queue(self):
        """Worker thread function to process clips from the queue."""
        while True:
            if self.classification_queue:
                clip, trigger, timestamp = self.classification_queue.popleft()
                
                if self.action_classifier:
                    # Run Stage 2 Verification
                    print("[INFO] Verifying trigger with action classifier...")
                    action_label, confidence = self.action_classifier.classify(clip)
                    print(f"[INFO] Classifier result: '{action_label}' (Confidence: {confidence:.2f})")

                    if action_label in ACTION_LABELS_OF_INTEREST and confidence >= ACTION_CONFIDENCE_THRESHOLD:
                        print(f"[ALERT] CONFIRMED AGGRESSION: {action_label.upper()} by ID {trigger['attacker_id']}")
                        event_data = {"trigger": trigger, "confirmed_action": action_label, "confidence": confidence, "timestamp": timestamp}
                        self.confirmed_events.append(event_data)
                        self._save_event(clip, event_data, clip[-1].copy())
                    else:
                        self.last_event_times[trigger['attacker_id']] = 0 # Reset cooldown on false alarm
                else:
                    # Fallback if classifier disabled
                    action_label = trigger.get("type", "suspicious_action").replace("_candidate", "") 
                    confidence = 1.0 
                    print(f"[ALERT] HEURISTIC TRIGGER CONFIRMED (No Stage 2): {action_label.upper()} by ID {trigger['attacker_id']}")
                    event_data = {"trigger": trigger, "confirmed_action": action_label, "confidence": confidence, "timestamp": timestamp}
                    self.confirmed_events.append(event_data)
                    self._save_event(clip, event_data, clip[-1].copy()) 
            else:
                time.sleep(0.1) 

    def _save_event(self, clip_frames, event_data, annotated_frame):
        """Saves the video clip, metadata, and annotated frame."""
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event_data['timestamp']))
        action = event_data['confirmed_action']
        accuracy_percent = int(event_data['confidence'] * 100)
        
        event_folder_name = f"event_{ts_str}_{action}_{accuracy_percent}pct"
        event_dir = os.path.join(OUTPUT_FOLDER, event_folder_name)
        os.makedirs(event_dir, exist_ok=True)

        # Metadata
        meta_path = os.path.join(event_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(event_data, f, indent=4)

        # Video
        clip_path = os.path.join(event_dir, "clip.mp4")
        if clip_frames: 
            h, w = clip_frames[0].shape[:2]
            fps = 15.0 # Standard FPS for saved clips
            writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for frame in clip_frames:
                writer.write(frame)
            writer.release()
        
        # Image (DANGER FRAME)
        danger_frame_filename = f"DANGER_FRAME_{action}_{accuracy_percent}pct.jpg"
        danger_frame_path = os.path.join(event_dir, danger_frame_filename)
        banner_text = f"DANGER: {action.upper()} ({accuracy_percent}%)"
        drawing.draw_danger_banner(annotated_frame, text=banner_text)
        cv2.imwrite(danger_frame_path, annotated_frame)
        
        print(f"[INFO] Saved event to {event_dir}")

    def get_clip_from_buffer(self, frame_buffer):
        if not frame_buffer: return None
        # Grab approx last 3-4 seconds
        num_frames = int(ACTION_CLIP_DURATION_SECONDS * 15) 
        num_frames = min(len(frame_buffer), num_frames) 
        clip_data = list(frame_buffer)[-num_frames:]
        return [f for f, ts in clip_data]