# utils/drawing.py
import cv2
import numpy as np

def draw_person(frame, person):
    """Draws bounding box, state, and emotion for a single person."""
    bbox = person.get_current_bbox()
    if bbox is None:
        return
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # --- UPDATED: Color Logic for High-Level Intent ---
    color = (0, 255, 0) # Default Green
    status_label = ""
    
    if person.state == "SUSPICIOUS": color = (0, 165, 255); status_label = "SUSPICIOUS"
    if person.state == "STEALING": color = (0, 50, 255); status_label = "STEALING"
    if person.state == "SNEAKING": color = (0, 255, 255); status_label = "SNEAKING"
    if person.state == "PEAKING": color = (100, 100, 255); status_label = "PEAKING"
    
    # Overwrite if a physical threat is confirmed (handled in app.py)
    # The draw_confirmed_event function will draw RED over this if needed.

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Label Format: Intent / Emotion
    label = f"ID:{person.track_id} | {status_label} / {person.emotion}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_confirmed_event(frame, event, persons):
    """Highlights the attacker with a RED box when a threat is confirmed."""
    attacker_id = event['trigger']['attacker_id']
    for tid, person in persons.items():
        if tid == attacker_id:
            bbox = person.get_current_bbox()
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, "PHYSICAL THREAT", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def draw_danger_banner(frame, text="VIOLENCE DETECTED"):
    """Draws a large red banner at the top of the screen."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

def draw_hud(frame, fps, confirmed_events):
    """Draws FPS counter."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)