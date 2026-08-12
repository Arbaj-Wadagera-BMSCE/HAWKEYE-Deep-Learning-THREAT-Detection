# engine/heuristics.py
import math
import numpy as np
from config import (ELBOW_EXTENSION_VEL_THRESH, WRIST_SPEED_THRESH, 
                    ANKLE_SPEED_THRESH, PROXIMITY_PIXELS, HEAD_PROXIMITY_SLAP)

def _euclid(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _calculate_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

def check_interactions(persons):
    triggers = []
    tids = list(persons.keys())

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            p1 = persons[tids[i]]
            p2 = persons[tids[j]]

            if len(p1.keypoints_history) < 2 or len(p2.keypoints_history) < 2:
                continue

            # Check both directions
            trigger1 = _check_single_person_aggression(p1, p2)
            if trigger1: triggers.append(trigger1)

            trigger2 = _check_single_person_aggression(p2, p1)
            if trigger2: triggers.append(trigger2)
    
    return triggers

def _check_single_person_aggression(attacker, target):
    """
    Check for physical violence (Abuse, Slaps, Punches).
    """
    # --- CRITICAL FIX: REMOVED STATE CHECK ---
    # We now check for violence regardless of whether the user is "Suspicious" or "Neutral".
    # Action defines the state, not the other way around.

    kps_curr = attacker.keypoints_history[-1]['kps']
    ts_curr = attacker.keypoints_history[-1]['ts']
    kps_prev = attacker.keypoints_history[-2]['kps']
    ts_prev = attacker.keypoints_history[-2]['ts']
    
    dt = max(1e-6, ts_curr - ts_prev)

    target_kps = target.get_last_keypoints()['kps']
    if not target_kps: return None
    
    # Define Target Body Parts (Nose, Shoulders, Center Body)
    target_head = target_kps[0] 
    target_l_shoulder = target_kps[5]
    target_r_shoulder = target_kps[6]
    target_torso_center = ((target_kps[5][0] + target_kps[12][0]) / 2, (target_kps[5][1] + target_kps[12][1]) / 2)
    
    # Vulnerable points list
    vulnerable_points = [target_head, target_l_shoulder, target_r_shoulder, target_torso_center]

    try:
        # Check all hands
        for w_idx in [9, 10]: # Left Wrist (9), Right Wrist (10)
            wrist_pos = kps_curr[w_idx]
            wrist_v = _euclid(kps_curr[w_idx], kps_prev[w_idx]) / dt
            
            # --- 1. GENERAL ABUSE / HITTING HEURISTIC ---
            # Threshold: 250.0 (Sensitive for "beating" motions)
            if wrist_v > 250.0:
                for point in vulnerable_points:
                    dist = _euclid(wrist_pos, point)
                    
                    # Generous proximity check (approx 1.5x head size)
                    if dist < (HEAD_PROXIMITY_SLAP * 1.5):
                        return {
                            "type": "slap_candidate", 
                            "attacker_id": attacker.track_id, 
                            "target_id": target.track_id,
                            "confidence": 0.95 # High confidence to force detection
                        }

            # --- 2. FAST PUNCH HEURISTIC ---
            if wrist_v > 650.0: 
                if w_idx == 9: # Left arm
                    sh, el = 5, 7
                else: # Right arm
                    sh, el = 6, 8
                    
                angle_curr = _calculate_angle(kps_curr[sh], kps_curr[el], kps_curr[w_idx])
                angle_prev = _calculate_angle(kps_prev[sh], kps_prev[el], kps_prev[w_idx])
                elbow_av = (angle_curr - angle_prev) / dt

                is_punch = (wrist_v > WRIST_SPEED_THRESH and elbow_av < -ELBOW_EXTENSION_VEL_THRESH)
                
                if is_punch and _euclid(wrist_pos, target_torso_center) < PROXIMITY_PIXELS:
                    return {
                        "type": "punch_candidate", 
                        "attacker_id": attacker.track_id, 
                        "target_id": target.track_id,
                        "confidence": 0.95
                    }

        # --- 3. KICK HEURISTIC ---
        for an_idx in [15, 16]: # Left Ankle (15), Right Ankle (16)
            ankle_v = _euclid(kps_curr[an_idx], kps_prev[an_idx]) / dt
            
            if ankle_v > ANKLE_SPEED_THRESH and _euclid(kps_curr[an_idx], target_torso_center) < PROXIMITY_PIXELS:
                return {
                    "type": "kick_candidate", 
                    "attacker_id": attacker.track_id, 
                    "target_id": target.track_id,
                    "confidence": 0.9
                }
                
    except IndexError:
        pass 

    return None