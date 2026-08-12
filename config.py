# config.py

# -- Model Configuration --
POSE_MODEL_NAME = "yolov8m-pose.pt"
AI_DEVICE = 0 
POSE_CONFIDENCE = 0.50 
WEAPON_MODEL_PATH = "yolov8x.pt" 
WEAPON_CONFIDENCE = 0.40
SUSPICIOUS_CLASS_NAMES = {"knife", "scissors", "gun", "hammer", "bat", "handgun", "rifle"}

# -- Facial Emotion Recognition --
ENABLE_EMOTION_DETECTION = True
SUSPICIOUS_EMOTIONS = {"angry", "fear", "sad"}

# -- Action Recognition (Stage 2) --
ACTION_MODEL_NAME = 'slowfast_r50'
CUSTOM_ACTION_MODEL_PATH = "hawkeye_custom_action_model.pt"
CUSTOM_ACTION_LABELS_PATH = "hawkeye_custom_action_labels.json"
ACTION_CONFIDENCE_THRESHOLD = 0.70
ACTION_CLIP_DURATION_SECONDS = 4
# --- UPDATED: Use the new, unified labels ---
ACTION_LABELS_OF_INTEREST = {"fighting", "punching", "slapping", "violence", "abuse"} 
CRITICAL_BANNER_TEXT = "VIOLENCE DETECTED" # <-- NEW: The word you requested

# -- Heuristic Triggers (Stage 1) --
ELBOW_EXTENSION_VEL_THRESH = 4.0
WRIST_SPEED_THRESH = 1200.0
ANKLE_SPEED_THRESH = 1000.0
PROXIMITY_PIXELS = 120.0
HEAD_PROXIMITY_SLAP = 80.0 # <-- NEW: Hand-to-head distance threshold

# -- Person State --
HISTORY_MAXLEN = 30
AGGRESSIVE_POSE_FRAMES = 5

# -- Output Settings --
EVENT_COOLDOWN_SECONDS = 5.0
OUTPUT_FOLDER = "events"
WEAPON_SAVE_COOLDOWN_SECONDS = 5.0 

# -- Display --
WINDOW_NAME = "Hawkeye v3.3"
BUFFER_MAX_SECONDS = 10


# config.py

# ... (other settings) ...
WEAPON_CONFIDENCE = 0.45
SUSPICIOUS_CLASS_NAMES = {"knife", "scissors", "gun", "hammer", "bat", "handgun", "rifle"}

# --- NEW BEHAVIOR RECOGNITION CLASSES ---
BEHAVIOR_CLASSES = ["Normal", "Peaking", "Sneaking", "Stealing"] 
BEHAVIOR_MODEL_PATH = "yolov8s.pt" # <-- Placeholder. Replace with your custom trained model.
BEHAVIOR_CONFIDENCE = 0.50
# ----------------------------------------