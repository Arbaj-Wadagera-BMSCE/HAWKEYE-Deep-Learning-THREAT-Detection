import cv2
import time
from collections import deque
import os
import threading
from datetime import datetime
from flask import (
    Flask, render_template, Response, request, jsonify, 
    send_from_directory, session, redirect, url_for
)
import firebase_admin
from firebase_admin import credentials, auth
import glob    
import shutil  
import numpy as np

# Import from our project structure
import config
from engine.tracker import PoseTracker
from engine.action_classifier import ActionClassifier
from engine.emotion_analyzer import EmotionAnalyzer
import engine.heuristics as heuristics
from utils.event_manager import EventManager
import utils.drawing as drawing
from ultralytics import YOLO

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "development-secret-key")

# --- Firebase Admin SDK Initialization ---
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    print("[INFO] Firebase Admin SDK initialized.")
except FileNotFoundError:
    print("[ERROR] 'serviceAccountKey.json' not found.")
    exit()
except Exception as e:
    print(f"[INFO] Firebase Check: {e}")

# Define the path for our new recent events gallery
RECENT_DIR = os.path.join('static', 'recent_events')
os.makedirs(RECENT_DIR, exist_ok=True)
OUTPUT_FOLDER = "events"
WEAPON_OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, "weapon_events")

# --- Global State Management ---
class VideoProcessor:
    def __init__(self):
        self.cap = None 
        print("[INFO] Loading AI models...")
        self.tracker = PoseTracker(config.POSE_MODEL_NAME, config.POSE_CONFIDENCE)
        self.action_classifier = ActionClassifier()
        self.event_manager = EventManager(self.action_classifier)
        self.weapon_detector = YOLO(config.WEAPON_MODEL_PATH)

        if config.ENABLE_EMOTION_DETECTION:
             print("[INFO] Initializing Emotion Analyzer...")
             self.emotion_analyzer = EmotionAnalyzer()
        else:
             self.emotion_analyzer = None 
        
        print("[INFO] System Ready.")
        
        self.weapon_output_dir = os.path.join(config.OUTPUT_FOLDER, "weapon_events")
        os.makedirs(self.weapon_output_dir, exist_ok=True)
        
        self.last_weapon_save_time = 0 
        
        self.tracked_persons = {}
        self.frame_buffer = deque(maxlen=int(30 * config.BUFFER_MAX_SECONDS))
        self.fps_estimates = deque(maxlen=30)
        self.frame_count = 0
        self.lock = threading.Lock() 

global_frame_data = {"frame": None, "lock": threading.Lock(), "processed_emotions": {}}
processor = VideoProcessor()

# --- Auth/Routes ---
@app.route('/auth')
def auth_page():
    if session.get('is_logged_in'): return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    try:
        decoded_token = auth.verify_id_token(data.get('idToken'))
        user = auth.get_user(decoded_token['uid'])
        session['user_name'] = user.display_name or user.email
        session['is_logged_in'] = True
        return jsonify({"status": "success"})
    except: return jsonify({"status": "error"}), 401

@app.route('/api/signup', methods=['POST'])
def api_signup():
    d = request.get_json()
    try:
        auth.create_user(email=d['email'], password=d['password'], display_name=d['username'])
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout(): session.clear(); return jsonify({"status": "success"})

@app.route('/')
def index(): return render_template('index.html', user_name=session.get('user_name')) if session.get('is_logged_in') else redirect(url_for('auth_page'))

@app.route('/statistics')
def statistics_page(): return render_template('statistics.html') if session.get('is_logged_in') else redirect(url_for('auth_page'))

@app.route('/get_recent_events')
def get_recent_events():
    if not session.get('is_logged_in'): return jsonify({"status": "error"}), 401
    try:
        files = glob.glob(os.path.join(WEAPON_OUTPUT_DIR, "*.jpg")) + glob.glob(os.path.join(config.OUTPUT_FOLDER, "event_*", "DANGER_FRAME_*.jpg"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        for f in glob.glob(os.path.join(RECENT_DIR, "*.jpg")): os.remove(f)
        res = []
        for p in files[:4]:
            name = os.path.basename(p)
            shutil.copy(p, os.path.join(RECENT_DIR, name))
            res.append(name)
        return jsonify({"status": "success", "images": res})
    except: return jsonify({"status": "error", "images": []})

@app.route('/start_stream', methods=['POST'])
def start_stream():
    if not session.get('is_logged_in'): return jsonify({"status": "error"}), 401
    d = request.get_json()
    
    with processor.lock:
        if processor.cap: processor.cap.release()

        if d.get('source_type') == 'camera':
            src = int(d.get('cam_index')) if str(d.get('cam_index')).isdigit() else d.get('cam_index')
        elif d.get('source_type') == 'upload':
            # [FIXED] Corrected path construction to avoid server crash
            base_dir = os.path.dirname(os.path.abspath(__file__))
            src = os.path.join(base_dir, "uploads", "temp_video.mp4")
            
            if not os.path.exists(src): 
                print(f"[ERROR] File not found at: {src}")
                return jsonify({"status": "error", "message": "File not found. Please upload again."})
        
        print(f"[INFO] Starting stream: {src}")
        processor.cap = cv2.VideoCapture(src)
        processor.tracked_persons = {}
        processor.frame_count = 0
        processor.fps_estimates.clear()
        processor.frame_buffer.clear()
        
        if not processor.cap.isOpened(): return jsonify({"status": "error", "message": f"Could not open source: {src}"})

    return jsonify({"status": "success"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    if not session.get('is_logged_in'): return jsonify({"status": "error"}), 401
    with processor.lock:
        if processor.cap: processor.cap.release(); processor.cap = None
    return jsonify({"status": "success"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if not session.get('is_logged_in'): 
        return jsonify({"status": "error", "message": "Not logged in"}), 401
    
    print("[INFO] Upload request received...")

    with processor.lock:
        if processor.cap:
            print("[INFO] Releasing active video capture to unlock file...")
            processor.cap.release()
            processor.cap = None

    f = request.files.get('video') or request.files.get('file')
    
    if f: 
        try:
            # [FIXED] Using absolute path for uploads folder
            upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            save_path = os.path.join(upload_dir, "temp_video.mp4")
            
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except PermissionError:
                    return jsonify({"status": "error", "message": "Server file is locked. Stop stream first."})

            f.save(save_path)
            print(f"[INFO] File saved to {save_path}")
            return jsonify({"status": "success", "message": f"Uploaded {f.filename}. Click 'Start From Upload'."})
            
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
            return jsonify({"status": "error", "message": str(e)})
            
    return jsonify({"status": "error", "message": "No file provided"})

# --- Video Feed Route (The Core AI Loop) ---
def generate_frames():
    last_boxes, last_confs, last_clss = [], [], []
    AI_SKIP = 3
    
    # [FIXED] Explicitly define the classes we want to detect (overriding config)
    # 43: Knife, 76: Scissors, 67: Cell Phone, 42: Fork
    TARGET_CLASSES = [43, 76, 67, 42]
    
    # [FIXED] Custom Label Map for display (Renaming Fork -> Spoke)
    CUSTOM_LABELS = {
        43: 'KNIFE',
        76: 'SCISSORS',
        67: 'MOBILE',
        42: 'SPOKE'
    }

    while True:
        with processor.lock:
            if not processor.cap:
                is_streaming = False
            else:
                is_streaming = True
                ret, frame = processor.cap.read()
                if not ret: processor.cap.release(); processor.cap = None; continue
        
        if not is_streaming:
            time.sleep(0.1)
            blank = np.zeros((480, 640, 3), np.uint8); cv2.putText(blank, "STREAM STOPPED", (180, 240), 0, 1, (100,100,100), 2)
            yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', blank)[1].tobytes() + b'\r\n')
            continue

        start_t = time.time()
        
        if processor.frame_count % 15 == 0: 
            with global_frame_data["lock"]: global_frame_data["frame"] = frame.copy()

        with global_frame_data["lock"]:
            new_emotions = global_frame_data["processed_emotions"]

        with processor.lock:
            processor.frame_buffer.append((frame.copy(), start_t))
            processor.tracked_persons = processor.tracker.process_frame(frame, processor.tracked_persons)
            
            for tid, emo in new_emotions.items():
                if tid in processor.tracked_persons: processor.tracked_persons[tid].update_emotion(emo)

            # --- 3. Weapon Detection (Updated for Fork/Mobile) ---
            if processor.frame_count % AI_SKIP == 0:
                try:
                    # [FIXED] Passed TARGET_CLASSES to detect forks and mobile
                    res = processor.weapon_detector.predict(frame, conf=config.WEAPON_CONFIDENCE, classes=TARGET_CLASSES, verbose=False, device=config.AI_DEVICE)
                    
                    if len(res[0].boxes) > 0:
                        last_boxes = res[0].boxes.xyxy.cpu().numpy()
                        last_confs = res[0].boxes.conf.cpu().numpy()
                        last_clss = res[0].boxes.cls.cpu().numpy()
                    else:
                        last_boxes, last_confs, last_clss = [], [], []
                except Exception as e:
                    print(f"[ERROR] Weapon Detector Inference Failed: {e}")
                    last_boxes, last_confs, last_clss = [], [], []

            # 4. Logic & Saving (Weapon Snapshot)
            if len(last_boxes) > 0 and (start_t - processor.last_weapon_save_time > config.WEAPON_SAVE_COOLDOWN_SECONDS):
                try:
                    save_f = frame.copy()
                    max_conf = 0.0
                    main_label = "DETECTED"
                    
                    for b, c, cl in zip(last_boxes, last_confs, last_clss):
                        x1, y1, x2, y2 = map(int, b)
                        if c > max_conf: max_conf = c 
                        
                        # Use custom label
                        class_id = int(cl)
                        label_text = CUSTOM_LABELS.get(class_id, processor.weapon_detector.names[class_id].upper())
                        main_label = label_text

                        cv2.rectangle(save_f, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(save_f, f"{label_text} {int(c*100)}%", (x1, y1-10), 0, 0.7, (255, 0, 255), 2)
                    
                    fn = f"{main_label}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jpg"
                    cv2.imwrite(os.path.join(processor.weapon_output_dir, fn), save_f)
                    processor.last_weapon_save_time = start_t
                    print(f"[INFO] Saved snapshot: {fn}")
                except Exception as e:
                    print(f"[ERROR] Failed to save snapshot: {e}")

            # 5. Actions
            triggers = heuristics.check_interactions(processor.tracked_persons)
            processor.event_manager.process_triggers(triggers, processor.frame_buffer)
            
            # 6. Drawing
            for p in processor.tracked_persons.values(): drawing.draw_person(frame, p)
            
            if len(processor.event_manager.confirmed_events) > 0 and (start_t - processor.event_manager.confirmed_events[-1]['timestamp'] < 5.0):
                evt = processor.event_manager.confirmed_events[-1]
                drawing.draw_danger_banner(frame, f"DANGER: {evt['confirmed_action'].upper()} ({int(evt['confidence']*100)}%)")
                drawing.draw_confirmed_event(frame, evt, processor.tracked_persons)

            # Draw Detections with Custom Names
            if len(last_boxes) > 0:
                try:
                    for b, c, cl in zip(last_boxes, last_confs, last_clss):
                        x1, y1, x2, y2 = map(int, b)
                        class_id = int(cl)
                        # [FIXED] Use custom label map (Spoke/Mobile)
                        label_text = CUSTOM_LABELS.get(class_id, processor.weapon_detector.names[class_id].upper())
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(frame, f"{label_text} {int(c*100)}%", (x1, y1-10), 0, 0.6, (255, 0, 255), 2)
                except Exception as e:
                    print(f"[ERROR] Drawing Failed: {e}")

            processor.frame_count += 1
            processor.fps_estimates.append(1.0 / (time.time() - start_t))
            drawing.draw_hud(frame, sum(processor.fps_estimates)/len(processor.fps_estimates), [])

        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

@app.route("/video_feed")
def vid(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def emotion_worker_thread():
    global processor, global_frame_data
    while True:
        time.sleep(1.5)
        with global_frame_data["lock"]:
            f = global_frame_data["frame"]
            global_frame_data["frame"] = None
        
        if f is not None and processor.emotion_analyzer:
            try:
                ps = processor.tracked_persons.copy()
                if ps:
                    res = processor.emotion_analyzer.analyze_faces(f, ps)
                    with processor.lock:
                        for tid, p in res.items():
                            if tid in processor.tracked_persons:
                                processor.tracked_persons[tid].update_emotion(p.emotion)
            except: pass

if __name__ == '__main__':
    threading.Thread(target=emotion_worker_thread, daemon=True).start()
    app.run(debug=False, threaded=True, use_reloader=False)