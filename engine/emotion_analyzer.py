from deepface import DeepFace
import cv2
import numpy as np

class EmotionAnalyzer:
    def __init__(self):
        print("[INFO] Warming up Emotion Analyzer...")
        # Run a dummy inference to load the model into memory (GPU/CPU)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            _ = DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')
            print("[INFO] Emotion Analyzer ready.")
        except Exception as e:
            print(f"[ERROR] Could not initialize DeepFace: {e}")

    def analyze_faces(self, frame, persons):
        """
        Detects faces in the frame and matches them to tracked Persons.
        Updates the Person's emotion if a match is found.
        """
        try:
            # 1. Detect faces and emotions
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')
            
            if isinstance(results, dict):
                results = [results]
            
            # 2. Match faces to tracked persons
            for face_data in results:
                region = face_data['region']
                face_cx = region['x'] + region['w'] / 2
                face_cy = region['y'] + region['h'] / 2
                emotion = face_data['dominant_emotion']
                
                best_match_tid = -1
                
                # Simple matching: Is the face center inside a person's bounding box?
                for tid, person in persons.items():
                    bbox = person.get_current_bbox()
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        if x1 < face_cx < x2 and y1 < face_cy < y2:
                            best_match_tid = tid
                            break
                
                # 3. Update the person object
                if best_match_tid != -1:
                    persons[best_match_tid].update_emotion(emotion)

        except Exception:
            # No faces detected
            pass
        
        return persons