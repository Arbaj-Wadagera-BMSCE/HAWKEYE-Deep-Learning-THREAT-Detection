# prepare_video_data.py
import os
import cv2
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = "HAWKEYE_TRAINING_DATA/action_videos"
OUTPUT_DIR = "HAWKEYE_TRAINING_DATA/clips_to_sort"
CLIP_DURATION_SECONDS = 4 # How long each clip should be
CLIP_OVERLAP_SECONDS = 1  # 1 second overlap to create more data

def main():
    print(f"--- Hawkeye Video Clip Preparer ---")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source folder not found: {SOURCE_DIR}")
        print("Please create it and add your .mp4 files.")
        return

    # Clean and create output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing old clips from {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving new clips to: {OUTPUT_DIR}")

    clip_counter = 0

    for video_file in os.listdir(SOURCE_DIR):
        if not video_file.endswith(".mp4"):
            continue
        
        video_path = os.path.join(SOURCE_DIR, video_file)
        print(f"Processing {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure FPS is valid
        if fps <= 0:
            print(f"  Skipping video {video_file}, invalid FPS: {fps}")
            cap.release()
            continue
            
        clip_frames = int(CLIP_DURATION_SECONDS * fps)
        # We need at least 1 frame to step forward
        step_frames = int((CLIP_DURATION_SECONDS - CLIP_OVERLAP_SECONDS) * fps)
        if step_frames <= 0:
             step_frames = 1 
        
        frame_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            
            # When the buffer is full (4 seconds of frames)
            if len(frame_buffer) == clip_frames:
                # Save this clip
                clip_filename = f"clip_{clip_counter:05d}.mp4"
                clip_path = os.path.join(OUTPUT_DIR, clip_filename)
                
                try:
                    # Get frame size from the first frame in the buffer
                    h, w, _ = frame_buffer[0].shape
                    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    for f in frame_buffer:
                        writer.write(f)
                    writer.release()
                    clip_counter += 1
                except Exception as e:
                    print(f"  Error writing clip {clip_filename}: {e}")
                
                # Remove the stepped frames from the buffer
                frame_buffer = frame_buffer[step_frames:]
        
        cap.release()
            
    print("\n--- Video Preparation Complete ---")
    print(f"Created {clip_counter} clips in '{OUTPUT_DIR}'.")
    print(f"\nNEXT STEP: Manually sort these clips into new subfolders (e.g., 'fighting', 'neutral') to prepare for training.")

if __name__ == "__main__":
    main()