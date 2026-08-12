# train_weapon_detector.py
import os
from roboflow import Roboflow
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
# Get your API key from Roboflow: https://app.roboflow.com/account/credentials
ROBOFLOW_API_KEY = "YOUR_PRIVATE_API_KEY_HERE" 

# Get these from your Roboflow project URL
# e.g., https://app.roboflow.com/YOUR_WORKSPACE/YOUR_PROJECT/VERSION
ROBOFLOW_WORKSPACE = "your_workspace_name"
ROBOFLOW_PROJECT = "your_project_name" # e.g., "hawkeye-weapons"
ROBOFLOW_VERSION = 1 # The version number of your dataset

# Training settings
TRAINING_EPOCHS = 75
IMAGE_SIZE = 640

def main():
    print("--- Hawkeye Weapon Detector Training ---")
    
    # --- 2. Download your annotated dataset from Roboflow ---
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
    except Exception as e:
        print(f"Error downloading dataset from Roboflow: {e}")
        print("Please ensure your API key, workspace, and project names are correct.")
        return

    # --- 3. Load the pre-trained model ---
    model = YOLO('yolov8x.pt') # Start from the powerful 'x' model
    print(f"\nModel loaded. Starting training on dataset at: {dataset.location}")

    # --- 4. Train the new model ---
    try:
        results = model.train(
            data=f'{dataset.location}/data.yaml',
            epochs=TRAINING_EPOCHS,
            imgsz=IMAGE_SIZE,
            patience=10 # Stop early if no improvement
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

    # --- 5. Finish ---
    print("\n--- Training Complete ---")
    print("Your new specialist model is saved in the 'runs/detect/train/weights/' folder.")
    print("The best model is named 'best.pt'.")
    print("\nTo use your new model:")
    print("1. Open config.py")
    print("2. Change WEAPON_MODEL_PATH to: 'runs/detect/train/weights/best.pt'")

if __name__ == "__main__":
    main()