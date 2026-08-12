# train_action_classifier.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler, RandomClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize,
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    RandomHorizontalFlipVideo,
)
import glob
import json

# --- CONFIGURATION ---
SORTED_CLIPS_DIR = "HAWKEYE_TRAINING_DATA/sorted_clips"
CLIP_DURATION = 4 # Seconds (must match prepare_video_data.py)
BATCH_SIZE = 2    # Start with 2. If you have a powerful GPU, you can try 4.
NUM_EPOCHS = 30   # 30-50 is a good range
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "hawkeye_custom_action_model.pt" # This is the file we'll use
LABELS_SAVE_PATH = "hawkeye_custom_action_labels.json" # To save our custom labels

def main():
    print("--- Hawkeye Action Classifier Training ---")
    
    # --- 1. Find Data and Create Labels ---
    print(f"Scanning for sorted clips in: {SORTED_CLIPS_DIR}")
    
    class_names = [d for d in os.listdir(SORTED_CLIPS_DIR) if os.path.isdir(os.path.join(SORTED_CLIPS_DIR, d))]
    class_names.sort()
    
    if not class_names:
        print(f"Error: No subfolders (labels) found in {SORTED_CLIPS_DIR}.")
        print("Please sort your clips into subfolders like 'fighting', 'neutral', etc.")
        return

    label_to_id = {label: i for i, label in enumerate(class_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    NUM_CLASSES = len(class_names)
    
    print(f"Found {NUM_CLASSES} classes: {class_names}")

    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(id_to_label, f, indent=4)
    print(f"Saved custom label map to: {LABELS_SAVE_PATH}")
    
    dataset_paths = []
    for class_name, label_id in label_to_id.items():
        class_dir = os.path.join(SORTED_CLIPS_DIR, class_name)
        for video_path in glob.glob(os.path.join(class_dir, "*.mp4")):
            dataset_paths.append((os.path.abspath(video_path), label_id))
            
    if not dataset_paths:
        print("Error: No .mp4 files found in any of the subfolders. Training cannot start.")
        return
        
    print(f"Found {len(dataset_paths)} total video clips for training.")

    # --- 2. Define Transforms (How to process the video) ---
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(8), 
                Lambda(lambda x: x / 255.0), 
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)), 
                ShortSideScale(size=256),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        ),
    )

    # --- 3. Create Dataset and DataLoader ---
    train_size = int(0.8 * len(dataset_paths))
    val_size = len(dataset_paths) - train_size
    train_paths, val_paths = random_split(dataset_paths, [train_size, val_size])
    
    print(f"Training with {len(train_paths)} clips, validating with {len(val_paths)} clips.")

    train_dataset = LabeledVideoDataset(
        labeled_video_paths=list(train_paths),
        clip_sampler=RandomClipSampler(clip_duration=CLIP_DURATION),
        transform=transform,
        decode_audio=False
    )
    
    val_dataset = LabeledVideoDataset(
        labeled_video_paths=list(val_paths),
        clip_sampler=RandomClipSampler(clip_duration=CLIP_DURATION),
        transform=transform,
        decode_audio=False
    )
    
    # --- MODIFIED: shuffle=False ---
    # The RandomClipSampler already handles shuffling.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    # --- 4. Load Pre-trained Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Note: We are using the older torch 1.13.1, so we force hub to use the main branch
    model = torch.hub.load('facebookresearch/pytorchvideo:main', 'slowfast_r50', pretrained=True)
    model = model.to(device)
    
    # --- 5. Replace the Final Layer ---
    model.blocks[6].proj = nn.Linear(model.blocks[6].proj.in_features, NUM_CLASSES)
    model.blocks[6].proj = model.blocks[6].proj.to(device)
    
    # --- 6. Start Training ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            inputs = batch["video"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            except Exception as e:
                print(f"Error during training batch: {e}")
        
        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["video"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = (100 * correct / total) if total > 0 else 0
        
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Val Accuracy: {accuracy:.2f}%")
                
    # --- 7. Save the Model ---
    torch.save(model, MODEL_SAVE_PATH)
    
    print("\n--- Training Complete ---")
    print(f"Your new action model has been saved to: {MODEL_SAVE_PATH}")
    print(f"Your new labels have been saved to: {LABELS_SAVE_PATH}")
    print("\nTo use your new model, open config.py and set:")
    print(f"CUSTOM_ACTION_MODEL_PATH = '{MODEL_SAVE_PATH}'")
    print(f"CUSTOM_ACTION_LABELS_PATH = '{LABELS_SAVE_PATH}'")

if __name__ == "__main__":
    main()