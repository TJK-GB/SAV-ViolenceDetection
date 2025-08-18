import os
import cv2
import pandas as pd
import numpy as np

# === FIXED PATHS ===
BASE_PATH = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject"
VIDEO_DIR = os.path.join(BASE_PATH, "DATASET", "youtube_videos")
ANNOTATION_PATH = os.path.join(BASE_PATH, "annotations", "Final", "Unimodal+video_source dataset_0805_ver_1.csv")
SAVE_PATH = os.path.join(BASE_PATH, "DATASET", "00. Actual Dataset", "error_colab")

# === LOAD DATA ===
annotation_df = pd.read_csv(ANNOTATION_PATH, encoding='latin1')

# === ENSURE OUTPUT DIR EXISTS ===
os.makedirs(SAVE_PATH, exist_ok=True)

# === LOOP OVER ALL SEGMENTS ===
for idx, row in annotation_df.iterrows():
    segment_id = str(row['Segment ID'])
    video_id = str(row['Video ID'])
    start_frame = int(row['Start frame'])
    end_frame = int(row['End frame'])
    expected_frames = end_frame - start_frame + 1

    npy_path = os.path.join(SAVE_PATH, f"{segment_id}.npy")

    # === SKIP IF ALREADY EXISTS ===
    if os.path.exists(npy_path):
        try:
            frames_np = np.load(npy_path)
            if frames_np.shape[0] == expected_frames:
                print(f"[SKIP] Segment {segment_id} already exists with correct shape.")
                continue
            else:
                print(f"[REPROCESS] Segment {segment_id} exists but has incorrect shape: {frames_np.shape[0]} vs {expected_frames}")
        except Exception as e:
            print(f"[REPROCESS] Segment {segment_id} exists but failed to load: {e}")

    # === MATCH VIDEO FILE ===
    matched_file = None
    for file in os.listdir(VIDEO_DIR):
        if file.startswith(f"{video_id}_"):
            matched_file = os.path.join(VIDEO_DIR, file)
            break

    if not matched_file:
        print(f"[ERROR] Video file for {video_id} not found.")
        continue

    # === EXTRACT FRAMES ===
    frames = []
    cap = cv2.VideoCapture(matched_file)
    frame_num = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_num > end_frame:
            break
        if frame_num >= start_frame:
            resized = cv2.resize(frame, (224, 224))
            frames.append(resized)
        frame_num += 1

    cap.release()

    # === SAVE .NPY ===
    if len(frames) == expected_frames:
        frames_np = np.array(frames, dtype=np.uint8)
        np.save(npy_path, frames_np)
        print(f"[DONE] Segment {segment_id} → {frames_np.shape} saved.")
    else:
        print(f"[ERROR] Segment {segment_id} → only {len(frames)} frames found, expected {expected_frames}")
