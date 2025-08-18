import os
import cv2
import pandas as pd

# Paths
BASE_PATH = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection"
VIDEO_DIR = os.path.join(BASE_PATH, "DATASET", "youtube_videos(200)")
ANNOTATION_PATH = os.path.join(BASE_PATH, "annotations", "Final", "Unimodal dataset_0801_final.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "DATASET","npy_segments_unimodal")


df = pd.read_csv(ANNOTATION_PATH, encoding='latin1')
# check the existence of the output folder for just in case
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop each segment from the CSV file
for idx, row in df.iterrows():
    video_id = str(row['Video ID'])
    segment_id = str(row['Segment ID'])
    start_frame = int(row['Start frame'])
    end_frame = int(row['End frame'])

    # Prepare segment folder name
    segment_folder = os.path.join(OUTPUT_DIR, segment_id)
    expected_frames = end_frame - start_frame + 1
    corrupted = False

    # If folder exists + contains images
    if os.path.exists(segment_folder):
        jpg_files = sorted([f for f in os.listdir(segment_folder) if f.endswith('.jpg')])
        if len(jpg_files) == expected_frames:
            for jpg in jpg_files:
                img_path = os.path.join(segment_folder, jpg)
                img = cv2.imread(img_path)
                if img is None or os.path.getsize(img_path) == 0:
                    corrupted = True
                    break
                
            if not corrupted:
                print(f"[SKIP] Segment {segment_id} already complete and valid.") # continue without saving
                continue
            else:
                print(f"[RE-EXTRACT] Segment {segment_id} contains corrupted images. Reprocessing...")
                for f in jpg_files:
                    os.remove(os.path.join(segment_folder, f))
        else:
            print(f"[RE-EXTRACT] Segment {segment_id} incomplete. Reprocessing...")
    else:
        os.makedirs(segment_folder)

    # Match with video file
    matched_file = None
    for file in os.listdir(VIDEO_DIR):
        if file.startswith(f"{video_id}_"):
            matched_file = os.path.join(VIDEO_DIR, file)
            break

    if not matched_file:
        print(f"[ERROR] Video {video_id} not found.")
        continue

    # Extract frames using cv2
    cap = cv2.VideoCapture(matched_file)
    frame_num = 0
    saved_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_num > end_frame:
            break
        if frame_num >= start_frame:
            filename = f"frame_{saved_count + 1:05d}.jpg"
            filepath = os.path.join(segment_folder, filename)
            success_save = cv2.imwrite(filepath, frame)
            if not success_save:
                print(f"[ERROR] Failed to save frame {saved_count + 1} at {filepath}")
            else:
                saved_count += 1
        frame_num += 1

    cap.release()
    print(f"[DONE] Segment {segment_id} → {saved_count} frames saved.")
