import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
FRAME_ROOT = r"D:\UK\00. 2024 QMUL\00. Course\Project\SAV-VIOLENCEDETECTION\Dataset\Frames_unimodal"
NPY_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\SAV-VIOLENCEDETECTION\Dataset\npy_segments_unimodal"
IMG_SIZE = (224, 224)
MAX_FRAMES = 320

os.makedirs(NPY_DIR, exist_ok=True)
segment_folders = sorted(os.listdir(FRAME_ROOT))

for folder in tqdm(segment_folders, desc='Converting segments'):
    folder_path = os.path.join(FRAME_ROOT, folder)
    save_path = os.path.join(NPY_DIR, f"{folder}.npy")

    if not os.path.isdir(folder_path):
        continue

    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    frames = []

    for fname in frame_files[:MAX_FRAMES]:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        frames.append(img)

    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8))

    segment_array = np.stack(frames)

    with open(save_path, 'wb') as f:
        np.save(f, segment_array)

    del segment_array
