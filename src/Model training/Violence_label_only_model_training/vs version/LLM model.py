import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
from io import BytesIO
from base64 import b64encode
from sklearn.metrics import classification_report, confusion_matrix

# System configuration
API_KEY = ""  # insert your API key here
client = OpenAI(api_key=API_KEY)

CSV_PATH = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Data_split\Violence_Label_Only_split\test.csv"
NPY_DIR = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Dataset\npy_segments_unimodal"
SAVE_DIR = r"D:\UK\00. 2024 QMUL\00. Course\SAV-ViolenceDetection\Results\LLM"
SAVE_NAME = "gpt4o_segment_predictions.csv"

NUM_FRAMES = 10
IMAGE_SIZE = (224, 224)


# Image encoding function
def encode_frame_to_base64(frame):
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    if frame.ndim == 2 or frame.shape[-1] != 3:
        frame = np.stack([frame] * 3, axis=-1)
    img = Image.fromarray(frame).resize(IMAGE_SIZE)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    encoded = b64encode(img_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"}
    }


# main 
def main():
    df = pd.read_csv(CSV_PATH)
    df["Segment ID"] = df["Segment ID"].astype(str)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        segment_id = row["Segment ID"]
        true_label = int(row["Violence label(video)"]) if "Violence label(video)" in row else int(row["Violence label"])
        npy_path = os.path.join(NPY_DIR, f"{segment_id}.npy")

        if not os.path.exists(npy_path):
            print(f"[Missing] {segment_id}")
            continue

        try:
            frames = np.load(npy_path)
            total_frames = len(frames)
            indices = np.linspace(0, total_frames - 1, min(total_frames, NUM_FRAMES), dtype=int)
            sampled_frames = frames[indices]

            encoded_images = [encode_frame_to_base64(f) for f in sampled_frames]

            messages = [
                {
                    "role": "system",
                    "content": "You are a video analysis assistant. Given several frames from a video segment, determine whether any violence occurs. If yes, list the frame numbers where it happens."
                },
                {
                    "role": "user",
                    "content": [
                        *encoded_images,
                        {
                            "type": "text",
                            "text": "These are frames from a short surveillance video segment. Is there any violent action? If so, which frames (e.g. 1, 4, 8) show violence? If none, say 'None'."
                        }
                    ]
                }
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=200
            )

            reply = response.choices[0].message.content
            gpt_pred_label = 1 if "frame" in reply.lower() or "yes" in reply.lower() else 0

            results.append({
                "segment_id": segment_id,
                "gpt_pred_label": gpt_pred_label,
                "true_label": true_label,
                "gpt_response": reply
            })

        except Exception as e:
            print(f"[Error] {segment_id}: {e}")
            results.append({
                "segment_id": segment_id,
                "gpt_pred_label": "error",
                "true_label": true_label,
                "gpt_response": str(e)
            })

    # save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_out = pd.DataFrame(results)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    df_out.to_csv(save_path, index=False)
    print(f"\nSaved predictions to: {save_path}")

    # evaluate
    df_valid = df_out[df_out["gpt_pred_label"] != "error"].copy()
    df_valid["gpt_pred_label"] = df_valid["gpt_pred_label"].astype(int)
    df_valid["true_label"] = df_valid["true_label"].astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(df_valid["true_label"], df_valid["gpt_pred_label"]))

    print("\nClassification Report:")
    print(classification_report(
        df_valid["true_label"],
        df_valid["gpt_pred_label"],
        target_names=["Non-violent", "Violent"]
    ))


if __name__ == "__main__":
    main()
