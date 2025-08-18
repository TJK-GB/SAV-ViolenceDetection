# class-based rewrite of the user's frame annotation GUI
# same logic, cleaner structure

import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import glob
from PIL import Image, ImageTk
import time
import threading

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame-Level Annotation Tool")
        self.root.geometry("1680x1050")

        self.BASE_PATH = r"Dataset" # replace with youtube_videos path
        self.BASE_PATH_SAVE = r"Annotations_Final\Old" #replace with where to save path
        self.VIDEO_DIR = os.path.join(self.BASE_PATH, "youtube_videos(200)")
        self.SAVE_PATH = os.path.join(self.BASE_PATH_SAVE, "annotations_test.xlsx") #saving file name #

        self.existing_df = pd.read_excel(self.SAVE_PATH) if os.path.exists(self.SAVE_PATH) else pd.DataFrame()
        self.done_files = set(self.existing_df['Filename'].tolist()) if not self.existing_df.empty else set()
        self.video_files = sorted([f for f in glob.glob(os.path.join(self.VIDEO_DIR, "*.mp4")) if os.path.basename(f) not in self.done_files])
        self.video_index = -1
        self.annotations = []

        self.cap = None
        self.is_playing = False
        self.start_frame = None
        self.end_frame = None
        self.last_end_frame = None
        self.fps = 0
        self.total_frames = 0
        self.current_frame = 0

        self.setup_ui()
        self.load_next_video()

    def setup_ui(self):
        self.violence_opts = ["", "Violent", "Non-violent", "N/A"]
        self.video_types = ["", "News", "CCTV", "Self-filmed", "Dashcam","Bodycam", "Others", "Transition", "Combination"]
        self.sound_types = ["", "Gunshot", "Screaming", "Crying", "Shouting", "Others"]
        self.text_types = ["", "Violent", "Non-violent", "N/A"]

        self.violence_var = tk.StringVar(value="")
        self.video_type_var = tk.StringVar(value="")
        self.sound_type_var = tk.StringVar(value="")
        self.text_type_var = tk.StringVar(value="")

        self.video_label = tk.Label(self.root, text="Video: ", font=("Arial", 12))
        self.video_label.grid(row=0, column=0, columnspan=4, pady=5)

        self.frame_label = tk.Label(self.root, text="Frame: 0 / 0", font=("Arial", 10))
        self.frame_label.grid(row=1, column=0, columnspan=4)

        self.canvas = tk.Label(self.root)
        self.canvas.grid(row=2, column=0, columnspan=4, pady=10)

        self.last_saved_label = tk.Label(self.root, text="Last saved: None", font=("Arial", 10), fg="green")
        self.last_saved_label.grid(row=1, column=3, columnspan=2, sticky="w", padx=10)

        self.create_dropdowns()
        self.create_text_fields()
        self.create_buttons()


    def create_dropdowns(self):
        ttk.Label(self.root, text="Violence (Video)").grid(row=3, column=0)
        ttk.OptionMenu(self.root, self.violence_var, self.violence_var.get(), *self.violence_opts).grid(row=3, column=1)

        ttk.Label(self.root, text="Video Type").grid(row=4, column=0)
        ttk.OptionMenu(self.root, self.video_type_var, self.video_type_var.get(), *self.video_types).grid(row=4, column=1)

        ttk.Label(self.root, text="Sound Type").grid(row=5, column=0)
        ttk.OptionMenu(self.root, self.sound_type_var, self.sound_type_var.get(), *self.sound_types).grid(row=5, column=1)

        ttk.Label(self.root, text="Text Violence").grid(row=6, column=0)
        ttk.OptionMenu(self.root, self.text_type_var, self.text_type_var.get(), *self.text_types).grid(row=6, column=1)

    def create_text_fields(self):
        tk.Label(self.root, text="Manual Text").grid(row=7, column=0)
        self.note_entry = tk.Entry(self.root, width=50)
        self.note_entry.grid(row=7, column=1, columnspan=3)

        tk.Label(self.root, text="Memo").grid(row=8, column=0)
        self.note_entry_memo = tk.Entry(self.root, width=60)
        self.note_entry_memo.grid(row=8, column=1, columnspan=3)

    def create_buttons(self):
        # Playback
        tk.Button(self.root, text="▶ Play", width=15, command=self.play_video).grid(row=9, column=0)
        tk.Button(self.root, text="⏸ Pause", width=15, command=self.pause_video).grid(row=9, column=1)
        tk.Button(self.root, text="⏪ 1f Back", width=15, command=lambda: self.step_frame(-1)).grid(row=9, column=2)
        tk.Button(self.root, text="⏪ 5f Back", width=15, command=lambda: self.step_frame(-5)).grid(row=10, column=2)
        tk.Button(self.root, text="⏪ 10f Back", width=15, command=lambda: self.step_frame(-10)).grid(row=11, column=2)
        tk.Button(self.root, text="⏩ 1f Forward", width=15, command=lambda: self.step_frame(1)).grid(row=9, column=3)        
        tk.Button(self.root, text="⏩ 5f Forward", width=15, command=lambda: self.step_frame(5)).grid(row=10, column=3)        
        tk.Button(self.root, text="⏩ 10f Forward", width=15, command=lambda: self.step_frame(10)).grid(row=11, column=3)
        tk.Button(self.root, text="🔁 Play from Start", width=15, command=self.play_from_start).grid(row=12, column=0, pady=5)
        tk.Label(self.root, text="Jump to Frame").grid(row=12, column=1, sticky="w")
        self.jump_frame_entry = tk.Entry(self.root, width=10)
        self.jump_frame_entry.grid(row=12, column=2, sticky="w")

        tk.Label(self.root, text="Jump to Time (mm:ss)").grid(row=13, column=1, sticky="w")
        self.jump_time_entry = tk.Entry(self.root, width=10)
        self.jump_time_entry.grid(row=13, column=2, sticky="w")
        tk.Button(self.root, text="⏩ Go", width=10, command=self.jump_to_place).grid(row=13, column=3, sticky="w")


        # Annotation
        tk.Button(self.root, text="Start Frame", width=15, command=lambda: self.set_frame('start')).grid(row=10, column=0)
        # by pressing start frame will automatically set the start frame
        tk.Button(self.root, text="End Frame", width=15, command=lambda: self.set_frame('end')).grid(row=10, column=1)
        # by pressing end frame will automatically set the end frame
        tk.Button(self.root, text="Save", width=15, command=self.save_annotation).grid(row=11, column=1)
        # automatically save to annotation file then reset the label boxes
        tk.Button(self.root, text="Next Video", width=15, command=self.load_next_video).grid(row=12, column=3)

    def update_canvas(self):
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if not ret:
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get original dimensions
        h, w, _ = img.shape
        max_w, max_h = 1200, 600  # target canvas size

        # Resize while keeping aspect ratio
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # img = cv2.resize(img, (880, 360))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.imgtk = imgtk
        self.canvas.config(image=imgtk)

        elapsed_secs = int(self.current_frame / self.fps)
        mm, ss = divmod(elapsed_secs, 60)
        time_str = f"{mm:02}:{ss:02}"
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}   |   Time: {time_str}")

    def play_video(self):
        if not self.is_playing:
            # Only set start_frame once per annotation segment
            if self.start_frame is None:
                current = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if self.last_end_frame is not None:
                    self.start_frame = self.last_end_frame + 1
                else:
                    self.start_frame = current

            self.is_playing = True

            def loop():
                last_update = time.time()
                while self.is_playing:
                    if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.total_frames - 1:
                        break
                    self.update_canvas()
                    now = time.time()
                    delay = max(1.0 / (self.fps * 0.55) - (now - last_update), 0)
                    time.sleep(delay)
                    last_update = now
            threading.Thread(target=loop, daemon=True).start()

    def play_from_start(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.start_frame = None  # Optional: reset annotation logic
        self.play_video()

    def jump_to_place(self):
        frame_str = self.jump_frame_entry.get().strip()
        time_str = self.jump_time_entry.get().strip()
        if frame_str:
            try:
                jump_frame = int(frame_str)
                if 0 <= jump_frame < self.total_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, jump_frame)
                    self.update_canvas()
                    return
                else:
                    messagebox.showerror("Invalid Frame", f"Frame must be between 0 and {self.total_frames - 1}")
                    return
            except:
                pass

        if time_str:
            try:
                if ":" in time_str:
                    mm, ss = map(int, time_str.split(":"))
                    seconds = mm * 60 + ss
                else:
                    seconds = int(time_str)

                frame = int(seconds * self.fps)
                if 0 <= frame < self.total_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    self.update_canvas()
                else:
                    messagebox.showerror("Invalid Time", f"Time exceeds video length.")
            except:
                messagebox.showerror("Invalid Time", "Enter time as mm:ss or seconds (e.g., 1:40 or 100)")

    def pause_video(self):
        self.is_playing = False

    def step_frame(self, step):
        new_frame = max(0, min(self.current_frame + step, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.is_playing = False
        self.update_canvas()

    def set_frame(self, which):
        frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if which == 'start':
            self.start_frame = frame_num
            messagebox.showinfo("Start Frame", f"Set to {frame_num}")
        else:
            self.end_frame = frame_num
            if self.start_frame is None and self.last_end_frame is not None:
                self.start_frame = self.last_end_frame + 1
                messagebox.showinfo("Auto Start", f"Start frame auto-set to {self.start_frame}")
            messagebox.showinfo("End Frame", f"Set to {frame_num}")
            self.save_annotation()

    def save_annotation(self):
        if self.start_frame is None or self.end_frame is None:
            messagebox.showerror("Error", "Start and End frame must be set.")
            return

        filename = os.path.basename(self.video_files[self.video_index])
        number = filename.split("_", 1)[0]

        annotation = {
            "Video ID": int(number),
            "Filename": filename,
            "Start Frame": self.start_frame,
            "End Frame": self.end_frame,
            "Start Time (s)": round(self.start_frame / self.fps, 2),
            "End Time (s)": round(self.end_frame / self.fps, 2),
            "Violence Type (Video)": self.violence_var.get(),
            "Video Type": self.video_type_var.get(),
            "Violence Type (Sound)": self.sound_type_var.get(),
            "Sound Type": self.sound_type_var.get(),
            "Violence Type (Text)": self.text_type_var.get(),
            "Manual Text": self.note_entry.get(),
            "Memo": self.note_entry_memo.get()
        }

        if os.path.exists(self.SAVE_PATH):
            df_existing = pd.read_excel(self.SAVE_PATH)
            df_new = pd.DataFrame([annotation])
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = pd.DataFrame([annotation])

        df.to_excel(self.SAVE_PATH, index=False)
        # messagebox.showinfo("Annotation saved", )
        start_time = round(self.start_frame / self.fps, 2)
        end_time = round(self.end_frame / self.fps, 2)


        # Format to mm:ss
        mm1, ss1 = divmod(int(start_time), 60)
        mm2, ss2 = divmod(int(end_time), 60)

        self.last_saved_label.config(
            text=f"Last saved: {self.start_frame} - {self.end_frame} ({mm1:02}:{ss1:02} – {mm2:02}:{ss2:02})")

        self.last_end_frame = self.end_frame
        self.start_frame = None
        self.end_frame = None

    def load_next_video(self):
        self.is_playing = False
        time.sleep(0.2)
        
        if self.cap:
            self.cap.release()
        self.video_index += 1
        if self.video_index >= len(self.video_files):
            messagebox.showinfo("Done", "All videos completed.")
            self.root.quit()
            return

        path = self.video_files[self.video_index]
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open: {os.path.basename(path)}")
            self.load_next_video()
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.start_frame = None
        self.end_frame = None
        self.last_end_frame = None
        self.note_entry.delete(0, tk.END)
        self.note_entry_memo.delete(0, tk.END)
        self.jump_frame_entry.delete(0, tk.END)
        self.jump_time_entry.delete(0, tk.END)

        self.video_label.config(text=f"Video {self.video_index + 1}/{len(self.video_files)}: {os.path.basename(path)}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()
