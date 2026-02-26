import cv2
import time
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.face_mesh import get_forehead_roi
from methods.green_channel import GreenChannel
from methods.chrom import CHROM
from utils.logger import save_results
from datetime import datetime


class App:
    def __init__(self, root):

        self.root = root
        self.root.title("Contactless PPG System - Green & CHROM")
        self.root.geometry("1200x750")

        self.cap = None
        self.running = False
        self.recording = False

        self.start_time = None
        self.timestamps = []

        self.video_writer = None
        self.video_path = None

        self.fps = 30

        # -------- Methoden--------
        self.green_method = GreenChannel()
        self.chrom_method = CHROM()

        self.signals = {
            "green": [],
            "chrom": []
        }

        self._build_ui()

    # ================= UI =================
    def _build_ui(self):

        left = ttk.Frame(self.root)
        left.pack(side="left", padx=10, pady=10)

        right = ttk.Frame(self.root)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        
        self.video_label = ttk.Label(left)
        self.video_label.pack()

        # Buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=10)

        self.btn_start = ttk.Button(btn_frame, text="▶ Start", command=self.start)
        self.btn_start.grid(row=0, column=0, padx=5)

        self.btn_stop = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=5)

        self.btn_save = ttk.Button(btn_frame, text="💾 Save", command=self.save, state="disabled")
        self.btn_save.grid(row=0, column=2, padx=5)

        self.btn_cancel = ttk.Button(btn_frame, text="❌ Cancel", command=self.cancel)
        self.btn_cancel.grid(row=0, column=3, padx=5)

        
        self.bpm_var = tk.StringVar(value="GREEN BPM: -- | CHROM BPM: --")
        ttk.Label(left, textvariable=self.bpm_var,
                  font=("Arial", 13, "bold")).pack(pady=5)

       
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 8))

        # GREEN
        self.ax1.set_title("GREEN rPPG Signal", color="green")
        self.ax1.set_ylabel("Normalized Amplitude")
        self.ax1.grid(True, alpha=0.3)
        self.line_green, = self.ax1.plot([], [], color="green", lw=2)

        # CHROM
        self.ax2.set_title("CHROM rPPG Signal", color="blue")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("Normalized Amplitude")
        self.ax2.grid(True, alpha=0.3)
        self.line_chrom, = self.ax2.plot([], [], color="blue", lw=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ================= Cancel =================
    def cancel(self):

        self.running = False
        self.recording = False

        if self.cap:
            self.cap.release()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()
        os._exit(0)

    # ================= Start =================
    def start(self):

        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        self.running = True
        self.recording = True
        self.start_time = time.time()
        self.timestamps = []

        # إعادة تهيئة الطريقتين
        self.green_method = GreenChannel()
        self.chrom_method = CHROM()

        self.signals = {"green": [], "chrom": []}

        # ---------- حفظ الفيديو باسم مقروء ----------
        os.makedirs("data/videos", exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.video_path = f"data/videos/session_{ts}.avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_save.config(state="disabled")

        self.update_frame()

    # ================= Stop =================
    def stop(self):

        self.running = False
        self.recording = False

        if self.cap:
            self.cap.release()

        if self.video_writer:
            self.video_writer.release()

        self.green_method.finalize(self.fps)
        self.chrom_method.finalize(self.fps)

        self.bpm_var.set(
            f"GREEN BPM: {self.green_method.bpm:.2f} | "
            f"CHROM BPM: {self.chrom_method.bpm:.2f}"
        )

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_save.config(state="normal")

    # ================= Save =================
    def save(self):

        base = os.path.basename(self.video_path).split(".")[0]
        os.makedirs("data/results", exist_ok=True)

        results = {
            "video": self.video_path,
            "fps": self.fps,
            "methods": {
                "green": {
                    "bpm": self.green_method.bpm,
                    "signal_length": len(self.green_method.filtered)
                },
                "chrom": {
                    "bpm": self.chrom_method.bpm,
                    "signal_length": len(self.chrom_method.filtered)
                }
            }
        }

        save_results(results, self.signals,
                     self.timestamps, self.fps,
                     self.video_path, base)

        self.btn_save.config(state="disabled")

    # ================= Update Frame =================
    def update_frame(self):

        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        roi, frame_draw = get_forehead_roi(frame, draw=True)

        if roi is not None:

            # Verarbeitung beider Methoden
            self.green_method.process(roi)
            self.chrom_method.process(roi)

            t = time.time() - self.start_time
            self.timestamps.append(t)

            self.signals["green"].append(self.green_method.raw[-1])
            self.signals["chrom"].append(self.chrom_method.raw[-1])

            
            self.green_method.finalize(self.fps)
            self.chrom_method.finalize(self.fps)

            self.bpm_var.set(
                f"GREEN BPM: {self.green_method.bpm:.2f} | "
                f"CHROM BPM: {self.chrom_method.bpm:.2f}"
            )

            self.update_plots()

        if self.recording:
            self.video_writer.write(frame_draw)

        
        rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((480, 360))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    # ================= Update Plots =================
    def update_plots(self):

        x = np.array(self.timestamps)

        # GREEN
        y1 = np.array(self.green_method.raw)
        if len(y1) > 2:
            y1 = (y1 - y1.mean()) / (y1.std() + 1e-6)
            self.line_green.set_data(x, y1)
            self.ax1.set_xlim(max(0, x[-1] - 10), x[-1] + 0.1)
            self.ax1.set_ylim(-3, 3)

        # CHROM
        y2 = np.array(self.chrom_method.raw)
        if len(y2) > 2:
            y2 = (y2 - y2.mean()) / (y2.std() + 1e-6)
            self.line_chrom.set_data(x, y2)
            self.ax2.set_xlim(max(0, x[-1] - 10), x[-1] + 0.1)
            self.ax2.set_ylim(-3, 3)

        self.canvas.draw_idle()
