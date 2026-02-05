
import matplotlib.pyplot as plt
import numpy as np
import os

class LivePlot:
    def __init__(self, fps, save_prefix=None):
        self.fps = fps
        self.method = "green"
        self.save_prefix = save_prefix

        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 4))

        # Signal line
        self.line_sig, = self.axes[0].plot([], [], lw=2, color="green")
        self.axes[0].set_title("GREEN PPG Signal", fontsize=14, fontweight="bold")
        self.axes[0].set_xlabel("Time (s)")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].grid(True, alpha=0.3)

        # FFT line
        self.line_fft, = self.axes[1].plot([], [], lw=2, color="green")
        self.axes[1].set_title("GREEN FFT Spectrum", fontsize=14, fontweight="bold")
        self.axes[1].set_xlabel("BPM")
        self.axes[1].set_ylabel("Magnitude")
        self.axes[1].set_xlim(40, 200)
        self.axes[1].grid(True, alpha=0.3)

    def update(self, green_signal):
        if len(green_signal) < 10:
            return

        t = np.arange(len(green_signal)) / self.fps

        # --- Signal plot ---
        self.line_sig.set_data(t, green_signal)
        self.axes[0].set_xlim(0, max(5, t[-1]))
        self.axes[0].set_ylim(np.min(green_signal)-0.1, np.max(green_signal)+0.1)

        # --- FFT plot ---
        fft = np.abs(np.fft.rfft(green_signal))
        freqs = np.fft.rfftfreq(len(green_signal), 1/self.fps) * 60
        self.line_fft.set_data(freqs, fft)
        self.axes[1].set_ylim(0, np.max(fft)+0.1)

        plt.pause(0.001)

    def save_png(self):
        if self.save_prefix:
            os.makedirs("data/results", exist_ok=True)
            self.fig.savefig(f"data/results/{self.save_prefix}_green_plots.png", dpi=200)
