import numpy as np
from utils.filters import bandpass_filter
from scipy.signal import find_peaks

class CHROM:
    name = "chrom"

    def __init__(self):
        self.raw = []
        self.filtered = []
        self.bpm = 0
        self.peaks = []

    def process(self, roi):
        if roi is None:
            return
        r = np.mean(roi[:, :, 2])
        g = np.mean(roi[:, :, 1])
        b = np.mean(roi[:, :, 0])
        x = 3*r - 2*g
        y = 1.5*r + g - 1.5*b
        self.raw.append(x / (y + 1e-6))

    def finalize(self, fps):
        if len(self.raw) < 2:
            self.filtered = np.array([])
            self.bpm = 0
            return self.bpm

        sig = bandpass_filter(np.array(self.raw), fps)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
        self.filtered = sig

        self.peaks, _ = find_peaks(sig, distance=fps*0.4)
        if len(self.peaks) < 2:
            self.bpm = 0
        else:
            self.bpm = 60 / np.mean(np.diff(self.peaks)/fps)
        return self.bpm

    def get_fft(self, fps):
        if len(self.filtered) < 2:
            return np.array([]), np.array([])
        fft = np.abs(np.fft.rfft(self.filtered))
        freqs = np.fft.rfftfreq(len(self.filtered), 1/fps) * 60
        return freqs, fft
