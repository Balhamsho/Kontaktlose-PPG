from scipy.signal import butter, lfilter

def bandpass_filter(signal, fps, low=0.7, high=4):
    nyq = 0.5 * fps
    low /= nyq
    high /= nyq
    b, a = butter(3, [low, high], btype='band')
    return lfilter(b, a, signal)
