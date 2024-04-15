import librosa
import numpy as np
# import sounddevice as sd
# import matplotlib.pyplot as plt


def segment_audio(y):
    frame_size = 1024
    
    # Compute short-term energy
    energy = np.array([sum(abs(y[i:i+frame_size])**2) for i in range(0, len(y), frame_size)])
    
    # Compute zero-crossing rate
    zcr = np.array([sum(librosa.zero_crossings(y[i:i+frame_size], pad=False)) for i in range(0, len(y), frame_size)])

    # Normalize energy and ZCR for thresholding
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    zcr = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr))
    
    # plot(energy,'Short-term Energy of Audio Signal','Energy')
    # plot(zcr,'Zero Crossing Rate','ZCR')
    # Define thresholds for voiced/unvoiced decision
    energy_threshold = 0.05
    zcr_threshold = 0.15

    # Segment the audio
    voiced_segments = []
    unvoiced_segments = []
    for i in range(len(energy)):
        start = i * frame_size
        end = start + frame_size
        if energy[i] > energy_threshold and zcr[i] < zcr_threshold:
            voiced_segments.append(y[start:end])
        else:
            unvoiced_segments.append(y[start:end])

    # Return segmented audio
    return voiced_segments, unvoiced_segments, energy, zcr