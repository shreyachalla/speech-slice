import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

'''
TODO: Add method description 
'''
def input_and_plot(file_path, plot=False):

    # take input 
    sample_rate, audio_data = wavfile.read(file_path)
  
    time = np.arange(0, len(audio_data)) / sample_rate

    if plot: 
        # plot the normalized speech signal
        plt.figure(figsize=(10, 4))
        plt.plot(time, audio_data, color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Input Speech Signal')
        plt.grid(True)
        plt.show()

    return audio_data, sample_rate


'''
TODO: description 
'''
def split_window_preemphasize(audio_data, sample_rate, window_size_ms, overlap_ms, pre_emph_coef = 0.9, plot=False):
   
    # calculate samples
    window_size_samples = int(window_size_ms * sample_rate / 1000)
    overlap_size_samples = int(overlap_ms * sample_rate / 1000)
    window_step_samples = window_size_samples - overlap_size_samples

    # split into short windows 
    split_signal = [] 
    start = 0
    while start + window_size_samples <= len(audio_data):
          
        # normalize by using the maximum 
        # rms normalization has the risk of clipping, so we choose max normalization 
        end = start + window_size_samples
        normalized_data = audio_data[start:end] / np.max(np.abs(audio_data[start:end]))
        split_signal.append(librosa.effects.preemphasis(normalized_data, coef=pre_emph_coef))
        start += window_step_samples
    
    # apply Hamming window to each short window 
    hamming_window = np.hamming(window_size_samples)
    hamming_short_windows = [short_window * hamming_window for short_window in split_signal]

    # concatenate all the windows to one signal that has been Hamming windowed
    hamming_windowed_signal = np.concatenate(hamming_short_windows)

    # TODO: plot vertical lines to indicate short window boundaries 
    if plot: 
        # plot the normal and the Hamming windowed signal 
        plt.figure(figsize=(10, 6))
        
        # Hamming windowed signal
        plt.subplot(2, 1, 2)
        plt.plot(hamming_windowed_signal)
        plt.title('Hamming Windowed, Pre-Emphasized,Normalized Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.tight_layout()

    return hamming_short_windows, hamming_windowed_signal


# TODO: write method to plot narrowband and wideband spectrum 