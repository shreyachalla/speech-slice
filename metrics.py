import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from scipy.signal import find_peaks
from scipy.signal import lfilter

'''
TODO: description 
'''
# derive relevant metrics per window to use as an indicator upon which to segment 
def peakiness_segmentation(hamming_short_windows, peakiness_threshold=0.6):
    rms_peak_heights = []
    max_peak_heights = []

    for window in hamming_short_windows:
        # calculate autocrrelation 
        autocorr = np.correlate(window, window, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # find peaks 
        peaks, _ = find_peaks(autocorr)

        if len(peaks) > 0:
            peak_heights = autocorr[peaks]

            rms_peak_height = np.sqrt(np.mean(np.square(peak_heights)))
            max_peak_height = np.max(peak_heights)

            rms_peak_heights.append(rms_peak_height)
            max_peak_heights.append(max_peak_height)

    # calculate quality of peak 
    peakiness = [max_peak / rms_peak for max_peak, rms_peak in zip(max_peak_heights, rms_peak_heights)]
    normalized_peakiness = [value / max(peakiness) for value in peakiness]

    
    plt.figure(figsize=(10, 8))

    x = [i * len(hamming_short_windows[0]) for i in range(len(hamming_short_windows))]
    plt.subplot(2, 1, 1)
    plt.plot(x, normalized_peakiness, label='Normalized Peakiness')
    plt.plot(np.concatenate(hamming_short_windows), label='Normalized Hamming Windowed Signal')
    plt.title('Normalized Peakiness Based Segmentation')
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.legend()

    # observe changes in peakiness values, apply threshold, and draw boundaries there 
    peakiness_segment_boundaries_samples = [] 
    peakiness_indices, _ = find_peaks(normalized_peakiness)
    for x in peakiness_indices: 
        if normalized_peakiness[x] > peakiness_threshold:
            plt.axvline(x=x * len(hamming_short_windows[0]), color='r', linestyle='--') 
            peakiness_segment_boundaries_samples.append(x * len(hamming_short_windows[0]))

    # return the samples where boundaries were identified 
    return peakiness_segment_boundaries_samples

'''
TODO: description
'''
from scipy.signal import lfilter

def formant_segmentation(hamming_short_windows, sample_rate, formant_diff_threshold=0.2): 

    # use LPC to calculate formants 
    order = 8 

    formant_frequencies = [[], [], []] 

    for window in hamming_short_windows:
        lpc_coefficients = librosa.lpc(window, order=order)
        lpc_roots = np.roots(lpc_coefficients)
    
        # keep only one per conjugate pair 
        positive_imag_roots = lpc_roots[lpc_roots.imag > 0]
        # find the angle and convert rad/s to Hz 
        formants = (np.angle(positive_imag_roots) * sample_rate / (2 * np.pi)).tolist()[:3]
        bandwidths = [(-0.5 * sample_rate / (2 * np.pi) * np.log(np.abs(root))) for root in lpc_roots]
       
        # formants are at least 90Hz, and bandwidth is less than 400Hz
        for i in range(len(formants)): 
            if formants[i] >= 90 and bandwidths[i] < 400:
                formant_frequencies[i].append(formants[i])
            else:
                formant_frequencies[i].append(0)
    
    if (len(formant_frequencies[2]) < len(hamming_short_windows)):
        formant_frequencies[2] = np.pad(formant_frequencies[2], (0, len(hamming_short_windows) - len(formant_frequencies[2])), mode='constant', constant_values=0)

    x = [i * len(hamming_short_windows[0]) for i in range(len(hamming_short_windows))]
    
    plt.figure(figsize=(10, 8))
    plt.title('Formant Based Segmentation')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.plot(np.concatenate(hamming_short_windows))
    plt.plot(x, [ i / 8000 for i in formant_frequencies[0]])
    plt.plot(x, [ i / 8000 for i in formant_frequencies[1]])
    plt.plot(x, [ i / 8000 for i in formant_frequencies[2]])
    formant_segment_boundaries_samples = [] 
    formants_at_boundaries = [] 
    for i in range(1, len(formant_frequencies[0])):
        diff_1 = abs(formant_frequencies[0][i] - formant_frequencies[0][i-1]) / 8000
        diff_2 = abs(formant_frequencies[1][i] - formant_frequencies[1][i-1]) / 8000
        diff_3 = abs(formant_frequencies[2][i] - formant_frequencies[2][i-1]) / 8000
        if diff_1 > formant_diff_threshold and diff_2 > formant_diff_threshold and diff_3 > formant_diff_threshold:
            plt.axvline(x=i * len(hamming_short_windows[0]), color='r', linestyle='--')
            formant_segment_boundaries_samples.append(i * len(hamming_short_windows[0]))
            formants_at_boundaries.append([formant_frequencies[0][i], formant_frequencies[1][i], formant_frequencies[2][i]])
    return formant_segment_boundaries_samples, formants_at_boundaries

'''
TODO 
'''
def formants_to_vowels(formants): 
    '''
    i: feet
    I: big
    E: bird
    ae: ash 
    a: hard
    o: lord
    U: book 
    u: food 
    '''
    vowels_to_formants = 
        {'i': [280, 2250, 2890], 
        'I': [400, 1920, 2560], 
        'E': [550, 1770, 2490], 
        'ae': [690, 1660, 2490], 
        'a': [710, 1100, 2540], 
        'o': [590, 880, 2540], 
        'U': [450, 1030, 2380], 
        'u': [310, 870, 2250]}

'''
TODO 
'''

def teager_energy_segmentation(hamming_short_windows, energy_threshold=0.009): 
    teager_energy = []

    # compute Teager energy 
    for window in hamming_short_windows:
        energy = 0
        for i in range(0, len(window)):
            if (i == 0):
                energy += window[i] ** 2 - window[i + 1]
            elif (i == len(window) - 1):
                energy += window[i] **  2 - window[i - 1]
            else: 
                energy += window[i]**2 - (window[i + 1] * window[i - 1])
        teager_energy.append(energy)
    
    teager_energy = [value / max(teager_energy) for value in teager_energy]
    energy_segment_boundaries_samples = [] 
    plt.figure(figsize=(10, 8))
    plt.title('Teager Energy Based Segmentation')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.plot(np.concatenate(hamming_short_windows))
  

    for i in range(1, len(teager_energy)):
        diff = abs(teager_energy[i] - teager_energy[i-1])
        if diff > energy_threshold:
            plt.axvline(x=i * len(hamming_short_windows[0]), color='r', linestyle='--') 
            energy_segment_boundaries_samples.append(i * len(hamming_short_windows[0]))
    return energy_segment_boundaries_samples
