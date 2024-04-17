import librosa
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import lfilter
from scipy.integrate import simpson
from scipy.stats import chi2
from scipy.io import wavfile


class Segmentation(object):
    def __init__(self) -> None:
        pass

    def input_and_plot(self, file_path, plot):
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


    def split_window_preemphasize(self, audio_data, sample_rate, window_size_ms, overlap_ms, pre_emph_coef = 0.9, plot=False):
    
        # calculate samples
        window_size_samples = int(window_size_ms * sample_rate / 1000)
        overlap_size_samples = int(overlap_ms * sample_rate / 1000)
        window_step_samples = window_size_samples - overlap_size_samples

        # normalize by using the maximum 
        # rms normalization has the risk of clipping, so we choose max normalization
        normalized_data = audio_data / np.max(np.abs(audio_data))  
        # split into short windows 
        split_signal = [] 
        start = 0
        while start + window_size_samples <= len(audio_data):
            end = start + window_size_samples
            split_signal.append(librosa.effects.preemphasis(normalized_data[start:end], coef=pre_emph_coef))
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
            plt.title('Hamming Windowed, Pre-Emphasized, Normalized Signal')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.tight_layout()

        return hamming_short_windows, hamming_windowed_signal


    def peakiness_segmentation(self, hamming_short_windows, peakiness_threshold=0.6):
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
    
    
    def formant_segmentation(self, hamming_short_windows, sample_rate, formant_diff_threshold=0.2): 

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
        plt.subplot(4, 1, 1)
        plt.title('Formant Based Segmentation')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.plot(np.concatenate(hamming_short_windows))
        formant_segment_boundaries_samples = [] 
        formants_at_boundaries = [] 
        for i in range(1, len(formant_frequencies[0])):
            diff_1 = abs(formant_frequencies[0][i] - formant_frequencies[0][i-1]) / np.max(formant_frequencies[0])
            diff_2 = abs(formant_frequencies[1][i] - formant_frequencies[1][i-1]) / np.max(formant_frequencies[1])
            diff_3 = abs(formant_frequencies[2][i] - formant_frequencies[2][i-1]) / np.max(formant_frequencies[2])
            if diff_1 > formant_diff_threshold and diff_2 > formant_diff_threshold and diff_3 > formant_diff_threshold:
                plt.axvline(x=i * len(hamming_short_windows[0]), color='r', linestyle='--')
                formant_segment_boundaries_samples.append(i * len(hamming_short_windows[0]))
                formants_at_boundaries.append([formant_frequencies[0][i], formant_frequencies[1][i], formant_frequencies[2][i]])

        plt.subplot(4, 1, 2)
        plt.title('Formant Based Segmentation, F1')
        plt.xlabel('Sample')
        plt.ylabel('Frequency (Hz)')
        plt.plot(x, [ i for i in formant_frequencies[0]], color='c')
        for i in formant_segment_boundaries_samples:
            plt.axvline(x=i, color='r', linestyle='--')

        plt.subplot(4, 1, 3)
        plt.title('Formant Based Segmentation, F2')
        plt.xlabel('Sample')
        plt.ylabel('Frequency (Hz)')
        plt.plot(x, [ i for i in formant_frequencies[1]], color='g')
        for i in formant_segment_boundaries_samples:
            plt.axvline(x=i, color='r', linestyle='--')

        plt.subplot(4, 1, 4)
        plt.title('Formant Based Segmentation, F3')
        plt.xlabel('Sample')
        plt.ylabel('Frequency (Hz)')
        plt.plot(x, [ i for i in formant_frequencies[2]], color='m')
        for i in formant_segment_boundaries_samples:
            plt.axvline(x=i, color='r', linestyle='--')

        plt.tight_layout()
        return formant_segment_boundaries_samples, formants_at_boundaries

    def matusita_distance(self, pdf1, pdf2, x):
        integral = simpson(np.sqrt(pdf1 * pdf2), x=x)
        return 1 - integral 

    def matusita_dist_segmentation(self, hamming_short_windows, matusita_threshold=0.5):
        # https://www.csd.uoc.gr/~tziritas/papers/SpeechMusicEusipco.pdf 
        pdfs = [] 
        dfs = [] 
        for window in hamming_short_windows:
            # fit chi-squared distribution 
            df, loc, scale = chi2.fit(window)

            # save the pdf 
            x = np.linspace(chi2.ppf(0.001, df), chi2.ppf(0.999, df), 1000)
            pdf_estimate = chi2.pdf(x, df, loc, scale)
            pdfs.append(pdf_estimate)
            dfs.append(df)
        
        # compute matusita distances between consecutive windows 
        matusita_distances = []
        for i in range(1, len(hamming_short_windows) - 1):
            datavalues = np.linspace(chi2.ppf(0.001, dfs[i]), chi2.ppf(0.999, dfs[i]), 1000)
            matusita_distances.append(self.matusita_distance(pdfs[i - 1], pdfs[i + 1], datavalues))
        
        # padding at the end and beginning 
        matusita_distances.insert(0, 0)
        matusita_distances.append(0)

        norm_matusita_distances = [abs(i / abs(max(matusita_distances, key=abs))) for i in matusita_distances]

        x = [i * len(hamming_short_windows[0]) for i in range(len(hamming_short_windows))]
        plt.figure(figsize=(8, 6))
        plt.plot(x, norm_matusita_distances, label="Matusita Distance")
        plt.plot(np.concatenate(hamming_short_windows), label="Hamming Windowed Signal")
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Magnitude')
        plt.title('Matusita Distance Based Segmentation')
        
        # threshold 
        indices_above_threshold = [] 
        for i in range(len(norm_matusita_distances)):
            if norm_matusita_distances[i] > matusita_threshold:
                indices_above_threshold.append(i) 


        matusita_distance_segment_boundaries_samples = []
        # TODO: find the time instant where two successive frames are located before and after this instant have the maximum distance 
        for i in indices_above_threshold:
            plt.axvline(x=i * len(hamming_short_windows[0]), color='r', linestyle='--')
            matusita_distance_segment_boundaries_samples.append(i * len(hamming_short_windows[0]))
        plt.grid(True)
        plt.show()

        return matusita_distance_segment_boundaries_samples
    


    def teager_energy_segmentation(self, hamming_short_windows, energy_threshold=0.009): 
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
    



    def fitting_proc_segmentation(self, hamming_short_windows, a=1, b=0.01, c=2):  

        # calculate J  
        j_vals = []
        for seg in range(len(hamming_short_windows)):
            # compute  J function 
            j = 0
            for count in range(seg - a, seg):
                j += hamming_short_windows[seg][count]
            for count in range(seg + 1, seg + a + 1):
                j -= hamming_short_windows[seg][count]
            j_vals.append(j / a)

        # detect peaks in J according to threshold b 
        s = []
        for j in j_vals:
            if j >= b:
                s.append(1)
            else:
                s.append(0)

        p = 0
        q = 3
        acc = [0] * len(s) 
        for n in range(a, len(hamming_short_windows) - a - c):
            # compute f[n] for each n in the range (p, q)
            f_min = np.Inf
            min_index = 0  
        
            for v in range(n, n + c + 1):
                f = 0 
                for m in range(n, n + c + 1):
                    for i in range(1, 2):
                        f += s[m * i] * np.abs(v - m)
                if (f < f_min):
                    f_min = f 
                    min_index = v
            nwin = min_index 
        
            acc[nwin] += 1 
            
        new_acc = [1 if i > 2 else 0 for i in acc]
        np.count_nonzero(new_acc)

        # Plot segmented_signal
        plt.figure(figsize=(10, 6))
        plt.plot(np.concatenate(hamming_short_windows))

        # Find indices of nonzero values in new_acc and multiply by 220
        fitting_proc_segment_boundaries_samples = np.nonzero(new_acc)[0] * len(hamming_short_windows[0])

        # Plot vertical lines at the calculated indices
        for index in fitting_proc_segment_boundaries_samples:
            plt.axvline(x=index, color='r', linestyle='--')


        plt.title('Fitting Process Based Segmentation')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
        return fitting_proc_segment_boundaries_samples