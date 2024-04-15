from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from scipy.integrate import simps 

'''
'''
def matusita_distance(pdf1, pdf2, x):
    integral = simpson(np.sqrt(pdf1 * pdf2), x=x)
    return 1 - integral 

'''
''' 
def matusita_dist_segmentation(hamming_short_windows, matusita_threshold=0.5):
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
        matusita_distances.append(matusita_distance(pdfs[i - 1], pdfs[i + 1], datavalues))
    
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



'''
TODO 
'''
def fitting_proc_segmentation(hamming_short_windows, a=1, b=0.01, c=2):  

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
