import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
from dtw import *


class Vector_Embeddings(object):
    def __init__(self):
        pass

    def embed_vector(self, signal, sample_rate):

        # Some hyperparameters
        time_frame = 500 # in msec
        len_frame = time_frame * sample_rate // 1000
        len_signal = len(signal)
        num_frames = len_signal // len_frame
        basis_percentage = 0.75
        max_dtw_dimensionality = 100 # Maximum number of dimensions the vectors should span after DTW
        dtw_dimensionality = min(max_dtw_dimensionality, (int)(basis_percentage*num_frames))


        # Now, the actual MFCC algorithm
        n_mfcc = 7
        hann_window = scipy.signal.windows.hann(len_frame)
        mfcc_matrix = np.transpose(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, hop_length = (int)(len_frame*0.85), win_length=len_frame, window=hann_window, n_fft=len_frame))
        mfcc_matrix = mfcc_matrix[:, 1:] # Removing very negative first MFCC coefficient
        # mfcc_matrix = mfcc_matrix[:, 0:]

        # Now, perform the Dynamic Time Warping (DTW) on each word between itself and each of r reference words
        # Given a word W, perform DTW(W, W_i) -> int for i=1,2,...,r.
        # The end result is a vector of length r for each word W. Wxr
        
        # Obtaining the r reference words
        rng = np.random.default_rng()
        basis_words_idxs = rng.choice(len(mfcc_matrix), dtw_dimensionality, replace=False)

        dtw_matrix = []
        one_time = 2
        # Iterating over every word-like segment
        for word_idx in range(len(mfcc_matrix)):
            dtw_vector = []
            # Iterating over every reference word
            for index in basis_words_idxs:
                alignment = dtw(mfcc_matrix[word_idx], mfcc_matrix[index], distance_only=True) # Set distance_only=True if only the dist is needed (no plotting later on)
                dtw_vector.append(alignment.normalizedDistance) # Later, try alignment.normalizedDistance, but I think that will be worse
                # Generate a few plots for reference and visualization
                """
                if word_idx == 74 and one_time == 2:
                    ax1 = dtwPlotAlignment(d=alignment, xlab="Query Index", ylab="Reference Index")
                    ax2 = dtwPlotTwoWay(d=alignment, xts=mfcc_matrix[word_idx], yts=mfcc_matrix[index], xlab="Query Index", ylab="Reference Index")
                    ax3 = dtwPlotThreeWay(d=alignment, xts=mfcc_matrix[word_idx], yts=mfcc_matrix[index], xlab="Query Index", ylab="Reference Index", match_indices=5, main="Three Way")
                    one_time = 3
                """
            dtw_matrix.append(dtw_vector)
        return np.array(dtw_matrix)