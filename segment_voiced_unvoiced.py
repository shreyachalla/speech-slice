import librosa
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt


def plot(energy,title,y_label):
    # Create a new figure
    plt.figure(figsize=(10, 4))
    # Plot the energy
    plt.plot(energy)
    # Set the title and labels
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel(y_label)
    # Display the plot
    plt.show()

def play(voiced_segments, sample_rate):
    # Concatenate all voiced segments into one array for playback
    voiced_audio = np.concatenate(voiced_segments, axis=0)
    # Play the concatenated voiced audio
    sd.play(voiced_audio, samplerate=sample_rate)
    # Wait for the audio to finish playing
    sd.wait()
   

# Sample rate of your original audio file (e.g., 44100 Hz)
sample_rate = 22050  # Adjust this to the sample rate of your audio



def segment_audio(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)
   
    # Compute short-term energy
    energy = np.array([sum(abs(y[i:i+1024])**2) for i in range(0, len(y), 1024)])
    
    # Compute zero-crossing rate
    zcr = np.array([sum(librosa.zero_crossings(y[i:i+1024], pad=False)) for i in range(0, len(y), 1024)])

    # Normalize energy and ZCR for thresholding
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    zcr = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr))
    
    plot(energy,'Short-term Energy of Audio Signal','Energy')
    plot(zcr,'Zero Crossing Rate','ZCR')
    # Define thresholds for voiced/unvoiced decision
    energy_threshold = 0.05
    zcr_threshold = 0.15

    # Segment the audio
    voiced_segments = []
    unvoiced_segments = []
    for i in range(len(energy)):
        start = i * 1024
        end = start + 1024
        if energy[i] > energy_threshold and zcr[i] < zcr_threshold:
            voiced_segments.append(y[start:end])
        else:
            unvoiced_segments.append(y[start:end])

    # Return segmented audio
    return voiced_segments, unvoiced_segments

def vector_embedding(audio_path):
    # Load the audio file
    y, sample_rate = librosa.load(audio_path)

    # Segment the audio file into frames to be passed into 
    time_frame = 20 # 20 msec
    y_duration = len(y) / sample_rate
    num_frames = y_duration * 1000 // time_frame
    len_frame = time_frame * sample_rate
    y_modified = y[0:num_frames * len_frame]

    # Call a function which takes in y_modified and returns a jagged 2D array
    # of dimension Nx(Txlen_frame) where N is number of words identified, T
    # is number of frames in the longest identified word, and len_frame is
    # the number of samples in a frame

    mfcc_matrix = []
    # Assuming the above described jagged 2d array is given by the variable all_words
    for word_idx in range(len(all_words)):
        word = all_words[word_idx]
        n_mfcc = 13
        hann_window = scipy.signals.window.hann(len_frame)
        mfcc = librosa.feature.mfcc(y=y_modified, sr=sample_rate, n_mfcc=n_mfcc, hop_length = len_frame//4, win_length=len_frame, window=hann_window)
        mfcc_matrix.append(mfcc)

    # Now, perform the Dynamic Time Warping (DTW) on each word between itself and each of r reference words
    # Given a word W, perform DTW(W, W_i) -> int for i=1,2,...,r.
    # The end result is a vector of length r for each word W






audio_path = 'test_name.wav'

#segment audio file
voiced_segments, unvoiced_segments = segment_audio(audio_path)

#play separated segments
play(voiced_segments, sample_rate)
play(unvoiced_segments,sample_rate)