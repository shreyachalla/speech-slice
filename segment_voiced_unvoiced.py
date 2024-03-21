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



audio_path = 'test_name.wav'

#segment audio file
voiced_segments, unvoiced_segments = segment_audio(audio_path)

#play separated segments
play(voiced_segments, sample_rate)
play(unvoiced_segments,sample_rate)