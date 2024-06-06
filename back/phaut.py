import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import soundfile as sf
#from google.colab import files
import librosa.display

# Create an audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# Create a high-pass filter
def butter_highpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Apply the filter
def apply_highpass_filter(audio, sr, cutoff_frequency=100):
    b, a = butter_highpass(cutoff_frequency, sr, order=5)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

# Spectral subtraction
def noise_reduction_spectral_subtraction(audio, sr, n_fft=2048, hop_length=512):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    noise_mag = np.median(magnitude, axis=1, keepdims=True)
    reduced_magnitude = np.maximum(magnitude - noise_mag, 0)
    reduced_stft = reduced_magnitude * phase
    clean_audio = librosa.istft(reduced_stft, hop_length=hop_length)
    return clean_audio

# Audio plot
def plot_audio(audio, sr, title='Audio Waveform'):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Audio saving
def save_audio(file_path, audio, sr):
    sf.write(file_path, audio, sr)

# Defining the path
input_path = 'C.wav' 
output_path = input_path  
audio, sr = load_audio(input_path)

# Plot original audio
plot_audio(audio, sr, title='Original Audio')

# Plot firstly filtered audio
filtered_audio = apply_highpass_filter(audio, sr)
plot_audio(filtered_audio, sr, title='Filtered Audio with Highpass Filter')

# Plot secondly filtered audio
clean_audio = noise_reduction_spectral_subtraction(filtered_audio, sr)
plot_audio(clean_audio, sr, title='Audio After Combined Noise Reduction')

# Clean audio saving
save_audio(output_path, clean_audio, sr)
print("Audio processed and saved as:", output_path)

# uploading the audio
#files.download(output_path)

