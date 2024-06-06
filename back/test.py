import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

# Load the audio file
def load_audio(file_path):
    sr, audio = wavfile.read(file_path)
    # Ensure the audio is in mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

# Function to get the dominant frequency
def get_dominant_frequency(audio, sr):
    n_fft = 8192  
    audio = audio / np.max(np.abs(audio))
    fft_spectrum = np.fft.rfft(audio, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, d=1./sr)
    magnitude_spectrum = np.abs(fft_spectrum)
    dominant_freq = freq[np.argmax(magnitude_spectrum)]
    return dominant_freq, freq, magnitude_spectrum

# Function to convert frequency to note name without octave
def freq_to_note_name(freq):
    A4 = 440.0
    C0 = A4 * np.power(2, -4.75)  
    half_steps = np.round(12 * np.log2(freq / C0))
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_index = int(half_steps) % 12
    note_name = note_names[note_index]
    return note_name

# Plot audio waveform
def plot_audio(audio, sr, title='Audio Waveform'):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio)/sr, num=len(audio)), audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Plot frequency spectrum with dynamic zoom
def plot_frequency_spectrum(audio, sr, dominant_freq, zoom_range=200):
    n_fft = 8192
    audio = audio / np.max(np.abs(audio))
    fft_spectrum = np.fft.rfft(audio, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, d=1./sr)
    magnitude_spectrum = np.abs(fft_spectrum)
    
    plt.figure(figsize=(10, 4))
    plt.plot(freq, magnitude_spectrum)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(dominant_freq - zoom_range, dominant_freq + zoom_range)  # Dynamic zooming around the dominant frequency
    plt.show()

# Load the original audio file
file_path = 'Re.wav'
audio, sr = load_audio(file_path)

# Plot original audio
#plot_audio(audio, sr, title='Original Audio')

# Find the dominant frequency in the original audio
dominant_frequency, freq, magnitude_spectrum = get_dominant_frequency(audio, sr)
note_name = freq_to_note_name(dominant_frequency)

# Plot frequency spectrum with dynamic zoom
plot_frequency_spectrum(audio, sr, dominant_frequency)

print(f"The dominant frequency is: {dominant_frequency:.2f} Hz")
print(f"The corresponding note is: {note_name}")
