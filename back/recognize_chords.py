import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks
import csv

# Load the audio file
def load_audio(file_path):
    sr, audio = wavfile.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

# Function to get the dominant frequencies
def get_dominant_frequencies(audio, sr, n_peaks=5):
    n_fft = 8192
    audio = audio / np.max(np.abs(audio))
    fft_spectrum = np.fft.rfft(audio, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, d=1./sr)
    magnitude_spectrum = np.abs(fft_spectrum)
    
    peaks, _ = find_peaks(magnitude_spectrum, height=0.1*np.max(magnitude_spectrum))
    dominant_freqs = freq[peaks]
    
    dominant_freqs = dominant_freqs[np.argsort(magnitude_spectrum[peaks])[-n_peaks:]]
    dominant_freqs = np.sort(dominant_freqs)
    
    return dominant_freqs, freq, magnitude_spectrum

# Function to convert frequency to note name without octave
def freq_to_note_name(freq):
    A4 = 440.0
    C0 = A4 * np.power(2, -4.75)
    half_steps = np.round(12 * np.log2(freq / C0))
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_index = int(half_steps) % 12
    note_name = note_names[note_index]
    return note_name

# Function to identify chord based on detected notes
def identify_chord(detected_notes, chord_data):
    detected_notes_set = set(detected_notes)
    for chord, notes in chord_data.items():
        if detected_notes_set == set(notes):
            return chord
    return "Unknown Chord"

# Load chord data from CSV
def load_chord_data(csv_file):
    chord_data = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            chord = row[0]
            notes = row[1].split()
            chord_data[chord] = notes
    return chord_data

# Plot audio waveform
def plot_audio(audio, sr, title='Audio Waveform'):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio)/sr, num=len(audio)), audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Plot frequency spectrum with dynamic zoom
def plot_frequency_spectrum(audio, sr, dominant_freqs, zoom_range=200):
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
    for dominant_freq in dominant_freqs:
        plt.axvline(x=dominant_freq, color='r', linestyle='--')
    plt.xlim(dominant_freqs[0] - zoom_range, dominant_freqs[-1] + zoom_range)
    plt.show()

# Load the original audio file
file_path = 'sounds/Cmaj7.wav'
audio, sr = load_audio(file_path)

# Load chord data from CSV
chord_data = load_chord_data('chords_extended.csv')

# Plot original audio
# plot_audio(audio, sr, title='Original Audio')

# Find the dominant frequencies in the original audio
dominant_frequencies, freq, magnitude_spectrum = get_dominant_frequencies(audio, sr, n_peaks=5)
detected_notes = [freq_to_note_name(freq) for freq in dominant_frequencies]

# Identify the chord based on detected notes
identified_chord = identify_chord(detected_notes, chord_data)

# Plot frequency spectrum with dynamic zoom
plot_frequency_spectrum(audio, sr, dominant_frequencies)

print(f"The dominant frequencies are: {dominant_frequencies}")
print(f"The corresponding notes are: {', '.join(detected_notes)}")
print(f"The identified chord is: {identified_chord}")
