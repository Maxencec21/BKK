import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Charger un fichier audio
def load_audio_file(file_path):
    audio, sample_rate = librosa.load(file_path)
    return audio, sample_rate

# Extraire des caractéristiques (MFCC)
def extract_features(audio, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return np.array([mfcc_scaled])

# Charger le modèle de détection de notes
def load_model():
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(40, 1)),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(12, activation='softmax')  # 12 notes de musique
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Prédire les notes de musique
def predict_notes(model, features):
    prediction = model.predict(features)
    notes = np.argmax(prediction, axis=1)
    return notes

# Utilisation des fonctions
if __name__ == '__main__':
    audio_path = 'path_to_your_audio_file.wav'
    audio, sr = load_audio_file(audio_path)
    features = extract_features(audio, sr)
    model = load_model()
    predicted_notes = predict_notes(model, features)
    print("Predicted Notes:", predicted_notes)
