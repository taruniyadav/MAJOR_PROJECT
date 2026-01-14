import os
import numpy as np
import librosa
import joblib
import soundfile as sf
import tempfile

# =============================
# Configuration
# =============================
MODEL_PATH = "speech_emotion_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

SAMPLE_RATE = 22050
N_MFCC = 40
CONFIDENCE_THRESHOLD = 0.45

# =============================
# Load trained artifacts
# =============================
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# =============================
# Audio utilities
# =============================
def convert_to_wav(input_audio_path: str) -> str:
    signal, sr = librosa.load(
        input_audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    temp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    sf.write(temp_wav.name, signal, sr)
    return temp_wav.name


def extract_mfcc_features(audio_path: str) -> np.ndarray:
    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std])
    return features.reshape(1, -1)


# =============================
# Prediction
# =============================
def predict_speech_emotion(audio_path: str):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    if not audio_path.lower().endswith(".wav"):
        audio_path = convert_to_wav(audio_path)

    features = extract_mfcc_features(audio_path)

    probabilities = model.predict_proba(features)[0]
    best_index = np.argmax(probabilities)

    confidence = probabilities[best_index]
    emotion = label_encoder.inverse_transform([best_index])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence

    return emotion, confidence


# =============================
# Entry point
# =============================
if __name__ == "__main__":
    audio_file = "archive (1)/Actor_21/03-01-05-02-01-02-21.wav"

    emotion, confidence = predict_speech_emotion(audio_file)

    print(f"Predicted Emotion : {emotion}")
    print(f"Confidence        : {confidence:.2f}")

