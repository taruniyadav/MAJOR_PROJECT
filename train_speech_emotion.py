import os
import numpy as np
import librosa
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# =============================
# Configuration
# =============================
DATASET_ROOT = "C:/EmotionAdaptiveLearning/archive (1)"
SAMPLE_RATE = 22050
N_MFCC = 40

MODEL_PATH = "speech_emotion_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

# RAVDESS emotion mapping
EMOTION_MAP = {
    1: "neutral",
    2: "neutral",    # calm → merged into neutral
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "angry",     # disgust → merged into angry
    8: "surprise"
}

# =============================
# Utilities
# =============================
def extract_emotion_from_filename(filename: str) -> str:
    emotion_id = int(filename.split("-")[2])
    return EMOTION_MAP[emotion_id]


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

    # 🔑 Mean + Standard Deviation (CRITICAL FIX)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std])
    return features


def build_dataset(dataset_root: str):
    X, y = [], []

    for actor_folder in os.listdir(dataset_root):
        actor_path = os.path.join(dataset_root, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue

            try:
                audio_path = os.path.join(actor_path, file)
                features = extract_mfcc_features(audio_path)
                emotion = extract_emotion_from_filename(file)

                X.append(features)
                y.append(emotion)

            except Exception as e:
                print(f"Skipping {file}: {e}")

    return np.array(X), np.array(y)


# =============================
# Training
# =============================
def train_model():
    print("Building dataset...")
    X, y = build_dataset(DATASET_ROOT)

    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 🔑 Pipeline = Scaling + SVM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            C=10,
            gamma="scale"
        ))
    ])

    pipeline.fit(X, y_encoded)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print("Training complete.")
    print("Saved model:", MODEL_PATH)
    print("Saved encoder:", ENCODER_PATH)
    print("Emotion classes:", list(label_encoder.classes_))


if __name__ == "__main__":
    train_model()
