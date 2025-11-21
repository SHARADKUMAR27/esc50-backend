import os
import numpy as np
import librosa
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import tempfile


# ---------------------- Attention Layer ----------------------
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


# ---------------------- EXACT Training Preprocessing ----------------------
def extract_mel(file_path):
    """
    EXACT same preprocessing as training (no augmentation).
    - sr = 22050
    - mel n_mels = 128
    - fix_length(..., size=216, axis=1)
    - librosa.util.normalize on log-mel
    Output: (216, 128, 1)
    """
    # 1) Load audio like in training
    y, sr = librosa.load(file_path, sr=22050)

    # Guard against completely empty or invalid audio
    if y is None or y.size == 0:
        raise ValueError("Empty or invalid audio received.")

    # 2) Mel spectrogram (same defaults as training)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    # 3) Convert to log-mel
    log_S = librosa.power_to_db(S, ref=np.max)

    # 4) Force time dimension to 216 frames
    log_S = librosa.util.fix_length(log_S, size=216, axis=1)

    # 5) Normalize log-mel (same as training)
    log_S = librosa.util.normalize(log_S)

    # 6) Return (216, 128, 1)
    return log_S.T[..., np.newaxis]


# ---------------------- Load Model + Classes ----------------------
MODEL_PATH = "esc50_crnn_model.h5"
CLASSES_PATH = "esc50_classes.npy"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Classes file '{CLASSES_PATH}' not found.")

model = load_model(
    MODEL_PATH,
    custom_objects={"Attention": Attention}
)

CLASSES = np.load(CLASSES_PATH, allow_pickle=True)

print("✅ Model and classes loaded.")


# ---------------------- FastAPI App ----------------------
app = FastAPI(title="ESC-50 Audio Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "ESC-50 CRNN audio classifier is running."}


# ---------------------- Prediction Endpoint ----------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "ESC-50 backend running"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Read uploaded bytes
    contents = await file.read()

    # Save to temp file (librosa.load expects a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Preprocess -> (1, 216, 128, 1)
        features = extract_mel(tmp_path)
        features = np.expand_dims(features, axis=0)

        # Predict
        preds = model.predict(features)[0]
        idx = int(np.argmax(preds))
        label = str(CLASSES[idx])
        confidence = float(preds[idx])

        print(f"[DEBUG] Predicted: {label} ({confidence:.4f})")

        return {
            "label": label,
            "confidence": confidence
        }

    except ValueError as e:
        # For empty or bad audio
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Any unexpected issue
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
