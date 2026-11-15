import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_facial_emotion_model.keras")

EMOTIONS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def crop_face(image, face_box):
    x, y, w, h = face_box
    image_height, image_width = image.shape[:2]
    padding_x = int(w * 0.20)
    padding_y = int(h * 0.25)
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image_width, x + w + padding_x)
    y2 = min(image_height, y + h + padding_y)
    return image[y1:y2, x1:x2]


def preprocess_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    image_array = gray.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    return np.expand_dims(image_array, axis=0)


def calculate_risk_indicator(emotion, confidence):
    """Project-level, non-clinical expression alert heuristic."""
    confidence = float(confidence)

    if confidence < 50:
        return {
            "level": "Uncertain",
            "class": "uncertain",
            "icon": "○",
            "description": "Model confidence is low, so the expression should not be strongly interpreted."
        }

    if emotion in {"fear", "angry"}:
        return {
            "level": "High",
            "class": "high",
            "icon": "!",
            "description": "A strong fear/anger expression was detected by the model."
        }

    if emotion in {"sad", "disgust"}:
        return {
            "level": "Moderate",
            "class": "moderate",
            "icon": "•",
            "description": "A strong negative facial-expression pattern was detected."
        }

    return {
        "level": "Low",
        "class": "low",
        "icon": "✓",
        "description": "No elevated expression-alert pattern was detected by this project heuristic."
    }


def _find_gradcam_layer():
    """Find the last convolution-like 4-D feature layer."""
    fallback = None

    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
            rank = len(shape)
        except Exception:
            continue

        if rank != 4:
            continue

        fallback = fallback or layer
        name = f"{layer.name} {layer.__class__.__name__}".lower()
        if "conv" in name:
            return layer

    return fallback


GRADCAM_LAYER = _find_gradcam_layer()
GRADCAM_MODEL = None

if GRADCAM_LAYER is not None:
    try:
        GRADCAM_MODEL = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[GRADCAM_LAYER.output, model.output]
        )
    except Exception as error:
        print(f"Grad-CAM initialization warning: {error}")


def generate_gradcam(face, predicted_index):
    """Return a heatmap overlay for the detected face, or None if unsupported."""
    if GRADCAM_MODEL is None:
        return None

    processed = preprocess_face(face)
    tensor = tf.convert_to_tensor(processed, dtype=tf.float32)

    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = GRADCAM_MODEL(tensor, training=False)
            class_score = predictions[:, predicted_index]

        gradients = tape.gradient(class_score, conv_outputs)
        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        if float(max_value.numpy()) <= 1e-8:
            return None

        heatmap = heatmap / max_value
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(
            heatmap,
            (face.shape[1], face.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        heatmap_uint8 = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face, 0.58, color_heatmap, 0.42, 0)
        return overlay

    except Exception as error:
        print(f"Grad-CAM generation warning: {error}")
        return None


def predict_emotion(image_path, generate_explanation=True):
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise ValueError(f"Image file does not exist: {image_path}")

    try:
        pil_image = Image.open(image_path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        rgb_image = np.array(pil_image)
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    except Exception as error:
        raise ValueError(f"Could not open uploaded image: {error}")

    if image is None or image.size == 0:
        raise ValueError("Uploaded image is empty or could not be decoded.")

    face_box = detect_face(image)

    if face_box is None:
        return {
            "success": False,
            "message": "No clear face was detected. Please upload a clear front-facing facial image."
        }

    face = crop_face(image, face_box)
    if face is None or face.size == 0:
        raise ValueError("Detected face region is empty.")

    processed_image = preprocess_face(face)
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_emotion = EMOTIONS[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    scores = {
        emotion: float(predictions[index]) * 100
        for index, emotion in enumerate(EMOTIONS)
    }

    risk = calculate_risk_indicator(predicted_emotion, confidence)
    heatmap = generate_gradcam(face, predicted_index) if generate_explanation else None

    return {
        "success": True,
        "emotion": predicted_emotion,
        "confidence": confidence,
        "scores": scores,
        "face_box": {
            "x": int(face_box[0]),
            "y": int(face_box[1]),
            "width": int(face_box[2]),
            "height": int(face_box[3])
        },
        "risk": risk,
        "gradcam": heatmap,
        "face_crop": face
    }
