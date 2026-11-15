import os
import uuid
import tempfile

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from predict import predict_emotion

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
EXPLAINABILITY_FOLDER = os.path.join(UPLOAD_FOLDER, "explainability")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLAINABILITY_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_gradcam(heatmap, source_filename):
    if heatmap is None:
        return None

    stem = os.path.splitext(source_filename)[0]
    heatmap_filename = f"{stem}_gradcam.jpg"
    heatmap_path = os.path.join(EXPLAINABILITY_FOLDER, heatmap_filename)

    if not cv2.imwrite(heatmap_path, heatmap):
        return None

    return f"explainability/{heatmap_filename}"


def save_face_crop(face_crop, source_filename):
    if face_crop is None:
        return None

    stem = os.path.splitext(source_filename)[0]
    face_filename = f"{stem}_face.jpg"
    face_path = os.path.join(EXPLAINABILITY_FOLDER, face_filename)

    if not cv2.imwrite(face_path, face_crop):
        return None

    return f"explainability/{face_filename}"


# OpenCV is only needed when saving Grad-CAM images.
import cv2


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="Please select an image.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Please select an image.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Supported formats: JPG, JPEG, PNG and WEBP."
        )

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    filepath = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    try:
        file.save(filepath)
        result = predict_emotion(filepath)

        if not result["success"]:
            return render_template("index.html", error=result["message"])

        heatmap_filename = save_gradcam(result.get("gradcam"), filename)
        face_filename = save_face_crop(result.get("face_crop"), filename)

        return render_template(
            "result.html",
            image_filename=filename,
            emotion=result["emotion"],
            confidence=result["confidence"],
            scores=result["scores"],
            face_box=result["face_box"],
            risk=result["risk"],
            heatmap_filename=heatmap_filename,
            face_filename=face_filename,
            gradcam_available=heatmap_filename is not None,
        )

    except Exception as error:
        import traceback
        traceback.print_exc()
        return render_template("index.html", error=f"Prediction error: {str(error)}")


@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    if "frame" not in request.files:
        return jsonify({"success": False, "message": "No camera frame received."}), 400

    frame = request.files["frame"]
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False, dir=UPLOAD_FOLDER
        ) as temp_file:
            temp_path = temp_file.name
            frame.save(temp_path)

        result = predict_emotion(temp_path, generate_explanation=False)

        if not result["success"]:
            return jsonify(result)

        # Grad-CAM is intentionally not returned for every live frame because
        # it is more computationally expensive than normal inference.
        return jsonify({
            "success": True,
            "emotion": result["emotion"],
            "confidence": round(float(result["confidence"]), 2),
            "scores": {
                key: round(float(value), 2)
                for key, value in result["scores"].items()
            },
            "face_box": result["face_box"],
            "risk": result["risk"],
        })

    except Exception as error:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(error)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
