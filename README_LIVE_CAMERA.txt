# EmotionAI — Live Camera + Explainable AI + Risk Indicator

This update keeps the existing upload-image workflow and professional Live Camera mode, and adds:

- **Grad-CAM Explainable AI (XAI)** for uploaded-image predictions
- **Expression Risk Indicator** for uploaded images and live camera
- Clear non-clinical wording so the indicator is not presented as a medical diagnosis

## What the new features do

### 1. Explainable AI — Grad-CAM
After an uploaded image is analyzed, EmotionAI generates a Grad-CAM heatmap showing facial regions that contributed most strongly to the selected emotion prediction.

The result page shows:
- Detected face region
- Grad-CAM heatmap overlay
- Emotion probabilities
- Model confidence

The heatmap is an explanation aid, not proof that one facial region is solely responsible for the prediction.

### 2. Expression Risk Indicator
The project adds four visual levels:
- Low
- Moderate
- High
- Uncertain

This is a **project-level, non-clinical heuristic** based on the predicted facial expression and model confidence. It must not be described as a medical, psychological, safety or mental-health diagnosis.

The same indicator is shown in Live Camera mode. Grad-CAM is generated for uploaded-image analysis; it is not generated on every live frame because that would add unnecessary computation.

## Important model path

Your `predict.py` expects:

    model/
    └── best_facial_emotion_model.keras

## Files to copy

Replace the corresponding files in your project:

- `app.py`
- `predict.py`
- `templates/index.html`
- `templates/result.html`
- `static/css/style.css`
- `static/js/script.js`
- `requirements.txt`

Do not delete your trained model.

## Expected project structure

    E:\FER_major_project\
    ├── app.py
    ├── predict.py
    ├── requirements.txt
    ├── model\
    │   └── best_facial_emotion_model.keras
    ├── templates\
    │   ├── index.html
    │   └── result.html
    └── static\
        ├── css\
        │   └── style.css
        ├── js\
        │   └── script.js
        └── uploads\
            └── explainability\

The `uploads` and `explainability` folders are created automatically if they do not exist.

## Run

PowerShell:

    cd E:\FER_major_project
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python app.py

Then open:

    http://127.0.0.1:5000

## Upload-image flow

1. Choose `Upload Image`.
2. Select a clear facial image.
3. Click `Analyze Facial Expression`.
4. The CNN predicts one of the seven emotion classes.
5. The result page shows confidence and all emotion probabilities.
6. The Risk Indicator is displayed.
7. The Explainable AI section displays the Grad-CAM heatmap when a compatible convolutional feature layer is available.

## Live Camera flow

1. Click `Live Camera`.
2. Click `Start Camera`.
3. Allow camera permission in Chrome.
4. Click `Start Live Analysis`.
5. The browser captures frames and sends them to `/predict_camera`.
6. Flask runs the same CNN prediction pipeline.
7. The live panel updates emotion, confidence, probability scores and the Risk Indicator.
8. Click `Stop Camera` to release the webcam.

## Browser requirement

Camera access uses the browser webcam API. Localhost is suitable for local development; deployment on a real domain should use HTTPS.

## Performance note

Live analysis sends approximately one frame every 900 ms. If inference is slow on your PC, increase the interval in `static/js/script.js` from `900` to `1200` or `1500`.
