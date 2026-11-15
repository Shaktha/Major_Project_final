const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");

if (imageInput) {
    imageInput.addEventListener("change", function () {
        if (this.files.length > 0) fileName.textContent = "Selected: " + this.files[0].name;
    });
}

const modeButtons = document.querySelectorAll(".mode-button");
const uploadMode = document.getElementById("uploadMode");
const cameraMode = document.getElementById("cameraMode");

modeButtons.forEach((button) => {
    button.addEventListener("click", () => {
        modeButtons.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        const mode = button.dataset.mode;
        if (mode === "camera") {
            uploadMode.classList.remove("active");
            cameraMode.classList.add("active");
        } else {
            cameraMode.classList.remove("active");
            uploadMode.classList.add("active");
            if (typeof stopCamera === "function") stopCamera();
        }
    });
});

const video = document.getElementById("cameraVideo");
const canvas = document.getElementById("cameraCanvas");
const startCameraBtn = document.getElementById("startCameraBtn");
const liveAnalyzeBtn = document.getElementById("liveAnalyzeBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const cameraStatus = document.getElementById("cameraStatus");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const liveBadge = document.getElementById("liveBadge");
const liveEmotion = document.getElementById("liveEmotion");
const liveEmotionIcon = document.getElementById("liveEmotionIcon");
const liveConfidence = document.getElementById("liveConfidence");
const liveConfidenceFill = document.getElementById("liveConfidenceFill");
const liveMessage = document.getElementById("liveMessage");
const liveRiskCard = document.getElementById("liveRiskCard");
const liveRiskIcon = document.getElementById("liveRiskIcon");
const liveRiskLevel = document.getElementById("liveRiskLevel");
const liveRiskDescription = document.getElementById("liveRiskDescription");

let cameraStream = null;
let analysisTimer = null;
let analysisBusy = false;

const emotionIcons = {
    angry: "😠", disgust: "🤢", fear: "😨", happy: "😊",
    neutral: "😐", sad: "😢", surprise: "😲"
};

function setCameraStatus(text, online = false) {
    if (!cameraStatus) return;
    cameraStatus.innerHTML = `<span class="status-dot ${online ? "online" : "offline"}"></span>${text}`;
}

function updateRisk(risk) {
    if (!risk || !liveRiskCard) return;
    liveRiskCard.classList.remove("low", "moderate", "high", "uncertain");
    liveRiskCard.classList.add(risk.class || "uncertain");
    liveRiskIcon.textContent = risk.icon || "○";
    liveRiskLevel.textContent = risk.level || "Uncertain";
    liveRiskDescription.textContent = risk.description || "Waiting for a confident prediction.";
}

function updateLiveResult(data) {
    if (!data.success) {
        liveEmotion.textContent = "No Face";
        liveEmotionIcon.textContent = "—";
        liveConfidence.textContent = "0.00%";
        liveConfidenceFill.style.width = "0%";
        liveMessage.textContent = data.message || "No clear face detected.";
        updateRisk({class: "uncertain", icon: "○", level: "Uncertain", description: "A clear face is required for the indicator."});
        return;
    }

    const emotion = data.emotion;
    const confidence = Number(data.confidence || 0);
    liveEmotion.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    liveEmotionIcon.textContent = emotionIcons[emotion] || "🙂";
    liveConfidence.textContent = confidence.toFixed(2) + "%";
    liveConfidenceFill.style.width = Math.min(confidence, 100) + "%";
    liveMessage.textContent = "Face detected • AI analysis is active.";
    updateRisk(data.risk);
    updateMiniScores(data.scores || {});
}

function updateMiniScores(scores) {
    const container = document.getElementById("liveScores");
    if (!container) return;
    const preferred = ["happy", "neutral", "sad", "surprise"];
    container.innerHTML = preferred.map((emotion) => {
        const value = Number(scores[emotion] || 0);
        return `<div class="score-mini"><span>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span><b>${value.toFixed(1)}%</b></div>`;
    }).join("");
}

async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        liveMessage.textContent = "Camera access is not supported by this browser.";
        return;
    }
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
            audio: false
        });
        video.srcObject = cameraStream;
        await video.play();
        cameraPlaceholder.classList.add("hidden");
        startCameraBtn.disabled = true;
        liveAnalyzeBtn.disabled = false;
        stopCameraBtn.disabled = false;
        setCameraStatus("Camera Ready", true);
        liveMessage.textContent = "Camera ready. Click “Start Live Analysis”.";
    } catch (error) {
        console.error("Camera error:", error);
        setCameraStatus("Camera Off", false);
        if (error.name === "NotAllowedError") liveMessage.textContent = "Camera permission was denied. Please allow camera access.";
        else if (error.name === "NotFoundError") liveMessage.textContent = "No camera was found on this device.";
        else liveMessage.textContent = "Unable to access the camera.";
    }
}

async function analyzeCurrentFrame() {
    if (!cameraStream || analysisBusy || video.readyState < 2) return;
    analysisBusy = true;
    try {
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        context.save();
        context.translate(width, 0);
        context.scale(-1, 1);
        context.drawImage(video, 0, 0, width, height);
        context.restore();
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
        if (!blob) return;
        const formData = new FormData();
        formData.append("frame", blob, "camera_frame.jpg");
        const response = await fetch("/predict_camera", { method: "POST", body: formData });
        const data = await response.json();
        updateLiveResult(data);
    } catch (error) {
        console.error("Live prediction error:", error);
        liveMessage.textContent = "Unable to reach the AI prediction service.";
    } finally {
        analysisBusy = false;
    }
}

function startLiveAnalysis() {
    if (!cameraStream || analysisTimer) return;
    liveAnalyzeBtn.disabled = true;
    liveBadge.classList.add("active");
    setCameraStatus("Live Analysis", true);
    liveMessage.textContent = "Analyzing facial expression...";
    analyzeCurrentFrame();
    analysisTimer = setInterval(analyzeCurrentFrame, 900);
}

function stopCamera() {
    if (analysisTimer) {
        clearInterval(analysisTimer);
        analysisTimer = null;
    }
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }
    if (video) video.srcObject = null;
    if (liveBadge) liveBadge.classList.remove("active");
    if (startCameraBtn) startCameraBtn.disabled = false;
    if (liveAnalyzeBtn) liveAnalyzeBtn.disabled = true;
    if (stopCameraBtn) stopCameraBtn.disabled = true;
    if (cameraPlaceholder) cameraPlaceholder.classList.remove("hidden");
    setCameraStatus("Camera Off", false);
    if (liveEmotion) liveEmotion.textContent = "Waiting";
    if (liveEmotionIcon) liveEmotionIcon.textContent = "—";
    if (liveConfidence) liveConfidence.textContent = "0.00%";
    if (liveConfidenceFill) liveConfidenceFill.style.width = "0%";
    if (liveMessage) liveMessage.textContent = "Start the camera to begin AI analysis.";
    updateRisk({class: "uncertain", icon: "○", level: "Uncertain", description: "Waiting for a confident prediction."});
}

if (startCameraBtn) startCameraBtn.addEventListener("click", startCamera);
if (liveAnalyzeBtn) liveAnalyzeBtn.addEventListener("click", startLiveAnalysis);
if (stopCameraBtn) stopCameraBtn.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);
