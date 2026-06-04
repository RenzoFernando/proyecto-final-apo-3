const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startButton = document.getElementById("startButton");
const scanButton = document.getElementById("scanButton");
const modelSelect = document.getElementById("modelSelect");
const cameraState = document.getElementById("cameraState");
const scanState = document.getElementById("scanState");
const qualityValue = document.getElementById("qualityValue");
const confidenceValue = document.getElementById("confidenceValue");
const destinationValue = document.getElementById("destinationValue");
const meterFill = document.getElementById("meterFill");
const scoreList = document.getElementById("scoreList");
const modelUsed = document.getElementById("modelUsed");

let stream = null;
let scanning = false;
let loop = null;

async function loadModels() {
    const response = await fetch("/api/models");
    const data = await response.json();
    modelSelect.innerHTML = "";
    const models = data.models || [];
    if (models.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "Análisis provisional";
        modelSelect.appendChild(option);
        return;
    }
    models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.name;
        option.textContent = model.name;
        modelSelect.appendChild(option);
    });
}

async function startCamera() {
    stream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: "environment",
            width: { ideal: 1280 },
            height: { ideal: 720 }
        },
        audio: false
    });
    video.srcObject = stream;
    cameraState.textContent = "Cámara activa";
    scanState.textContent = "En espera";
    scanButton.disabled = false;
    startButton.textContent = "Cámara activa";
    startButton.disabled = true;
    loop = window.setInterval(scanFrame, 3500);
}

function captureFrame() {
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.translate(width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, width, height);
    return canvas.toDataURL("image/jpeg", 0.9);
}

async function scanFrame() {
    if (!stream || scanning || video.readyState < 2) {
        return;
    }
    scanning = true;
    scanButton.disabled = true;
    scanState.textContent = "Escaneando";
    try {
        const image = captureFrame();
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image, model: modelSelect.value })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "No se pudo escanear");
        }
        renderResult(data);
        scanState.textContent = "Lectura completa";
    } catch (error) {
        scanState.textContent = "Sin lectura";
    } finally {
        scanning = false;
        scanButton.disabled = false;
    }
}

function renderResult(data) {
    const confidence = Number(data.confidence || 0);
    qualityValue.textContent = data.quality || "Sin lectura";
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    destinationValue.textContent = data.destination || "Pendiente";
    meterFill.style.width = `${Math.max(0, Math.min(100, confidence))}%`;
    modelUsed.textContent = data.model || "Modelo pendiente";
    scoreList.innerHTML = "";
    (data.scores || []).forEach((item) => {
        const row = document.createElement("div");
        row.className = "score-row";
        const name = document.createElement("span");
        name.textContent = item.label;
        const bar = document.createElement("div");
        bar.className = "score-bar";
        const fill = document.createElement("span");
        fill.style.width = `${Math.max(0, Math.min(100, Number(item.value || 0)))}%`;
        const value = document.createElement("strong");
        value.textContent = `${Number(item.value || 0).toFixed(1)}%`;
        bar.appendChild(fill);
        row.appendChild(name);
        row.appendChild(bar);
        row.appendChild(value);
        scoreList.appendChild(row);
    });
}

startButton.addEventListener("click", async () => {
    try {
        await startCamera();
    } catch (error) {
        cameraState.textContent = "Cámara no disponible";
        scanState.textContent = "Permiso requerido";
    }
});

scanButton.addEventListener("click", scanFrame);

window.addEventListener("beforeunload", () => {
    if (loop) {
        window.clearInterval(loop);
    }
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
    }
});

loadModels();
