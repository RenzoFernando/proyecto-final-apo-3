const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startButton = document.getElementById("startButton");
const scanButton = document.getElementById("scanButton");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const viewer = document.querySelector(".viewer");
const qualityModelSelect = document.getElementById("qualityModelSelect");
const sizeModelSelect = document.getElementById("sizeModelSelect");
const cameraState = document.getElementById("cameraState");
const scanState = document.getElementById("scanState");
const qualityValue = document.getElementById("qualityValue");
const confidenceValue = document.getElementById("confidenceValue");
const destinationValue = document.getElementById("destinationValue");
const sizeValue = document.getElementById("sizeValue");
const sizeDetail = document.getElementById("sizeDetail");
const meterFill = document.getElementById("meterFill");
const scoreList = document.getElementById("scoreList");
const modelUsed = document.getElementById("modelUsed");

let stream = null;
let scanning = false;
let loop = null;
let uploadedImage = "";

function addDefaultOption(select, text) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = text;
    select.appendChild(option);
}

function fillModelSelect(select, models, defaultText) {
    select.innerHTML = "";
    addDefaultOption(select, defaultText);
    models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.name;
        option.textContent = model.name;
        select.appendChild(option);
    });
}

async function loadModels() {
    const response = await fetch("/api/models");
    const data = await response.json();
    const models = data.models || [];
    const qualityModels = models.filter((model) => model.kind !== "size");
    const sizeModels = models.filter((model) => model.kind === "size");
    fillModelSelect(qualityModelSelect, qualityModels, "Análisis provisional");
    fillModelSelect(sizeModelSelect, sizeModels, "Estimación visual");
}

async function startCamera() {
    uploadedImage = "";
    previewImage.removeAttribute("src");
    viewer.classList.remove("has-image");
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
    if (loop) {
        window.clearInterval(loop);
    }
    loop = window.setInterval(scanCurrentImage, 4200);
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

function currentImageData() {
    if (uploadedImage) {
        return uploadedImage;
    }
    if (!stream || video.readyState < 2) {
        return "";
    }
    return captureFrame();
}

async function scanCurrentImage() {
    if (scanning) {
        return;
    }
    const image = currentImageData();
    if (!image) {
        scanState.textContent = "Sin imagen";
        return;
    }
    scanning = true;
    scanButton.disabled = true;
    scanState.textContent = "Analizando";
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image,
                quality_model: qualityModelSelect.value,
                size_model: sizeModelSelect.value
            })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "No se pudo analizar");
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

function renderScores(scores) {
    scoreList.innerHTML = "";
    (scores || []).forEach((item) => {
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

function renderResult(data) {
    const confidence = Number(data.confidence || 0);
    const sizeConfidence = Number(data.size_confidence || 0);
    qualityValue.textContent = data.quality || "Sin lectura";
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    destinationValue.textContent = data.destination || "Pendiente";
    sizeValue.textContent = data.size || "Pendiente";
    sizeDetail.textContent = `Diámetro relativo: ${Number(data.normalized_diameter || 0).toFixed(2)} | Confianza tamaño: ${sizeConfidence.toFixed(1)}%`;
    meterFill.style.width = `${Math.max(0, Math.min(100, confidence))}%`;
    modelUsed.textContent = `${data.quality_model || "Calidad provisional"} · ${data.size_model || "Tamaño visual"}`;
    renderScores(data.scores || []);
}

startButton.addEventListener("click", async () => {
    try {
        await startCamera();
    } catch (error) {
        cameraState.textContent = "Cámara no disponible";
        scanState.textContent = "Permiso requerido";
    }
});

scanButton.addEventListener("click", scanCurrentImage);

imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
        return;
    }
    const reader = new FileReader();
    reader.onload = () => {
        uploadedImage = String(reader.result || "");
        previewImage.src = uploadedImage;
        viewer.classList.add("has-image");
        cameraState.textContent = "Imagen cargada";
        scanState.textContent = "Lista para analizar";
        scanButton.disabled = false;
        scanCurrentImage();
    };
    reader.readAsDataURL(file);
});

window.addEventListener("beforeunload", () => {
    if (loop) {
        window.clearInterval(loop);
    }
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
    }
});

loadModels();
