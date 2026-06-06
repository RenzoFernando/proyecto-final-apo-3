const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const toggleAnalysisButton = document.getElementById("toggleAnalysisButton");
const scanButton = document.getElementById("scanButton");
const clearButton = document.getElementById("clearButton");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const viewer = document.querySelector(".viewer");
const modelPairSelect = document.getElementById("modelPairSelect");
const productSelect = document.getElementById("productSelect");
const distanceInput = document.getElementById("distanceInput");
const fovInput = document.getElementById("fovInput");
const activeModelSummary = document.getElementById("activeModelSummary");
const cameraState = document.getElementById("cameraState");
const scanState = document.getElementById("scanState");
const qualityValue = document.getElementById("qualityValue");
const confidenceValue = document.getElementById("confidenceValue");
const destinationValue = document.getElementById("destinationValue");
const destinationDetail = document.getElementById("destinationDetail");
const sizeValue = document.getElementById("sizeValue");
const sizeDetail = document.getElementById("sizeDetail");
const sizeCmValue = document.getElementById("sizeCmValue");
const sizeCmDetail = document.getElementById("sizeCmDetail");
const productValue = document.getElementById("productValue");
const productDetail = document.getElementById("productDetail");
const resultTimestamp = document.getElementById("resultTimestamp");
const meterFill = document.getElementById("meterFill");
const scoreList = document.getElementById("scoreList");
const liveBadge = document.getElementById("liveBadge");
const hudProduct = document.getElementById("hudProduct");
const hudDistance = document.getElementById("hudDistance");
const hudFrame = document.getElementById("hudFrame");

const CAPTURE_ROI_RATIO = 0.66;
const MIN_DISTANCE_CM = 10;
const MAX_DISTANCE_CM = 100;
const DEFAULT_DISTANCE_CM = 20;
document.documentElement.style.setProperty("--capture-ratio", `${CAPTURE_ROI_RATIO * 100}%`);

let stream = null;
let scanning = false;
let pendingScan = false;
let loop = null;
let uploadedImage = "";
let modelPairs = [];
let products = [];
let analysisEnabled = false;
let readCount = 0;
let lastScanAt = 0;

function addOption(select, value, text, disabled) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    option.disabled = Boolean(disabled);
    select.appendChild(option);
}

function selectedPair() {
    return modelPairs.find((pair) => pair.key === modelPairSelect.value) || null;
}

function selectedProduct() {
    return products.find((product) => product.key === productSelect.value) || null;
}

function setLiveState(text, active) {
    liveBadge.textContent = text;
    viewer.classList.toggle("live", Boolean(active));
}

function normalizeDistanceValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return DEFAULT_DISTANCE_CM;
    }
    return Math.max(MIN_DISTANCE_CM, Math.min(MAX_DISTANCE_CM, number));
}

function normalizeDistanceInput() {
    const value = normalizeDistanceValue(distanceInput.value);
    distanceInput.value = String(value);
    return value;
}

function refreshHud() {
    const product = selectedProduct();
    const distance = normalizeDistanceValue(distanceInput.value);
    hudProduct.textContent = product ? product.label : "Sin selección";
    hudDistance.textContent = `${distance.toFixed(0)} cm`;
    hudFrame.textContent = `Lecturas: ${readCount}`;
}

function renderActiveModelSummary() {
    const pair = selectedPair();
    if (!pair) {
        activeModelSummary.textContent = "Sin modelo emparejado";
        return;
    }
    activeModelSummary.textContent = `${pair.label} · calidad + tamaño`;
}

function renderProductStatic() {
    const product = selectedProduct();
    productValue.textContent = product ? product.label : "Pendiente";
    productDetail.textContent = product ? `Producto elegido manualmente · ${product.measure}` : "No se detecta automáticamente";
    refreshHud();
}

async function loadCatalog() {
    try {
        const response = await fetch("/api/models");
        const data = await response.json();
        modelPairs = data.pairs || [];
        products = data.products || [];
        modelPairSelect.innerHTML = "";
        productSelect.innerHTML = "";
        if (products.length === 0) {
            addOption(productSelect, "unknown", "Producto no especificado", false);
        } else {
            products.forEach((product) => addOption(productSelect, product.key, product.label, false));
        }
        const defaultProduct = products.find((product) => product.key === "apple") || products[0];
        if (defaultProduct) {
            productSelect.value = defaultProduct.key;
        }
        if (modelPairs.length === 0) {
            addOption(modelPairSelect, "", "Sin modelos en carpeta models", true);
            activeModelSummary.textContent = "Copia los .pkl o .keras en models";
            scanState.textContent = "Sin modelos";
            return;
        }
        modelPairs.forEach((pair) => addOption(modelPairSelect, pair.key, pair.label, !pair.available));
        const firstAvailable = modelPairs.find((pair) => pair.available) || modelPairs[0];
        modelPairSelect.value = firstAvailable.key;
        renderActiveModelSummary();
        renderProductStatic();
        scanState.textContent = "Modelo listo";
    } catch (error) {
        modelPairSelect.innerHTML = "";
        productSelect.innerHTML = "";
        addOption(modelPairSelect, "", "No se pudieron cargar modelos", true);
        addOption(productSelect, "unknown", "Producto no especificado", false);
        activeModelSummary.textContent = "Revisa app/main.py";
        scanState.textContent = "Error de modelos";
    }
}

function startLoop() {
    if (loop) {
        window.clearInterval(loop);
    }
    loop = window.setInterval(() => {
        if (analysisEnabled) {
            scanCurrentImage();
        }
    }, 1400);
}

function stopLoop() {
    if (loop) {
        window.clearInterval(loop);
        loop = null;
    }
}

function setAnalysisEnabled(enabled) {
    analysisEnabled = Boolean(enabled);
    toggleAnalysisButton.textContent = analysisEnabled ? "Pausar análisis" : "Iniciar análisis";
    scanState.textContent = analysisEnabled ? "Analizando en tiempo real" : "Análisis pausado";
    setLiveState(analysisEnabled ? "Tiempo real" : "Pausado", analysisEnabled);
    if (analysisEnabled) {
        startLoop();
        setTimeout(scanCurrentImage, 250);
    }
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
    startButton.disabled = true;
    stopButton.disabled = false;
    toggleAnalysisButton.disabled = false;
    scanButton.disabled = false;
    clearButton.disabled = true;
    setAnalysisEnabled(true);
}

function stopCamera() {
    stopLoop();
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
    video.removeAttribute("srcObject");
    video.srcObject = null;
    analysisEnabled = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    toggleAnalysisButton.disabled = true;
    scanButton.disabled = uploadedImage ? false : true;
    toggleAnalysisButton.textContent = "Iniciar análisis";
    cameraState.textContent = uploadedImage ? "Imagen cargada" : "Cámara inactiva";
    scanState.textContent = uploadedImage ? "Imagen fija lista" : "Sistema listo";
    setLiveState(uploadedImage ? "Imagen fija" : "Sin señal", false);
}

function clearImage() {
    uploadedImage = "";
    imageInput.value = "";
    previewImage.removeAttribute("src");
    viewer.classList.remove("has-image");
    clearButton.disabled = true;
    scanButton.disabled = stream ? false : true;
    cameraState.textContent = stream ? "Cámara activa" : "Cámara inactiva";
    scanState.textContent = stream ? "Cámara lista" : "Sistema listo";
    setLiveState(stream && analysisEnabled ? "Tiempo real" : "Sin señal", stream && analysisEnabled);
}

function captureFrame() {
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const visibleSide = Math.min(width, height);
    const side = Math.max(1, Math.floor(visibleSide * CAPTURE_ROI_RATIO));
    const sourceX = Math.floor((width - side) / 2);
    const sourceY = Math.floor((height - side) / 2);
    canvas.width = side;
    canvas.height = side;
    const context = canvas.getContext("2d");
    context.save();
    context.translate(side, 0);
    context.scale(-1, 1);
    context.drawImage(video, sourceX, sourceY, side, side, 0, 0, side, side);
    context.restore();
    return canvas.toDataURL("image/jpeg", 0.92);
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

function requestPayload(image, pair) {
    return {
        image,
        pair_key: pair.key,
        quality_model: pair.quality_model,
        size_model: pair.size_model,
        product_key: productSelect.value || "unknown",
        distance_cm: normalizeDistanceInput(),
        fov_degrees: Number(fovInput.value || 60),
        roi_ratio: uploadedImage ? 1 : CAPTURE_ROI_RATIO
    };
}

async function scanCurrentImage() {
    if (scanning) {
        pendingScan = true;
        return;
    }
    const now = Date.now();
    if (analysisEnabled && now - lastScanAt < 900) {
        return;
    }
    const image = currentImageData();
    if (!image) {
        scanState.textContent = "Sin imagen";
        return;
    }
    const pair = selectedPair();
    if (!pair || !pair.available) {
        scanState.textContent = "Selecciona un modelo válido";
        return;
    }
    scanning = true;
    pendingScan = false;
    lastScanAt = now;
    scanButton.disabled = true;
    scanState.textContent = "Analizando";
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestPayload(image, pair))
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "No se pudo analizar");
        }
        readCount += 1;
        renderResult(data);
        scanState.textContent = uploadedImage ? "Imagen analizada" : analysisEnabled ? "Lectura en tiempo real" : "Lectura manual";
        refreshHud();
    } catch (error) {
        scanState.textContent = error.message || "Sin lectura";
    } finally {
        scanning = false;
        scanButton.disabled = false;
        if (pendingScan) {
            pendingScan = false;
            setTimeout(scanCurrentImage, 120);
        }
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
    productValue.textContent = data.product_label || "Producto no especificado";
    productDetail.textContent = `Seleccionado manualmente · ${data.product_measure || "medida"}`;
    qualityValue.textContent = data.quality || "Sin lectura";
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    destinationValue.textContent = data.destination || "Pendiente";
    destinationDetail.textContent = "Según calidad predicha";
    sizeValue.textContent = data.size || "Pendiente";
    sizeDetail.textContent = `Confianza tamaño: ${sizeConfidence.toFixed(1)}% · índice visual: ${Number(data.normalized_metric || 0).toFixed(2)}`;
    sizeCmValue.textContent = `${Number(data.size_cm || 0).toFixed(1)} cm`;
    sizeCmDetail.textContent = `${data.product_measure || "Medida"}: ${Number(data.size_range_min_cm || 0).toFixed(1)}-${Number(data.size_range_max_cm || 0).toFixed(1)} cm · óptico ${Number(data.optical_cm || 0).toFixed(1)} cm · ${data.measure_method || "estimación"}`;
    meterFill.style.width = `${Math.max(0, Math.min(100, confidence))}%`;
    resultTimestamp.textContent = "Lectura actualizada";
    renderScores(data.scores || []);
}

startButton.addEventListener("click", async () => {
    try {
        await startCamera();
    } catch (error) {
        cameraState.textContent = "Cámara no disponible";
        scanState.textContent = "Permiso requerido";
        setLiveState("Sin señal", false);
    }
});

stopButton.addEventListener("click", stopCamera);

scanButton.addEventListener("click", scanCurrentImage);

clearButton.addEventListener("click", clearImage);

toggleAnalysisButton.addEventListener("click", () => {
    setAnalysisEnabled(!analysisEnabled);
});

modelPairSelect.addEventListener("change", () => {
    renderActiveModelSummary();
    if (uploadedImage || stream) {
        scanCurrentImage();
    }
});

productSelect.addEventListener("change", () => {
    renderProductStatic();
    if (uploadedImage || stream) {
        scanCurrentImage();
    }
});

[distanceInput, fovInput].forEach((input) => {
    input.addEventListener("input", () => {
        refreshHud();
    });
    input.addEventListener("change", () => {
        if (input === distanceInput) {
            normalizeDistanceInput();
        }
        refreshHud();
        if (uploadedImage || stream) {
            scanCurrentImage();
        }
    });
});

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
        scanState.textContent = "Imagen fija lista";
        scanButton.disabled = false;
        clearButton.disabled = false;
        setAnalysisEnabled(false);
        setLiveState("Imagen fija", false);
        scanCurrentImage();
    };
    reader.readAsDataURL(file);
});

window.addEventListener("beforeunload", () => {
    stopLoop();
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
    }
});

loadCatalog();
refreshHud();