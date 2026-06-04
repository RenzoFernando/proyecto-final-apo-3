from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import base64
import io
import json
import mimetypes
import pickle
import sys
import time

import numpy as np
from PIL import Image, ImageOps

try:
    import joblib
except Exception:
    joblib = None

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
SIZE_THRESHOLDS_PATH = ANNOTATIONS_DIR / "size_thresholds.json"
MODEL_EXTENSIONS = {".pkl", ".joblib", ".h5", ".keras"}
MODEL_CACHE = {}

QUALITY_NAMES = {
    "good": "Buena",
    "bueno": "Buena",
    "buena": "Buena",
    "healthy": "Buena",
    "fresh": "Buena",
    "regular": "Regular",
    "normal": "Regular",
    "mala": "Mala",
    "malo": "Mala",
    "bad": "Mala",
    "poor": "Mala",
    "rotten": "Mala",
    "damaged": "Mala"
}

SIZE_NAMES = {
    "small": "Pequeño",
    "pequeno": "Pequeño",
    "pequeña": "Pequeño",
    "pequena": "Pequeño",
    "chico": "Pequeño",
    "chica": "Pequeño",
    "medium": "Mediano",
    "mediano": "Mediano",
    "mediana": "Mediano",
    "regular": "Mediano",
    "large": "Grande",
    "grande": "Grande"
}

DESTINATIONS = {
    "Buena": "Consumo o venta directa",
    "Regular": "Revisión o consumo rápido",
    "Mala": "Descarte o aprovechamiento no comercial"
}


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw)


def config_candidates(model_path):
    return [
        model_path.with_name(model_path.stem + "_config.json"),
        model_path.with_suffix(".json"),
        MODELS_DIR / "model_config.json",
        MODELS_DIR / "config.json"
    ]


def read_config(model_path):
    for path in config_candidates(model_path):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def model_kind_from(path, config):
    text = " ".join([path.stem.lower(), str(config.get("target", "")).lower(), str(config.get("task", "")).lower(), str(config.get("output", "")).lower()])
    if "size" in text or "tamano" in text or "tamaño" in text:
        return "size"
    return "quality"


def list_model_files():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(MODELS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
            config = read_config(path)
            items.append({
                "name": path.name,
                "path": str(path.relative_to(PROJECT_ROOT)).replace(chr(92), "/"),
                "kind": model_kind_from(path, config)
            })
    return items


def safe_model_path(name):
    if not name:
        return None
    path = (MODELS_DIR / name).resolve()
    try:
        path.relative_to(MODELS_DIR.resolve())
    except ValueError:
        return None
    if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
        return path
    return None


def load_model(model_path):
    key = str(model_path.resolve())
    cached = MODEL_CACHE.get(key)
    if cached:
        return cached
    suffix = model_path.suffix.lower()
    config = read_config(model_path)
    if suffix in {".h5", ".keras"}:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
        model_type = "keras"
    elif joblib is not None:
        try:
            model = joblib.load(model_path)
        except Exception:
            with model_path.open("rb") as file:
                model = pickle.load(file)
        model_type = "sklearn"
    else:
        with model_path.open("rb") as file:
            model = pickle.load(file)
        model_type = "sklearn"
    MODEL_CACHE[key] = {"model": model, "config": config, "type": model_type, "kind": model_kind_from(model_path, config)}
    return MODEL_CACHE[key]


def decode_image(data_url):
    if not data_url:
        raise ValueError("Imagen no recibida")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    data = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def normalize_quality_label(value):
    text = str(value).strip()
    key = text.lower().replace("_", " ").replace("-", " ")
    for alias, name in QUALITY_NAMES.items():
        if alias in key.split() or alias in key:
            return name
    return text[:1].upper() + text[1:] if text else "Sin clase"


def normalize_size_label(value):
    text = str(value).strip()
    key = text.lower().replace("_", " ").replace("-", " ")
    for alias, name in SIZE_NAMES.items():
        if alias in key.split() or alias in key:
            return name
    return text[:1].upper() + text[1:] if text else "Sin tamaño"


def destination_for(label):
    return DESTINATIONS.get(label, "Clasificación")


def labels_from(model, config, task):
    if task == "size":
        labels = config.get("size_labels") or config.get("classes") or config.get("labels")
    else:
        labels = config.get("quality_labels") or config.get("classes") or config.get("labels")
    if labels:
        return list(labels)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return [str(item) for item in list(classes)]
    return ["small", "medium", "large"] if task == "size" else ["bad", "regular", "good"]


def image_size_from(config):
    size = config.get("img_size") or config.get("image_size") or config.get("target_image_size") or 128
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return (int(size[0]), int(size[1]))
    return (128, 128)


def preprocess_image(image, config, model_type):
    size = image_size_from(config)
    color_mode = str(config.get("color_mode", "RGB")).upper()
    scale = float(config.get("scale", 255.0))
    padded = ImageOps.pad(image, size, color=(255, 255, 255), centering=(0.5, 0.5))
    if color_mode in {"L", "GRAY", "GRAYSCALE"}:
        prepared = padded.convert("L")
    else:
        prepared = padded.convert("RGB")
    arr = np.asarray(prepared).astype("float32")
    if scale > 0:
        arr = arr / scale
    if model_type == "keras":
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        return arr.reshape((1,) + arr.shape)
    flatten = bool(config.get("flatten", True))
    if flatten:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr.reshape((1,) + arr.shape)


def scores_from_probabilities(probabilities, labels, normalizer):
    values = np.asarray(probabilities).reshape(-1).astype(float)
    if values.size == 0:
        return []
    if values.max() > 1 or values.min() < 0:
        exps = np.exp(values - np.max(values))
        values = exps / np.sum(exps)
    total = float(values.sum())
    if total > 0:
        values = values / total
    scores = []
    for index, value in enumerate(values):
        label = labels[index] if index < len(labels) else str(index)
        scores.append({"label": normalizer(label), "value": round(float(value) * 100, 1)})
    return sorted(scores, key=lambda item: item["value"], reverse=True)


def predict_model_task(image, model_name, task):
    model_path = safe_model_path(model_name)
    if model_path is None:
        return None
    package = load_model(model_path)
    model = package["model"]
    config = package["config"]
    model_type = package["type"]
    labels = labels_from(model, config, task)
    normalizer = normalize_size_label if task == "size" else normalize_quality_label
    x = preprocess_image(image, config, model_type)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]
        scores = scores_from_probabilities(probabilities, labels, normalizer)
        label = scores[0]["label"] if scores else "Sin clase"
        confidence = scores[0]["value"] if scores else 0.0
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x))
        values = decision.reshape(-1).astype(float)
        if values.size == 1 and len(labels) == 2:
            positive = 1.0 / (1.0 + np.exp(-values[0]))
            probabilities = np.array([1.0 - positive, positive])
        else:
            probabilities = values
        scores = scores_from_probabilities(probabilities, labels, normalizer)
        label = scores[0]["label"] if scores else "Sin clase"
        confidence = scores[0]["value"] if scores else 0.0
    else:
        prediction = model.predict(x)
        if isinstance(prediction, list):
            prediction = np.asarray(prediction)
        arr = np.asarray(prediction)
        if arr.ndim >= 2 and arr.shape[-1] > 1:
            scores = scores_from_probabilities(arr[0], labels, normalizer)
            label = scores[0]["label"] if scores else "Sin clase"
            confidence = scores[0]["value"] if scores else 0.0
        else:
            raw = arr.reshape(-1)[0]
            if isinstance(raw, (np.integer, int)) and int(raw) < len(labels):
                raw = labels[int(raw)]
            label = normalizer(raw)
            confidence = 100.0
            scores = [{"label": label, "value": confidence}]
    return {"label": label, "confidence": round(float(confidence), 1), "scores": scores[:3], "model": model_path.name}


def estimate_size_features(image, image_size=160):
    working_image = ImageOps.pad(image, (image_size, image_size), color=(255, 255, 255), centering=(0.5, 0.5))
    array = np.asarray(working_image).astype(np.float32)
    border_pixels = np.concatenate([
        array[:5, :, :].reshape(-1, 3),
        array[-5:, :, :].reshape(-1, 3),
        array[:, :5, :].reshape(-1, 3),
        array[:, -5:, :].reshape(-1, 3)
    ], axis=0)
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(array - background, axis=2)
    threshold = max(18.0, float(np.percentile(distance, 75)) * 0.75)
    mask = distance > threshold
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        object_area_ratio = 0.0
        diameter_px = 0.0
    else:
        object_area_ratio = float(mask.mean())
        diameter_px = float(np.sqrt(4 * mask.sum() / np.pi))
    normalized_diameter = float(diameter_px / image_size)
    return object_area_ratio, normalized_diameter


def read_size_thresholds():
    if not SIZE_THRESHOLDS_PATH.exists():
        return None
    try:
        return json.loads(SIZE_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def visual_size_prediction(image):
    object_area_ratio, normalized_diameter = estimate_size_features(image)
    thresholds = read_size_thresholds()
    if thresholds and thresholds.get("global"):
        q1 = float(thresholds["global"]["q1"])
        q2 = float(thresholds["global"]["q2"])
    else:
        q1 = 0.46
        q2 = 0.64
    if normalized_diameter <= q1:
        label = "Pequeño"
        margin = q1 - normalized_diameter
    elif normalized_diameter <= q2:
        label = "Mediano"
        margin = min(normalized_diameter - q1, q2 - normalized_diameter)
    else:
        label = "Grande"
        margin = normalized_diameter - q2
    confidence = float(np.clip(58 + margin * 120, 52, 92))
    return {
        "label": label,
        "confidence": round(confidence, 1),
        "object_area_ratio": round(object_area_ratio, 4),
        "normalized_diameter": round(normalized_diameter, 4),
        "model": "Estimación visual"
    }


def provisional_quality_prediction(image):
    arr = np.asarray(image.resize((160, 160))).astype("float32") / 255.0
    gray = arr.mean(axis=2)
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = np.mean((max_channel - min_channel) / np.maximum(max_channel, 0.001))
    brightness = float(gray.mean())
    dark_area = float((gray < 0.23).mean())
    bright_area = float((gray > 0.74).mean())
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    texture = float((dx + dy) * 4.0)
    brightness_score = 1.0 - abs(brightness - 0.58) / 0.58
    color_score = min(1.0, saturation * 1.8)
    texture_score = min(1.0, texture * 2.4)
    exposure_penalty = min(0.35, dark_area * 0.45 + bright_area * 0.18)
    score = 0.42 * brightness_score + 0.32 * color_score + 0.26 * texture_score - exposure_penalty
    score = float(np.clip(score, 0, 1))
    if dark_area > 0.46 or score < 0.38:
        label = "Mala"
        base = 0.54 + (0.38 - min(score, 0.38))
    elif score >= 0.66:
        label = "Buena"
        base = 0.58 + (score - 0.66)
    else:
        label = "Regular"
        base = 0.58 + (0.16 - abs(score - 0.52))
    confidence = float(np.clip(base * 100, 50, 91))
    good = float(np.clip(score, 0.05, 0.9))
    bad = float(np.clip(1 - score - 0.18, 0.05, 0.85))
    regular = float(max(0.05, 1 - good - bad))
    total = good + regular + bad
    raw_scores = {"Buena": good / total, "Regular": regular / total, "Mala": bad / total}
    raw_scores[label] = max(raw_scores[label], confidence / 100)
    total = sum(raw_scores.values())
    scores = [{"label": name, "value": round(value / total * 100, 1)} for name, value in raw_scores.items()]
    scores = sorted(scores, key=lambda item: item["value"], reverse=True)
    return {"label": label, "confidence": round(confidence, 1), "scores": scores[:3], "model": "Análisis provisional"}


def predict(image, quality_model_name, size_model_name):
    quality_prediction = predict_model_task(image, quality_model_name, "quality") or provisional_quality_prediction(image)
    size_prediction = predict_model_task(image, size_model_name, "size")
    visual_size = visual_size_prediction(image)
    if size_prediction is None:
        size_prediction = visual_size
    return {
        "quality": quality_prediction["label"],
        "confidence": quality_prediction["confidence"],
        "destination": destination_for(quality_prediction["label"]),
        "size": size_prediction["label"],
        "size_confidence": size_prediction["confidence"],
        "object_area_ratio": visual_size["object_area_ratio"],
        "normalized_diameter": visual_size["normalized_diameter"],
        "quality_model": quality_prediction["model"],
        "size_model": size_prediction["model"],
        "scores": quality_prediction["scores"],
        "timestamp": int(time.time())
    }


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.serve_file(APP_DIR / "index.html")
            return
        if parsed.path == "/api/models":
            json_response(self, {"models": list_model_files()})
            return
        if parsed.path.startswith("/static/"):
            target = (APP_DIR / parsed.path.lstrip("/")).resolve()
            try:
                target.relative_to(APP_DIR.resolve())
            except ValueError:
                self.send_error(404)
                return
            self.serve_file(target)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404)
            return
        try:
            payload = read_json(self)
            image = decode_image(payload.get("image"))
            result = predict(image, payload.get("quality_model") or payload.get("model"), payload.get("size_model"))
            json_response(self, result)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=400)

    def serve_file(self, path):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
