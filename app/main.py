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
MODEL_EXTENSIONS = {".pkl", ".joblib", ".h5", ".keras"}
MODEL_CACHE = {}

QUALITY_NAMES = {
    "good": "Buena",
    "bueno": "Buena",
    "buena": "Buena",
    "healthy": "Buena",
    "fresh": "Buena",
    "regular": "Regular",
    "medium": "Regular",
    "normal": "Regular",
    "mala": "Mala",
    "malo": "Mala",
    "bad": "Mala",
    "poor": "Mala",
    "rotten": "Mala",
    "damaged": "Mala"
}

DESTINATIONS = {
    "Buena": "Consumo",
    "Regular": "Revisión",
    "Mala": "Descarte"
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


def list_model_files():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(MODELS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
            items.append({"name": path.name, "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")})
    return items


def safe_model_name(name):
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
    MODEL_CACHE[key] = {"model": model, "config": config, "type": model_type}
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


def normalized_label(value):
    text = str(value).strip()
    key = text.lower().replace("_", " ").replace("-", " ")
    for alias, name in QUALITY_NAMES.items():
        if alias in key.split() or alias in key:
            return name
    return text[:1].upper() + text[1:] if text else "Sin clase"


def destination_for(label):
    return DESTINATIONS.get(label, "Clasificación")


def labels_from(model, config):
    labels = config.get("classes") or config.get("labels") or config.get("quality_labels")
    if labels:
        return list(labels)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return [str(item) for item in list(classes)]
    return ["bad", "regular", "good"]


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
    if color_mode in {"L", "GRAY", "GRAYSCALE"}:
        prepared = image.convert("L").resize(size)
    else:
        prepared = image.convert("RGB").resize(size)
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


def scores_from_probabilities(probabilities, labels):
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
        scores.append({"label": normalized_label(label), "value": round(float(value) * 100, 1)})
    return sorted(scores, key=lambda item: item["value"], reverse=True)


def predict_with_model(image, model_name):
    model_path = safe_model_name(model_name)
    if model_path is None:
        return provisional_prediction(image, "Análisis provisional")
    package = load_model(model_path)
    model = package["model"]
    config = package["config"]
    model_type = package["type"]
    labels = labels_from(model, config)
    x = preprocess_image(image, config, model_type)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]
        scores = scores_from_probabilities(probabilities, labels)
        label = scores[0]["label"] if scores else "Sin clase"
        confidence = scores[0]["value"] if scores else 0.0
    else:
        prediction = model.predict(x)
        if isinstance(prediction, list):
            prediction = np.asarray(prediction)
        arr = np.asarray(prediction)
        if arr.ndim >= 2 and arr.shape[-1] > 1:
            scores = scores_from_probabilities(arr[0], labels)
            label = scores[0]["label"] if scores else "Sin clase"
            confidence = scores[0]["value"] if scores else 0.0
        else:
            raw = arr.reshape(-1)[0]
            if isinstance(raw, (np.integer, int)) and int(raw) < len(labels):
                raw = labels[int(raw)]
            label = normalized_label(raw)
            confidence = 100.0
            scores = [{"label": label, "value": confidence}]
    return {
        "quality": label,
        "confidence": round(float(confidence), 1),
        "destination": destination_for(label),
        "model": model_path.name,
        "scores": scores[:3],
        "timestamp": int(time.time())
    }


def provisional_prediction(image, model_name):
    arr = np.asarray(image.resize((160, 160))).astype("float32") / 255.0
    gray = arr.mean(axis=2)
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = np.mean((max_channel - min_channel) / np.maximum(max_channel, 0.001))
    brightness = float(gray.mean())
    dark_area = float((gray < 0.23).mean())
    bright_area = float((gray > 0.72).mean())
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
    return {
        "quality": label,
        "confidence": round(confidence, 1),
        "destination": destination_for(label),
        "model": model_name,
        "scores": scores[:3],
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
            result = predict_with_model(image, payload.get("model"))
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
