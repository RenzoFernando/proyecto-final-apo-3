from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import base64
import contextlib
import io
import json
import math
import mimetypes
import os
import pickle
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from PIL import Image, ImageOps

try:
    import joblib
except Exception:
    joblib = None

try:
    import cv2
except Exception:
    cv2 = None

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
SIZE_THRESHOLDS_PATH = ANNOTATIONS_DIR / "size_thresholds.json"
MODEL_EXTENSIONS = {".pkl", ".joblib", ".h5", ".keras"}
MODEL_CACHE = {}
RESAMPLE_FILTER = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
BASE_PROCESSED_IMAGE_SIZE = 128
MIN_CAPTURE_DISTANCE_CM = 10.0
MAX_CAPTURE_DISTANCE_CM = 100.0
DEFAULT_CAPTURE_DISTANCE_CM = 20.0

QUALITY_LABELS = ["bad", "regular", "good"]
SIZE_LABELS = ["small", "medium", "large"]
PAIR_ORDER = ["cnn", "random_forest", "svm"]
SUPPORTED_PRODUCT_KEYS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]

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
    "Buena": "Venta directa",
    "Regular": "Separar para revisión",
    "Mala": "Retirar de la línea"
}

FAMILY_NAMES = {
    "cnn": "CNN",
    "random_forest": "Random Forest",
    "svm": "SVM lineal",
    "custom": "Modelo personalizado"
}

PRODUCT_CATALOG = {
    "apple": {
        "label": "Manzana",
        "measure": "diámetro",
        "axis": "equivalent",
        "ranges": {"Pequeño": [4.8, 6.2], "Mediano": [6.2, 8.0], "Grande": [8.0, 10.0]}
    },
    "banana": {
        "label": "Banano",
        "measure": "longitud",
        "axis": "major",
        "ranges": {"Pequeño": [12.0, 16.0], "Mediano": [16.0, 20.5], "Grande": [20.5, 26.0]}
    },
    "guava": {
        "label": "Guayaba",
        "measure": "diámetro",
        "axis": "equivalent",
        "ranges": {"Pequeño": [4.0, 5.5], "Mediano": [5.5, 7.0], "Grande": [7.0, 9.0]}
    },
    "lime": {
        "label": "Limón / lima",
        "measure": "diámetro",
        "axis": "equivalent",
        "ranges": {"Pequeño": [3.5, 4.5], "Mediano": [4.5, 5.8], "Grande": [5.8, 7.2]}
    },
    "orange": {
        "label": "Naranja",
        "measure": "diámetro",
        "axis": "equivalent",
        "ranges": {"Pequeño": [5.5, 6.8], "Mediano": [6.8, 8.5], "Grande": [8.5, 10.5]}
    },
    "pomegranate": {
        "label": "Granada",
        "measure": "diámetro",
        "axis": "equivalent",
        "ranges": {"Pequeño": [6.0, 7.5], "Mediano": [7.5, 9.5], "Grande": [9.5, 12.0]}
    },
    "tomato": {
        "label": "Tomate",
        "measure": "diámetro",
        "axis": "equivalent",
        "ranges": {"Pequeño": [4.0, 5.5], "Mediano": [5.5, 7.5], "Grande": [7.5, 10.0]}
    },
    "potato": {
        "label": "Papa",
        "measure": "longitud",
        "axis": "major",
        "ranges": {"Pequeño": [4.0, 6.5], "Mediano": [6.5, 9.0], "Grande": [9.0, 12.5]}
    },
    "pepper": {
        "label": "Pimentón",
        "measure": "longitud",
        "axis": "major",
        "ranges": {"Pequeño": [5.0, 7.0], "Mediano": [7.0, 10.0], "Grande": [10.0, 14.0]}
    },
    "carrot": {
        "label": "Zanahoria",
        "measure": "longitud",
        "axis": "major",
        "ranges": {"Pequeño": [8.0, 12.0], "Mediano": [12.0, 18.0], "Grande": [18.0, 25.0]}
    },
    "onion": {
        "label": "Cebolla",
        "measure": "diámetro",
        "axis": "equivalent",
        "ranges": {"Pequeño": [4.0, 5.8], "Mediano": [5.8, 8.0], "Grande": [8.0, 11.0]}
    },
    "unknown": {
        "label": "Producto no especificado",
        "measure": "medida",
        "axis": "equivalent",
        "ranges": {"Pequeño": [4.0, 7.0], "Mediano": [7.0, 10.0], "Grande": [10.0, 14.0]}
    }
}

PRODUCT_ORDER = SUPPORTED_PRODUCT_KEYS


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


def infer_model_family(model_path, config=None):
    config = config or {}
    text = " ".join([
        model_path.stem.lower(),
        str(config.get("model_family", "")).lower(),
        str(config.get("variant", "")).lower(),
        str(config.get("model", "")).lower()
    ])
    if "random" in text or "forest" in text or "rf" in text:
        return "random_forest"
    if "svm" in text or "svc" in text or "support vector" in text or "support_vector_machine" in text:
        return "svm"
    if "cnn" in text or model_path.suffix.lower() in {".h5", ".keras"}:
        return "cnn"
    return "custom"


def infer_task(model_path, config=None):
    config = config or {}
    text = " ".join([
        model_path.stem.lower(),
        str(config.get("target", "")).lower(),
        str(config.get("task", "")).lower(),
        str(config.get("output", "")).lower()
    ])
    if "size" in text or "tamano" in text or "tamaño" in text:
        return "size"
    return "quality"


def default_config(model_path):
    family = infer_model_family(model_path, {})
    task = infer_task(model_path, {})
    img_size = 128 if family == "cnn" else 48
    labels = SIZE_LABELS if task == "size" else QUALITY_LABELS
    config = {
        "task": task,
        "target": task,
        "classes": labels,
        "labels": labels,
        "quality_labels": QUALITY_LABELS,
        "size_labels": SIZE_LABELS,
        "img_size": [img_size, img_size],
        "image_size": [img_size, img_size],
        "color_mode": "RGB",
        "scale": 255.0,
        "flatten": family != "cnn",
        "model_family": family
    }
    return config


def normalize_config(model_path, config):
    merged = default_config(model_path)
    merged.update(config or {})
    task = infer_task(model_path, merged)
    family = infer_model_family(model_path, merged)
    merged["task"] = task
    merged["target"] = task
    merged["model_family"] = family
    if not merged.get("classes") and not merged.get("labels"):
        merged["classes"] = SIZE_LABELS if task == "size" else QUALITY_LABELS
        merged["labels"] = merged["classes"]
    if task == "quality" and not merged.get("quality_labels"):
        merged["quality_labels"] = merged.get("classes") or QUALITY_LABELS
    if task == "size" and not merged.get("size_labels"):
        merged["size_labels"] = merged.get("classes") or SIZE_LABELS
    return merged


def read_config(model_path):
    for path in config_candidates(model_path):
        if path.exists():
            try:
                return normalize_config(model_path, json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    return normalize_config(model_path, {})


def model_kind_from(path, config):
    return infer_task(path, config)


def model_label(path, config):
    family = infer_model_family(path, config)
    task = infer_task(path, config)
    label = FAMILY_NAMES.get(family, FAMILY_NAMES["custom"])
    suffix = "calidad" if task == "quality" else "tamaño"
    return f"{label} - {suffix}"


def model_info(path):
    config = read_config(path)
    return {
        "name": path.name,
        "path": str(path.relative_to(PROJECT_ROOT)).replace(chr(92), "/"),
        "kind": model_kind_from(path, config),
        "family": infer_model_family(path, config),
        "label": model_label(path, config),
        "available": True
    }


def list_model_files():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(MODELS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
            items.append(model_info(path))
    return items


def list_model_pairs():
    models = list_model_files()
    grouped = {}
    for item in models:
        grouped.setdefault(item["family"], {})[item["kind"]] = item
    pairs = []
    ordered_families = [family for family in PAIR_ORDER if family in grouped]
    ordered_families += sorted([family for family in grouped if family not in ordered_families])
    for family in ordered_families:
        group = grouped[family]
        quality = group.get("quality")
        size = group.get("size")
        label = FAMILY_NAMES.get(family, FAMILY_NAMES["custom"])
        available = bool(quality and size)
        pairs.append({
            "key": family,
            "label": label if available else f"{label} incompleto",
            "quality_model": quality["name"] if quality else "",
            "size_model": size["name"] if size else "",
            "available": available
        })
    return pairs


def list_products():
    products = []
    for key in PRODUCT_ORDER:
        item = PRODUCT_CATALOG[key]
        products.append({
            "key": key,
            "label": item["label"],
            "measure": item["measure"]
        })
    return products


def pair_by_key(pair_key):
    for pair in list_model_pairs():
        if pair["key"] == pair_key and pair["available"]:
            return pair
    return None


def product_by_key(product_key):
    return PRODUCT_CATALOG.get(product_key) or PRODUCT_CATALOG["unknown"]


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



@contextlib.contextmanager
def suppress_external_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def quiet_predict(model, x):
    try:
        return model.predict(x, verbose=0)
    except TypeError:
        return model.predict(x)


def load_model(model_path):
    key = str(model_path.resolve())
    cached = MODEL_CACHE.get(key)
    if cached:
        return cached
    suffix = model_path.suffix.lower()
    config = read_config(model_path)
    if suffix in {".h5", ".keras"}:
        with suppress_external_output():
            from tensorflow.keras.models import load_model as keras_load_model
            model = keras_load_model(model_path)
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
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
    config = complete_config_from_model(model, config, model_type)
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
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) > 0:
        return [str(item) for item in list(classes)]
    if labels:
        return [str(item) for item in list(labels)]
    return SIZE_LABELS if task == "size" else QUALITY_LABELS


def image_size_from(config):
    size = config.get("img_size") or config.get("image_size") or config.get("target_image_size") or 128
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return (int(size[0]), int(size[1]))
    return (128, 128)


def infer_sklearn_image_size(model, config):
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        return image_size_from(config)
    for channels in [3, 1]:
        side = math.sqrt(int(n_features) / channels)
        rounded = int(round(side))
        if rounded > 0 and rounded * rounded * channels == int(n_features):
            config["color_mode"] = "RGB" if channels == 3 else "L"
            config["img_size"] = [rounded, rounded]
            config["image_size"] = [rounded, rounded]
            config["flatten"] = True
            return (rounded, rounded)
    return image_size_from(config)


def infer_keras_image_size(model, config):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 4:
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        if height and width:
            config["img_size"] = [int(width), int(height)]
            config["image_size"] = [int(width), int(height)]
            config["flatten"] = False
            if channels == 1:
                config["color_mode"] = "L"
            else:
                config["color_mode"] = "RGB"
            return (int(width), int(height))
    return image_size_from(config)


def complete_config_from_model(model, config, model_type):
    if model_type == "keras":
        infer_keras_image_size(model, config)
        config["flatten"] = False
    else:
        infer_sklearn_image_size(model, config)
        config["flatten"] = bool(config.get("flatten", True))
    return config


def preprocess_image(image, config, model_type):
    size = image_size_from(config)
    color_mode = str(config.get("color_mode", "RGB")).upper()
    scale = float(config.get("scale", 255.0))
    base_size = int(config.get("processed_base_size", BASE_PROCESSED_IMAGE_SIZE) or BASE_PROCESSED_IMAGE_SIZE)
    base = ImageOps.pad(image.convert("RGB"), (base_size, base_size), method=RESAMPLE_FILTER, color=(255, 255, 255), centering=(0.5, 0.5))
    padded = ImageOps.pad(base, size, method=RESAMPLE_FILTER, color=(255, 255, 255), centering=(0.5, 0.5))
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
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
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


def class_from_prediction(raw, labels):
    if isinstance(raw, (np.integer, int)) and int(raw) < len(labels):
        return labels[int(raw)]
    if isinstance(raw, (np.floating, float)) and float(raw).is_integer() and int(raw) < len(labels):
        return labels[int(raw)]
    return raw


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
        prediction = quiet_predict(model, x)
        if isinstance(prediction, list):
            prediction = np.asarray(prediction)
        arr = np.asarray(prediction)
        if arr.ndim >= 2 and arr.shape[-1] > 1:
            scores = scores_from_probabilities(arr[0], labels, normalizer)
            label = scores[0]["label"] if scores else "Sin clase"
            confidence = scores[0]["value"] if scores else 0.0
        else:
            raw = class_from_prediction(arr.reshape(-1)[0], labels)
            label = normalizer(raw)
            confidence = 100.0
            scores = [{"label": label, "value": confidence}]
    return {"label": label, "confidence": round(float(confidence), 1), "scores": scores[:3], "model": model_path.name}


def largest_component(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = []
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    clean = np.zeros_like(mask, dtype=bool)
    if best:
        yy, xx = zip(*best)
        clean[np.array(yy), np.array(xx)] = True
    return clean


def normalize_product_name(product_key):
    text = str(product_key or "unknown").lower().strip()
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def remove_small_components(mask, min_area):
    if cv2 is None:
        return mask
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    clean = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 1
    return clean.astype(bool)


def build_skin_mask(rgb_array):
    if cv2 is None:
        return np.zeros(rgb_array.shape[:2], dtype=bool)
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    h, s, v = cv2.split(hsv)
    y, cr, cb = cv2.split(ycrcb)
    skin_ycrcb = (cr >= 133) & (cr <= 178) & (cb >= 77) & (cb <= 135)
    skin_hsv = (h <= 25) & (s >= 20) & (s <= 135) & (v >= 50)
    return skin_ycrcb & skin_hsv


def build_fruit_color_mask(rgb_array, product_key="unknown"):
    if cv2 is None:
        return np.zeros(rgb_array.shape[:2], dtype=bool)
    product = normalize_product_name(product_key)
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red_mask = (((h <= 12) | (h >= 168)) & (s >= 45) & (v >= 35))
    green_mask = ((h >= 28) & (h <= 95) & (s >= 35) & (v >= 35))
    yellow_mask = ((h >= 12) & (h <= 42) & (s >= 35) & (v >= 45))
    orange_brown_mask = ((h >= 5) & (h <= 30) & (s >= 30) & (v >= 25) & (v <= 235))
    dark_damage_mask = ((s >= 25) & (v >= 20) & (v <= 120))
    if "lime" in product or "limon" in product or "lemon" in product:
        return green_mask | yellow_mask | orange_brown_mask | dark_damage_mask
    if "apple" in product or "manzana" in product:
        return red_mask | green_mask | yellow_mask | orange_brown_mask | dark_damage_mask
    if "orange" in product or "naranja" in product or "mandarina" in product:
        return yellow_mask | orange_brown_mask | dark_damage_mask
    if "tomato" in product or "tomate" in product:
        return red_mask | green_mask | orange_brown_mask | dark_damage_mask
    if "banana" in product or "banano" in product or "platano" in product:
        return yellow_mask | green_mask | dark_damage_mask
    return red_mask | green_mask | yellow_mask | orange_brown_mask | dark_damage_mask


def build_background_distance_mask(rgb_array):
    array = rgb_array.astype(np.float32)
    border = np.concatenate([
        array[:6, :, :].reshape(-1, 3),
        array[-6:, :, :].reshape(-1, 3),
        array[:, :6, :].reshape(-1, 3),
        array[:, -6:, :].reshape(-1, 3)
    ], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(array - background, axis=2)
    threshold = max(14.0, float(np.percentile(distance, 72)) * 0.70)
    return distance > threshold


def select_best_component(mask, rgb_array, product_key="unknown"):
    if cv2 is None:
        return np.zeros_like(mask, dtype=bool), None
    h, w = mask.shape
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    fruit_color_mask = build_fruit_color_mask(rgb_array, product_key)
    best_label = None
    best_score = -np.inf
    best_payload = None
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < h * w * 0.004 or area > h * w * 0.78:
            continue
        component = labels == label
        component_area_ratio = area / (h * w)
        color_ratio = float(fruit_color_mask[component].mean()) if area > 0 else 0.0
        center_x, center_y = centroids[label]
        center_distance = math.sqrt(((center_x - w / 2) / w) ** 2 + ((center_y - h / 2) / h) ** 2)
        border_touch = int(x <= 1 or y <= 1 or x + width >= w - 1 or y + height >= h - 1)
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = float(4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        score = (component_area_ratio * 3.0) + (color_ratio * 1.5) + (circularity * 0.7) - (center_distance * 1.2) - (border_touch * 1.0)
        if score > best_score:
            best_score = score
            best_label = label
            best_payload = {"x": x, "y": y, "width": width, "height": height, "area": area, "color_ratio": color_ratio, "circularity": circularity, "center_distance": center_distance, "border_touch": border_touch}
    if best_label is None:
        return np.zeros_like(mask, dtype=bool), None
    return labels == best_label, best_payload


def segment_fruit(rgb_array, product_key="unknown"):
    if cv2 is None:
        return np.zeros(rgb_array.shape[:2], dtype=bool), np.zeros(rgb_array.shape[:2], dtype=bool), np.zeros(rgb_array.shape[:2], dtype=bool), None
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    fruit_color_mask = build_fruit_color_mask(rgb_array, product_key)
    background_mask = build_background_distance_mask(rgb_array)
    skin_mask = build_skin_mask(rgb_array)
    saturation_mask = (saturation >= 38) & (value >= 30)
    candidate = (fruit_color_mask | (background_mask & saturation_mask)) & (~skin_mask)
    candidate = remove_small_components(candidate, int(rgb_array.shape[0] * rgb_array.shape[1] * 0.0025))
    kernel = np.ones((5, 5), np.uint8)
    candidate_u8 = candidate.astype(np.uint8) * 255
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    candidate = candidate_u8 > 0
    selected_mask, payload = select_best_component(candidate, rgb_array, product_key)
    return selected_mask, skin_mask, fruit_color_mask, payload


def segmentation_confidence(payload, object_area_ratio):
    if payload is None:
        return 0.0
    value = payload["color_ratio"] * 0.45 + min(object_area_ratio / 0.20, 1.0) * 0.35 + payload["circularity"] * 0.20 - payload["border_touch"] * 0.20
    return float(np.clip(value, 0.0, 1.0))


def prepare_inference_source_image(image, product_key="unknown"):
    working_image = ImageOps.pad(image.convert("RGB"), (BASE_PROCESSED_IMAGE_SIZE, BASE_PROCESSED_IMAGE_SIZE), method=RESAMPLE_FILTER, color=(255, 255, 255), centering=(0.5, 0.5))
    if cv2 is None:
        return working_image
    rgb_array = np.asarray(working_image).astype(np.uint8)
    selected_mask, skin_mask, fruit_color_mask, payload = segment_fruit(rgb_array, product_key)
    if payload is None or selected_mask.sum() == 0:
        return working_image
    object_area_ratio = float(payload["area"] / selected_mask.size)
    if segmentation_confidence(payload, object_area_ratio) < 0.18:
        return working_image
    x = payload["x"]
    y = payload["y"]
    width = payload["width"]
    height = payload["height"]
    margin = int(max(width, height) * 0.18)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(BASE_PROCESSED_IMAGE_SIZE, x + width + margin)
    bottom = min(BASE_PROCESSED_IMAGE_SIZE, y + height + margin)
    crop = working_image.crop((left, top, right, bottom))
    return ImageOps.pad(crop, (BASE_PROCESSED_IMAGE_SIZE, BASE_PROCESSED_IMAGE_SIZE), method=RESAMPLE_FILTER, color=(255, 255, 255), centering=(0.5, 0.5))


def estimate_size_features_fallback(image, image_size=180):
    working_image = ImageOps.pad(image, (image_size, image_size), method=RESAMPLE_FILTER, color=(255, 255, 255), centering=(0.5, 0.5))
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
    if mask.mean() > 0.72:
        threshold = max(threshold, float(np.percentile(distance, 88)))
        mask = distance > threshold
    mask = largest_component(mask)
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        object_area_ratio = 0.0
        diameter_px = 0.0
        major_axis_px = 0.0
        minor_axis_px = 0.0
        bbox_width_px = 0.0
        bbox_height_px = 0.0
    else:
        object_area_ratio = float(mask.mean())
        diameter_px = float(np.sqrt(4 * mask.sum() / np.pi))
        bbox_width_px = float(x_indices.max() - x_indices.min() + 1)
        bbox_height_px = float(y_indices.max() - y_indices.min() + 1)
        major_axis_px = max(bbox_width_px, bbox_height_px)
        minor_axis_px = min(bbox_width_px, bbox_height_px)
    normalized_diameter = float(diameter_px / image_size)
    normalized_major_axis = float(major_axis_px / image_size)
    normalized_minor_axis = float(minor_axis_px / image_size)
    return {
        "object_area_ratio": round(object_area_ratio, 4),
        "normalized_diameter": round(normalized_diameter, 4),
        "normalized_major_axis": round(normalized_major_axis, 4),
        "normalized_minor_axis": round(normalized_minor_axis, 4),
        "bbox_width_ratio": round(float(bbox_width_px / image_size), 4),
        "bbox_height_ratio": round(float(bbox_height_px / image_size), 4),
        "segmentation_confidence": 0.0,
        "skin_noise_ratio": 0.0
    }


def estimate_size_features(image, product_key="unknown", image_size=180):
    if cv2 is None:
        return estimate_size_features_fallback(image, image_size)
    working_image = ImageOps.pad(image.convert("RGB"), (image_size, image_size), method=RESAMPLE_FILTER, color=(255, 255, 255), centering=(0.5, 0.5))
    rgb_array = np.asarray(working_image).astype(np.uint8)
    selected_mask, skin_mask, fruit_color_mask, payload = segment_fruit(rgb_array, product_key)
    if payload is None or selected_mask.sum() == 0:
        return estimate_size_features_fallback(image, image_size)
    area = float(payload["area"])
    bbox_width_px = float(payload["width"])
    bbox_height_px = float(payload["height"])
    object_area_ratio = area / float(image_size * image_size)
    equivalent_diameter = 2.0 * math.sqrt(area / math.pi)
    bbox_diameter = max(bbox_width_px, bbox_height_px)
    diameter_px = float(0.55 * bbox_diameter + 0.45 * equivalent_diameter)
    major_axis_px = max(bbox_width_px, bbox_height_px)
    minor_axis_px = min(bbox_width_px, bbox_height_px)
    return {
        "object_area_ratio": round(float(object_area_ratio), 4),
        "normalized_diameter": round(float(diameter_px / image_size), 4),
        "normalized_major_axis": round(float(major_axis_px / image_size), 4),
        "normalized_minor_axis": round(float(minor_axis_px / image_size), 4),
        "bbox_width_ratio": round(float(bbox_width_px / image_size), 4),
        "bbox_height_ratio": round(float(bbox_height_px / image_size), 4),
        "segmentation_confidence": round(segmentation_confidence(payload, object_area_ratio), 4),
        "skin_noise_ratio": round(float(skin_mask.mean()), 4)
    }

def read_size_thresholds():
    if not SIZE_THRESHOLDS_PATH.exists():
        return None
    try:
        return json.loads(SIZE_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def visual_metric_from_product(features, product_key):
    product = product_by_key(product_key)
    if product.get("axis") == "major":
        return float(features.get("normalized_major_axis", 0.0))
    return float(features.get("normalized_diameter", 0.0))


def visual_size_prediction(image, product_key="unknown"):
    features = estimate_size_features(image, product_key)
    normalized_metric = visual_metric_from_product(features, product_key)
    thresholds = read_size_thresholds()
    if thresholds and thresholds.get("global"):
        q1 = float(thresholds["global"]["q1"])
        q2 = float(thresholds["global"]["q2"])
    else:
        q1 = 0.46
        q2 = 0.64
    if normalized_metric <= q1:
        label = "Pequeño"
        margin = q1 - normalized_metric
    elif normalized_metric <= q2:
        label = "Mediano"
        margin = min(normalized_metric - q1, q2 - normalized_metric)
    else:
        label = "Grande"
        margin = normalized_metric - q2
    confidence = float(np.clip(58 + margin * 120, 52, 92))
    features["normalized_metric"] = round(normalized_metric, 4)
    features["model"] = "Estimación visual"
    return {
        "label": label,
        "confidence": round(confidence, 1),
        **features
    }


def clamp_float(value, default, minimum, maximum):
    try:
        number = float(value)
    except Exception:
        number = default
    if not math.isfinite(number):
        number = default
    return float(np.clip(number, minimum, maximum))


def range_position(normalized_metric, size_label):
    q1 = 0.46
    q2 = 0.64
    if size_label == "Pequeño":
        position = normalized_metric / q1 if q1 else 0.5
    elif size_label == "Mediano":
        position = (normalized_metric - q1) / (q2 - q1) if q2 > q1 else 0.5
    elif size_label == "Grande":
        position = (normalized_metric - q2) / (1 - q2) if q2 < 1 else 0.5
    else:
        position = 0.5
    return float(np.clip(position, 0.15, 0.85))


def estimate_physical_measure(product_key, size_label, visual_size, calibration):
    product = product_by_key(product_key)
    ranges = product.get("ranges", PRODUCT_CATALOG["unknown"]["ranges"])
    selected_range = ranges.get(size_label) or ranges.get("Mediano") or PRODUCT_CATALOG["unknown"]["ranges"]["Mediano"]
    normalized_metric = float(visual_size.get("normalized_metric", 0.0))
    position = range_position(normalized_metric, size_label)
    range_min = float(selected_range[0])
    range_max = float(selected_range[1])
    class_estimate = range_min + (range_max - range_min) * position
    distance_cm = clamp_float(calibration.get("distance_cm"), DEFAULT_CAPTURE_DISTANCE_CM, MIN_CAPTURE_DISTANCE_CM, MAX_CAPTURE_DISTANCE_CM)
    fov_degrees = clamp_float(calibration.get("fov_degrees"), 60.0, 30.0, 90.0)
    roi_ratio = clamp_float(calibration.get("roi_ratio"), 1.0, 0.35, 1.0)
    visible_span_cm = 2.0 * distance_cm * math.tan(math.radians(fov_degrees) / 2.0) * roi_ratio
    optical_estimate = normalized_metric * visible_span_cm
    plausible_min = range_min * 0.55
    plausible_max = range_max * 1.8
    if plausible_min <= optical_estimate <= plausible_max:
        final_estimate = class_estimate * 0.55 + optical_estimate * 0.45
        method = "clase + calibración óptica"
    else:
        final_estimate = class_estimate
        method = "rango por producto + segmentación"
    return {
        "value_cm": round(float(final_estimate), 1),
        "optical_cm": round(float(optical_estimate), 1),
        "range_min_cm": round(range_min, 1),
        "range_max_cm": round(range_max, 1),
        "measure": product["measure"],
        "method": method,
        "distance_cm": round(distance_cm, 1),
        "fov_degrees": round(fov_degrees, 1),
        "roi_ratio": round(roi_ratio, 3)
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


def predict(image, quality_model_name, size_model_name, product_key="unknown", calibration=None):
    calibration = calibration or {}
    product_key = product_key if product_key in PRODUCT_CATALOG else "unknown"
    product = product_by_key(product_key)
    model_image = prepare_inference_source_image(image, product_key)
    quality_prediction = predict_model_task(model_image, quality_model_name, "quality") or provisional_quality_prediction(model_image)
    size_prediction = predict_model_task(model_image, size_model_name, "size")
    visual_size = visual_size_prediction(image, product_key)
    if size_prediction is None:
        size_prediction = visual_size
    physical_measure = estimate_physical_measure(product_key, size_prediction["label"], visual_size, calibration)
    return {
        "product_key": product_key,
        "product_label": product["label"],
        "product_measure": product["measure"],
        "quality": quality_prediction["label"],
        "confidence": quality_prediction["confidence"],
        "destination": destination_for(quality_prediction["label"]),
        "size": size_prediction["label"],
        "size_confidence": size_prediction["confidence"],
        "size_cm": physical_measure["value_cm"],
        "optical_cm": physical_measure["optical_cm"],
        "size_range_min_cm": physical_measure["range_min_cm"],
        "size_range_max_cm": physical_measure["range_max_cm"],
        "measure_method": physical_measure["method"],
        "distance_cm": physical_measure["distance_cm"],
        "fov_degrees": physical_measure["fov_degrees"],
        "roi_ratio": physical_measure["roi_ratio"],
        "object_area_ratio": visual_size["object_area_ratio"],
        "normalized_diameter": visual_size["normalized_diameter"],
        "normalized_major_axis": visual_size["normalized_major_axis"],
        "normalized_minor_axis": visual_size["normalized_minor_axis"],
        "normalized_metric": visual_size["normalized_metric"],
        "quality_model": quality_prediction["model"],
        "size_model": size_prediction["model"],
        "scores": quality_prediction["scores"]
    }


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.serve_file(APP_DIR / "index.html")
            return
        if parsed.path == "/api/models":
            json_response(self, {"models": list_model_files(), "pairs": list_model_pairs(), "products": list_products()})
            return
        if parsed.path == "/api/products":
            json_response(self, {"products": list_products()})
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
            quality_model = payload.get("quality_model") or payload.get("model")
            size_model = payload.get("size_model")
            pair_key = payload.get("pair_key")
            if pair_key:
                pair = pair_by_key(pair_key)
                if pair:
                    quality_model = pair["quality_model"]
                    size_model = pair["size_model"]
            calibration = {
                "distance_cm": payload.get("distance_cm"),
                "fov_degrees": payload.get("fov_degrees"),
                "roi_ratio": payload.get("roi_ratio")
            }
            result = predict(image, quality_model, size_model, payload.get("product_key") or "unknown", calibration)
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Aplicación detenida correctamente.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()