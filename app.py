
#!/usr/bin/env python
from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request, send_from_directory

from foodclf import config
from foodclf import modelzoo as Z

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
UNCERTAIN_BELOW = float(os.environ.get("FOODCLF_UNCERTAIN_BELOW", "0.5"))

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ModelStore:
    def __init__(self, models_dir: Path, allow_unverified=False):
        self.dir = Path(models_dir)
        self.allow_unverified = allow_unverified
        self.valid = {}
        self.rejected = []
        self._loaded = {}
        self._metrics = {}
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self):
        self.valid, self.rejected = {}, []

        for entry in Z.load_registry(self.dir / "registry.json").get("models", []):
            reason = self._check(entry)

            if reason:
                self.rejected.append({
                    "id": entry.get("id"),
                    "reason": reason
                })
            else:
                self.valid[entry["id"]] = entry

    def _check(self, entry):
        weights = self.dir / entry.get("weights", "")
        metrics = self.dir / entry.get("metrics", "")

        if not weights.is_file() or not metrics.is_file():
            return "weights or metrics file missing"

        if Z.sha256_file(weights) != entry.get("sha256"):
            return "checksum mismatch"

        if not self.allow_unverified:
            if entry.get("dataset_id") != config.DATASET_ID:
                return "incorrect dataset"

            if entry.get("data_source") != "hub":
                return "not trained on the Hub dataset"

            if entry.get("partial_training_subset"):
                return "trained on a partial subset"

            if entry.get("pretrained") is False:
                return "trained without pretrained weights"

        model_dir = weights.parent
        onnx_file = model_dir / "food_classifier.onnx"
        metadata_file = model_dir / "food_metadata.json"

        if not onnx_file.is_file() or not metadata_file.is_file():
            return "ONNX model or metadata file missing"

        return None

    def metrics(self, mid):
        if mid not in self._metrics:
            path = self.dir / self.valid[mid]["metrics"]
            self._metrics[mid] = json.loads(path.read_text())

        return self._metrics[mid]

    def get(self, mid):
        with self._lock:
            if mid not in self._loaded:
                entry = self.valid[mid]
                model_dir = (self.dir / entry["weights"]).parent

                metadata = json.loads(
                    (model_dir / "food_metadata.json").read_text()
                )

                session = ort.InferenceSession(
                    str(model_dir / "food_classifier.onnx"),
                    providers=["CPUExecutionProvider"]
                )

                self._loaded[mid] = (session, metadata)

            return self._loaded[mid]


def preprocess(img, meta):
    size = int(meta["img_size"])
    resize_size = int(round(size * 1.14))

    width, height = img.size

    if width < height:
        new_width = resize_size
        new_height = int(resize_size * height / width)
    else:
        new_height = resize_size
        new_width = int(resize_size * width / height)

    img = img.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    left = int(round((new_width - size) / 2.0))
    top = int(round((new_height - size) / 2.0))

    img = img.crop((left, top, left + size, top + size))

    array = np.asarray(img, dtype=np.float32) / 255.0

    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)

    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


def evidence_for(metrics, cls):
    test = metrics["test"]["open_set_all_classes"]["per_class"]

    if cls in test:
        pc = test[cls]
        return {
            "basis": "official_test_split",
            "f1": pc["f1"],
            "precision": pc["precision"],
            "recall": pc["recall"],
            "n_images": pc["support"],
            "low_support": pc["low_support"]
        }

    pc = metrics["internal_holdout"]["open_set_all_classes"]["per_class"][cls]

    return {
        "basis": "internal_holdout_only",
        "f1": pc["f1"],
        "precision": pc["precision"],
        "recall": pc["recall"],
        "n_images": pc["support"],
        "low_support": pc["low_support"]
    }


def create_app(models_dir=None, allow_unverified=None):
    if allow_unverified is None:
        allow_unverified = os.environ.get("FOODCLF_ALLOW_UNVERIFIED") == "1"

    store = ModelStore(models_dir or config.MODELS_DIR, allow_unverified)

    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["STORE"] = store

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/status")
    def status():
        store.refresh()

        return jsonify({
            "ready": bool(store.valid),
            "n_models": len(store.valid),
            "rejected": store.rejected,
            "uncertain_below": UNCERTAIN_BELOW
        })

    @app.get("/api/models")
    def models():
        store.refresh()
        output = []

        for mid, entry in store.valid.items():
            metrics = store.metrics(mid)

            output.append({
                "id": mid,
                "arch": entry["arch"],
                "mode": entry["mode"],
                "created_at": entry["created_at"],
                "headline": entry["headline"],
                "training": metrics["training"],
                "class_coverage": metrics["class_coverage"]
            })

        return jsonify({"models": output})

    @app.get("/api/metrics/<mid>")
    def metrics(mid):
        if mid not in store.valid:
            return jsonify({"error": "unknown model"}), 404

        return jsonify(store.metrics(mid))

    @app.get("/api/dataset")
    def dataset():
        if config.DATASET_REPORT_PATH.exists():
            report = json.loads(config.DATASET_REPORT_PATH.read_text())
            source = "generated_by_inspect_dataset"
        elif config.DATASET_SNAPSHOT_PATH.exists():
            report = json.loads(config.DATASET_SNAPSHOT_PATH.read_text())
            source = "reference_snapshot_2026-10-11"
        else:
            report, source = {}, "none"

        return jsonify({
            "id": config.DATASET_ID,
            "url": config.DATASET_URL,
            "license": config.DATASET_LICENSE,
            "attribution": config.ATTRIBUTION,
            "source": source,
            "report": report
        })

    @app.post("/api/predict")
    def predict():
        store.refresh()

        if not store.valid:
            return jsonify({
                "error": "no_trained_models",
                "message": "No trained model is installed.",
                "rejected": store.rejected
            }), 503

        file = request.files.get("image")

        if file is None:
            return jsonify({
                "error": "missing_image",
                "message": "Attach an image in the image field."
            }), 400

        raw = file.read()

        try:
            probe = Image.open(io.BytesIO(raw))
            image_format = probe.format
            probe.verify()

            if image_format not in ALLOWED_FORMATS:
                return jsonify({
                    "error": "unsupported_format",
                    "message": "Use JPEG, PNG or WebP."
                }), 415

            img = Image.open(io.BytesIO(raw)).convert("RGB")

        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify({
                "error": "invalid_image",
                "message": "That file could not be read as an image."
            }), 400

        wanted = request.form.get("model", "")

        ids = (
            list(store.valid)
            if wanted == "all"
            else [wanted or next(iter(store.valid))]
        )

        if any(mid not in store.valid for mid in ids):
            return jsonify({
                "error": "unknown_model",
                "message": "Choose one of the listed models."
            }), 400

        try:
            top_k = max(1, min(10, int(request.form.get("top_k", 5))))
        except ValueError:
            return jsonify({
                "error": "invalid_top_k",
                "message": "top_k must be a whole number."
            }), 400

        results = []

        for mid in ids:
            session, meta = store.get(mid)

            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            image_array = preprocess(img, meta)

            logits = session.run(
                [output_name],
                {input_name: image_array}
            )[0][0]

            # Stable softmax
            shifted = logits - np.max(logits)
            exp_values = np.exp(shifted)
            probabilities = exp_values / np.sum(exp_values)

            indices = np.argsort(probabilities)[::-1][:top_k]

            metrics_data = store.metrics(mid)

            predictions = []

            for index in indices:
                class_name = meta["class_names"][int(index)]

                predictions.append({
                    "class": class_name,
                    "probability": float(probabilities[index]),
                    "evidence": evidence_for(metrics_data, class_name)
                })

            results.append({
                "model": mid,
                "arch": meta["arch"],
                "predictions": predictions,
                "uncertain": predictions[0]["probability"] < UNCERTAIN_BELOW
            })

        return jsonify({
            "results": results,
            "uncertain_below": UNCERTAIN_BELOW,
            "note": "The model only knows the dishes it was trained on and will always name one of them, even for a photo that is not food."
        })

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({
            "error": "too_large",
            "message": "Images must be 8 MB or smaller."
        }), 413

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
