"""Model construction, checkpoint saving/loading and the registry the website reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from . import config

# Architectures offered by train.py (all available in timm). Each is an ImageNet
# pretrained backbone whose classifier head is replaced with a 43-way head.
ARCHS = {
    "mobilenetv3_large_100": {"img_size": 224, "note": "small, fast on CPU"},
    "efficientnet_b0": {"img_size": 224, "note": "balanced"},
    "convnext_tiny": {"img_size": 224, "note": "larger, usually most accurate"},
}


def build_model(arch: str, num_classes: int, pretrained: bool):
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Path = config.REGISTRY_PATH) -> Dict:
    if not path.exists():
        return {"models": []}
    return json.loads(path.read_text())


def add_to_registry(entry: Dict, path: Path = config.REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = load_registry(path)
    reg["models"] = [m for m in reg["models"] if m["id"] != entry["id"]] + [entry]
    path.write_text(json.dumps(reg, indent=2))


def save_checkpoint(path: Path, state_dict, meta: Dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "meta": meta}, path)


def load_checkpoint(path: Path, device="cpu"):
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], len(meta["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta
