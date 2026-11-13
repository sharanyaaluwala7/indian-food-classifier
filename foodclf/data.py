"""Dataset loading and class-coverage handling.

The Hub dataset has 43 classes in `train` but only 22 (the same 22) in
`validation` and `test`. Consequences handled here:

* The label space is fixed from the TRAIN split (43 classes, sorted by name).
* Evaluation splits are checked to be subsets of that label space.
* `CoverageReport` records which classes can and cannot be evaluated on the
  official validation/test splits.
* `stratified_holdout` carves an internal holdout out of `train`, so that all
  43 classes get *some* held-out measurement (used for model selection).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence

from . import config


# --------------------------------------------------------------------------- loading
def load_dataset_dict(source: str = "hub", local_dir: str | None = None, cache_dir: str | None = None):
    """Return a `datasets.DatasetDict` with train/validation/test splits."""
    import datasets

    if source == "hub":
        return datasets.load_dataset(config.DATASET_ID, cache_dir=cache_dir)
    if source == "local":
        if not local_dir:
            raise ValueError("--local-dataset-dir is required when --data-source local")
        return datasets.load_from_disk(local_dir)
    raise ValueError(f"unknown data source: {source}")


def label_names_of(split_ds) -> List[str] | None:
    """Names from a ClassLabel feature, or None if labels are plain strings."""
    feat = split_ds.features.get("label")
    names = getattr(feat, "names", None)
    return list(names) if names else None


def label_strings(split_ds) -> List[str]:
    """Label of every row as a string, without decoding any image."""
    raw = split_ds["label"]
    names = label_names_of(split_ds)
    if names is None:
        return [str(x) for x in raw]
    return [names[int(x)] for x in raw]


# --------------------------------------------------------------------------- coverage
@dataclass
class CoverageReport:
    split_sizes: Dict[str, int]
    train_classes: List[str]
    eval_classes: List[str]          # classes present in BOTH validation and test
    val_only_classes: List[str]
    test_only_classes: List[str]
    unevaluated_classes: List[str]   # in train, absent from validation and test
    unknown_eval_classes: List[str]  # in val/test but not in train (must be empty)
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def analyze_coverage(dsd) -> CoverageReport:
    counts = {s: dict(Counter(label_strings(dsd[s]))) for s in (config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT)}
    train = sorted(counts[config.TRAIN_SPLIT])
    val = set(counts[config.VAL_SPLIT])
    test = set(counts[config.TEST_SPLIT])
    shared = sorted(val & test)
    return CoverageReport(
        split_sizes={s: len(dsd[s]) for s in counts},
        train_classes=train,
        eval_classes=shared,
        val_only_classes=sorted(val - test),
        test_only_classes=sorted(test - val),
        unevaluated_classes=sorted(set(train) - set(shared)),
        unknown_eval_classes=sorted((val | test) - set(train)),
        counts=counts,
    )


def require_valid_coverage(cov: CoverageReport) -> None:
    if cov.unknown_eval_classes:
        raise ValueError(
            "validation/test contain classes that never appear in train: "
            f"{cov.unknown_eval_classes}. Their labels cannot be predicted by a model trained on train."
        )


# --------------------------------------------------------------------------- holdout
def stratified_holdout(labels: Sequence[str], frac: float, seed: int):
    """Indices (fit_idx, holdout_idx) with every class represented in both parts."""
    import numpy as np

    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    fit, hold = [], []
    for y, idx in sorted(by_class.items()):
        idx = list(rng.permutation(idx))
        n_hold = max(1, int(round(len(idx) * frac)))
        if n_hold >= len(idx):
            raise ValueError(f"class {y!r} has too few images ({len(idx)}) for a holdout")
        hold += idx[:n_hold]
        fit += idx[n_hold:]
    return sorted(fit), sorted(hold)


# --------------------------------------------------------------------------- torch dataset
def build_transforms(img_size: int, train: bool):
    from torchvision import transforms as T

    norm = T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(),
            norm,
        ])
    resize = int(round(img_size * 1.14))
    return T.Compose([T.Resize(resize), T.CenterCrop(img_size), T.ToTensor(), norm])


class HFImageDataset:
    """Wraps a HF split (optionally a subset of indices) as a torch-style dataset."""

    def __init__(self, split_ds, class_to_idx: Dict[str, int], transform, indices: Sequence[int] | None = None):
        self.ds = split_ds
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.indices = list(indices) if indices is not None else list(range(len(split_ds)))
        names = label_strings(split_ds)
        self.targets = [class_to_idx[names[i]] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        row = self.ds[self.indices[k]]
        img = row["image"].convert("RGB")
        return self.transform(img), self.targets[k]


# --------------------------------------------------------------------------- leakage check
def dhash(img, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size))
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def image_fingerprints(split_ds):
    """(sha256 of decoded pixels, 64-bit difference hash) per row. Decodes every image."""
    out = []
    for i in range(len(split_ds)):
        img = split_ds[i]["image"].convert("RGB")
        out.append((hashlib.sha256(img.tobytes()).hexdigest(), dhash(img)))
    return out
