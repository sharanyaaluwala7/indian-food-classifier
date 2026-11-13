"""End-to-end tests on SYNTHETIC data (coloured noise images).

These verify the code paths only. The synthetic runs are written to a temp
directory, are marked as non-Hub / non-pretrained, and the app rejects them
unless FOODCLF_ALLOW_UNVERIFIED is set. No number from here is a result.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foodclf import data as D  # noqa: E402

TRAIN_CLASSES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
EVAL_CLASSES = ["alpha", "bravo", "charlie", "delta"]  # echo, foxtrot absent from val/test


def _img(cls_idx, rng):
    base = np.zeros((64, 64, 3), dtype=np.float32)
    base[..., cls_idx % 3] = 200
    base[..., (cls_idx + 1) % 3] = 40 * (cls_idx // 3 + 1)
    base += rng.normal(0, 25, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype("uint8"))


def _split(n_per, classes, rng, features):
    import datasets

    imgs, labels = [], []
    for c in classes:
        for _ in range(n_per):
            imgs.append(_img(TRAIN_CLASSES.index(c), rng))
            labels.append(TRAIN_CLASSES.index(c))
    return datasets.Dataset.from_dict({"image": imgs, "label": labels}, features=features)


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    import datasets

    rng = np.random.default_rng(0)
    feats = datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=TRAIN_CLASSES)})
    dsd = datasets.DatasetDict({
        "train": _split(20, TRAIN_CLASSES, rng, feats),
        "validation": _split(4, EVAL_CLASSES, rng, feats),
        "test": _split(4, EVAL_CLASSES, rng, feats),
    })
    p = tmp_path_factory.mktemp("synthetic") / "ds"
    dsd.save_to_disk(str(p))
    return p


def test_coverage_detects_missing_classes(synthetic_dir):
    dsd = D.load_dataset_dict("local", str(synthetic_dir))
    cov = D.analyze_coverage(dsd)
    assert cov.train_classes == sorted(TRAIN_CLASSES)
    assert cov.eval_classes == sorted(EVAL_CLASSES)
    assert cov.unevaluated_classes == ["echo", "foxtrot"]
    assert cov.unknown_eval_classes == []
    D.require_valid_coverage(cov)


def test_unknown_eval_class_is_rejected():
    cov = D.CoverageReport({}, ["a"], ["a"], [], [], [], ["zzz"])
    with pytest.raises(ValueError):
        D.require_valid_coverage(cov)


def test_holdout_is_stratified_and_disjoint():
    labels = ["a"] * 20 + ["b"] * 20
    fit, hold = D.stratified_holdout(labels, 0.1, seed=1)
    assert not set(fit) & set(hold)
    assert len(fit) + len(hold) == 40
    assert {labels[i] for i in hold} == {"a", "b"}


@pytest.fixture(scope="module")
def trained(synthetic_dir, tmp_path_factory):
    import train

    models = tmp_path_factory.mktemp("models")
    run_id = train.main([
        "--arch", "mobilenetv3_large_100", "--mode", "finetune", "--epochs", "3", "--batch-size", "16",
        "--img-size", "64", "--num-workers", "0", "--no-pretrained", "--device", "cpu",
        "--data-source", "local", "--local-dataset-dir", str(synthetic_dir),
        "--models-dir", str(models), "--run-name", "synthetic-test", "--lr", "1e-3",
    ])
    return models, run_id


def test_metrics_respect_class_coverage(trained):
    models, run_id = trained
    m = json.loads((models / run_id / "metrics.json").read_text())
    assert set(m["test"]["open_set_all_classes"]["per_class"]) == set(EVAL_CLASSES)
    assert set(m["internal_holdout"]["open_set_all_classes"]["per_class"]) == set(TRAIN_CLASSES)
    assert m["class_coverage"]["unevaluated_on_official_splits"] == ["echo", "foxtrot"]
    o, c = m["test"]["open_set_all_classes"], m["test"]["closed_set_split_classes"]
    assert c["accuracy"] >= o["accuracy"] - 1e-9  # restricting choices cannot hurt
    assert o["n_images"] == 16


def _png():
    b = io.BytesIO()
    Image.fromarray((np.random.rand(80, 80, 3) * 255).astype("uint8")).save(b, "PNG")
    b.seek(0)
    return b


def test_app_without_models_returns_503(tmp_path):
    from app import create_app

    c = create_app(tmp_path, allow_unverified=False).test_client()
    assert c.get("/api/status").get_json()["ready"] is False
    r = c.post("/api/predict", data={"image": (_png(), "x.png")}, content_type="multipart/form-data")
    assert r.status_code == 503 and r.get_json()["error"] == "no_trained_models"


def test_app_rejects_synthetic_run_in_strict_mode(trained):
    from app import create_app

    models, run_id = trained
    c = create_app(models, allow_unverified=False).test_client()
    st = c.get("/api/status").get_json()
    assert st["ready"] is False and st["rejected"][0]["id"] == run_id


def test_app_predicts_in_test_mode(trained):
    from app import create_app

    models, run_id = trained
    c = create_app(models, allow_unverified=True).test_client()
    assert c.get("/api/status").get_json()["ready"] is True
    r = c.post("/api/predict", data={"image": (_png(), "x.png"), "model": run_id}, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200
    preds = body["results"][0]["predictions"]
    assert abs(sum(p["probability"] for p in c.post("/api/predict", data={"image": (_png(), "x.png"), "top_k": "10"},
               content_type="multipart/form-data").get_json()["results"][0]["predictions"]) - 1.0) < 1e-4
    assert preds[0]["evidence"]["basis"] in {"official_test_split", "internal_holdout_only"}
    bad = c.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "x.png")}, content_type="multipart/form-data")
    assert bad.status_code == 400


def test_tampered_weights_rejected(trained, tmp_path):
    import shutil
    from app import create_app

    models, run_id = trained
    copy = tmp_path / "m"
    shutil.copytree(models, copy)
    with open(copy / run_id / "model.pt", "ab") as f:
        f.write(b"x")
    st = create_app(copy, allow_unverified=True).test_client().get("/api/status").get_json()
    assert st["ready"] is False and "checksum" in st["rejected"][0]["reason"]
