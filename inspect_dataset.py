#!/usr/bin/env python
"""Inspect the real dataset and write reports/dataset_report.json.

    python inspect_dataset.py                 # coverage, counts, licence, comparison to reference snapshot
    python inspect_dataset.py --check-leakage # also look for exact / near-duplicate images across splits (decodes all images)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from foodclf import config
from foodclf import data as D


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.array([bin(int(v)).count("1") for v in x.ravel()]).reshape(x.shape)


def leakage(dsd, max_hamming: int = 4):
    fp = {s: D.image_fingerprints(dsd[s]) for s in (config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT)}
    train_sha = {h for h, _ in fp[config.TRAIN_SPLIT]}
    train_hash = np.array([h for _, h in fp[config.TRAIN_SPLIT]], dtype=np.uint64)
    out = {}
    for s in (config.VAL_SPLIT, config.TEST_SPLIT):
        exact = sum(h in train_sha for h, _ in fp[s])
        q = np.array([h for _, h in fp[s]], dtype=np.uint64)
        near = 0
        for i in range(0, len(q), 256):
            d = popcount64(q[i:i + 256, None] ^ train_hash[None, :])
            near += int((d.min(1) <= max_hamming).sum())
        out[s] = {"exact_duplicates_of_train": int(exact), f"near_duplicates_dhash_le_{max_hamming}": near, "n": len(q)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="hub", choices=["hub", "local"])
    ap.add_argument("--local-dataset-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--check-leakage", action="store_true")
    a = ap.parse_args()

    dsd = D.load_dataset_dict(a.data_source, a.local_dataset_dir, a.cache_dir)
    cov = D.analyze_coverage(dsd)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": config.DATASET_ID,
        "license": config.DATASET_LICENSE,
        "attribution": config.ATTRIBUTION,
        "coverage": cov.to_dict(),
        "total_images": sum(cov.split_sizes.values()),
    }
    if config.DATASET_SNAPSHOT_PATH.exists():
        snap = json.loads(config.DATASET_SNAPSHOT_PATH.read_text())
        mism = []
        if snap["split_sizes"] != {k: cov.split_sizes[k] for k in snap["split_sizes"]}:
            mism.append("split sizes differ from reference snapshot")
        for s_key, c_key in (("train", "train"), ("validation", "validation"), ("test", "test")):
            if snap["label_counts"][s_key] != cov.counts[c_key]:
                mism.append(f"label counts differ for {s_key}")
        report["matches_reference_snapshot"] = not mism
        report["snapshot_mismatches"] = mism
    if a.check_leakage:
        report["leakage_check"] = leakage(dsd)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.DATASET_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"licence: {config.DATASET_LICENSE}  |  images: {report['total_images']}  | sizes: {cov.split_sizes}")
    print(f"train classes: {len(cov.train_classes)} | evaluable (val∩test): {len(cov.eval_classes)} | "
          f"unevaluated: {len(cov.unevaluated_classes)}")
    if cov.unknown_eval_classes:
        print("ERROR: eval classes missing from train:", cov.unknown_eval_classes)
    print("snapshot match:", report.get("matches_reference_snapshot", "n/a"))
    if "leakage_check" in report:
        print(json.dumps(report["leakage_check"], indent=2))
    print(f"wrote {config.DATASET_REPORT_PATH}")


if __name__ == "__main__":
    main()
