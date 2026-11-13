#!/usr/bin/env python
"""Re-evaluate a saved model on the official validation and test splits.

    python evaluate.py --run-id <id>

Writes models/<id>/metrics_reeval.json. Useful to confirm the numbers in
metrics.json by recomputing them from the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from foodclf import config
from foodclf import data as D
from foodclf import evaluation as E
from foodclf import modelzoo as Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--models-dir", default=str(config.MODELS_DIR))
    ap.add_argument("--data-source", default="hub", choices=["hub", "local"])
    ap.add_argument("--local-dataset-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args()

    run = Path(a.models_dir) / a.run_id
    model, meta = Z.load_checkpoint(run / "model.pt", "cpu")
    dsd = D.load_dataset_dict(a.data_source, a.local_dataset_dir, a.cache_dir)
    cov = D.analyze_coverage(dsd)
    D.require_valid_coverage(cov)
    if cov.train_classes != meta["class_names"]:
        raise SystemExit("class list of the dataset differs from the one this model was trained with")
    c2i = {n: i for i, n in enumerate(meta["class_names"])}
    tf = D.build_transforms(meta["img_size"], False)
    out = {}
    for split in (config.VAL_SPLIT, config.TEST_SPLIT):
        loader = DataLoader(D.HFImageDataset(dsd[split], c2i, tf), batch_size=a.batch_size, num_workers=a.num_workers)
        out[split] = E.evaluate_split(model, loader, "cpu", meta["class_names"], cov.eval_classes, split)
    (run / "metrics_reeval.json").write_text(json.dumps(out, indent=2))
    for split, m in out.items():
        o, c = m["open_set_all_classes"], m["closed_set_split_classes"]
        print(f"{split}: open-set acc={o['accuracy']:.4f} macroF1={o['macro_f1']:.4f} | "
              f"closed-set acc={c['accuracy']:.4f} macroF1={c['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
