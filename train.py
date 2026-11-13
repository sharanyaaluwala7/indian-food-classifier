#!/usr/bin/env python
"""Train a food classifier on SohlHealth/enhanced-indian-food-classification.

    python train.py --arch mobilenetv3_large_100 --mode linear_probe --epochs 8
    python train.py --arch efficientnet_b0 --mode finetune --epochs 12

Class-coverage policy (train has 43 classes, validation/test only 22):
  * the model has 43 outputs, fixed from the train split;
  * model selection uses an internal stratified holdout cut from train, which
    covers all 43 classes;
  * validation and test are scored only on the 22 classes they contain, in two
    clearly separated ways (open-set over 43 logits, closed-set over 22 logits);
  * the 21 classes absent from validation/test are listed as `unevaluated_on_official_splits`;
    their only measurement is the internal holdout.
The test split is used exactly once, after the best epoch has been chosen.
Every number written to metrics.json is computed from this run's model outputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np

from foodclf import config
from foodclf import data as D
from foodclf import evaluation as E
from foodclf import modelzoo as Z


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="mobilenetv3_large_100", choices=sorted(Z.ARCHS))
    p.add_argument("--mode", default="linear_probe", choices=["linear_probe", "finetune"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 linear_probe, 2e-4 finetune")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=4, help="early-stop after N epochs without holdout macro-F1 gain")
    p.add_argument("--class-weighting", default="sqrt_inv", choices=["none", "sqrt_inv"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true", help="random init (only for tests)")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="debug: train on a random subset; the run is flagged as partial in metadata")
    p.add_argument("--data-source", default="hub", choices=["hub", "local"])
    p.add_argument("--local-dataset-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--models-dir", default=str(config.MODELS_DIR))
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def main(argv=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    img_size = args.img_size or Z.ARCHS[args.arch]["img_size"]
    lr = args.lr or (1e-3 if args.mode == "linear_probe" else 2e-4)

    # ---------------------------------------------------------------- data
    dsd = D.load_dataset_dict(args.data_source, args.local_dataset_dir, args.cache_dir)
    cov = D.analyze_coverage(dsd)
    D.require_valid_coverage(cov)
    class_names = cov.train_classes
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    print(f"train classes: {len(class_names)} | evaluable on val/test: {len(cov.eval_classes)} | "
          f"unevaluated on official splits: {len(cov.unevaluated_classes)}")

    train_ds = dsd[config.TRAIN_SPLIT]
    fit_idx, hold_idx = D.stratified_holdout(D.label_strings(train_ds), args.holdout_frac, args.seed)
    partial = False
    if args.max_train_samples and args.max_train_samples < len(fit_idx):
        rng = np.random.default_rng(args.seed)
        fit_idx = sorted(rng.choice(fit_idx, args.max_train_samples, replace=False).tolist())
        partial = True

    tf_train, tf_eval = D.build_transforms(img_size, True), D.build_transforms(img_size, False)
    fit = D.HFImageDataset(train_ds, class_to_idx, tf_train, fit_idx)
    hold = D.HFImageDataset(train_ds, class_to_idx, tf_eval, hold_idx)
    val = D.HFImageDataset(dsd[config.VAL_SPLIT], class_to_idx, tf_eval)
    test = D.HFImageDataset(dsd[config.TEST_SPLIT], class_to_idx, tf_eval)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    fit_loader, hold_loader, val_loader, test_loader = mk(fit, True), mk(hold, False), mk(val, False), mk(test, False)
    print(f"fit={len(fit)} holdout={len(hold)} val={len(val)} test={len(test)} device={device}")

    # ---------------------------------------------------------------- model
    model = Z.build_model(args.arch, len(class_names), pretrained=not args.no_pretrained).to(device)
    if args.mode == "linear_probe":
        for p_ in model.parameters():
            p_.requires_grad = False
        for p_ in model.get_classifier().parameters():
            p_.requires_grad = True
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    counts = np.bincount(fit.targets, minlength=len(class_names)).astype(float)
    w = np.ones(len(class_names))
    if args.class_weighting == "sqrt_inv":
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        w = w / w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device), label_smoothing=0.1)

    # ---------------------------------------------------------------- train
    all_ids = list(range(len(class_names)))
    best_f1, best_state, best_epoch, bad, history = -1.0, None, -1, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.mode == "linear_probe":
            model.eval()  # keep frozen BN statistics
            model.get_classifier().train()
        run_loss, n = 0.0, 0
        for x, y in fit_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        sched.step()
        logits, yh = E.predict_logits(model, hold_loader, device)
        sc = E.score(logits, yh, class_names, class_names, restrict_to_eval=False)
        history.append({"epoch": epoch, "train_loss": run_loss / n, "holdout_accuracy": sc["accuracy"],
                        "holdout_macro_f1": sc["macro_f1"], "elapsed_s": round(time.time() - t0, 1)})
        print(history[-1])
        if sc["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = sc["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)

    # ---------------------------------------------------------------- final evaluation (real outputs only)
    holdout_metrics = E.evaluate_split(model, hold_loader, device, class_names, class_names, "internal_holdout")
    val_metrics = E.evaluate_split(model, val_loader, device, class_names, cov.eval_classes, config.VAL_SPLIT)
    test_metrics = E.evaluate_split(model, test_loader, device, class_names, cov.eval_classes, config.TEST_SPLIT)

    run_id = args.run_name or f"{args.arch}-{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    out = __import__("pathlib").Path(args.models_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id, "arch": args.arch, "mode": args.mode, "img_size": img_size,
        "mean": list(config.IMAGENET_MEAN), "std": list(config.IMAGENET_STD),
        "class_names": class_names, "eval_class_names": cov.eval_classes,
        "unevaluated_on_official_splits": cov.unevaluated_classes,
    }
    Z.save_checkpoint(out / "model.pt", model.state_dict(), meta)

    import sklearn, timm
    metrics = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"id": config.DATASET_ID, "license": config.DATASET_LICENSE, "source": args.data_source,
                    "split_sizes": cov.split_sizes},
        "class_coverage": {
            "train_classes": len(class_names),
            "evaluable_on_official_val_test": cov.eval_classes,
            "unevaluated_on_official_splits": cov.unevaluated_classes,
            "policy": ("Official validation/test contain only the evaluable classes. Metrics on them average over "
                       "those classes only. The unevaluated classes are measured only on the internal holdout, "
                       "which is carved from train and may be optimistic if train contains near-duplicates."),
        },
        "training": {"arch": args.arch, "mode": args.mode, "epochs_run": len(history), "best_epoch": best_epoch,
                     "lr": lr, "batch_size": args.batch_size, "weight_decay": args.weight_decay,
                     "class_weighting": args.class_weighting, "holdout_frac": args.holdout_frac, "seed": args.seed,
                     "n_fit_images": len(fit), "n_holdout_images": len(hold),
                     "partial_training_subset": partial, "pretrained": not args.no_pretrained,
                     "selection_criterion": "internal holdout macro-F1 over all 43 classes",
                     "device": str(device), "train_seconds": round(time.time() - t0, 1)},
        "history": history,
        "internal_holdout": holdout_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                        "sklearn": sklearn.__version__},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    t_open, t_closed = test_metrics["open_set_all_classes"], test_metrics["closed_set_split_classes"]
    Z.add_to_registry({
        "id": run_id, "arch": args.arch, "mode": args.mode,
        "weights": f"{run_id}/model.pt", "metrics": f"{run_id}/metrics.json",
        "sha256": Z.sha256_file(out / "model.pt"),
        "created_at": metrics["created_at"],
        "dataset_id": config.DATASET_ID, "data_source": args.data_source,
        "partial_training_subset": partial, "pretrained": not args.no_pretrained,
        "headline": {
            "internal_holdout_macro_f1_43": holdout_metrics["open_set_all_classes"]["macro_f1"],
            "test_accuracy_open_set": t_open["accuracy"],
            "test_macro_f1_open_set_22": t_open["macro_f1"],
            "test_accuracy_closed_set_22": t_closed["accuracy"],
        },
    }, __import__("pathlib").Path(args.models_dir) / "registry.json")

    print(f"\nsaved {out}\n  internal holdout macro-F1 (43 classes): {holdout_metrics['open_set_all_classes']['macro_f1']:.4f}"
          f"\n  test accuracy, open-set (22 scored classes, 43 logits): {t_open['accuracy']:.4f}"
          f"\n  test accuracy, closed-set (22 logits): {t_closed['accuracy']:.4f}")
    return run_id


if __name__ == "__main__":
    main()
