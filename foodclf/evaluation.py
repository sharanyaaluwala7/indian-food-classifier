"""Metric computation. Everything here is computed from real model outputs;
nothing is estimated, defaulted or filled in."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import config


def predict_logits(model, loader, device):
    import torch

    model.eval()
    outs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            outs.append(model(x.to(device)).float().cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(outs), np.concatenate(ys)


def _prf(y_true, y_pred, label_ids: List[int]):
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    return p, r, f, s


def score(logits: np.ndarray, y_true: np.ndarray, class_names: List[str], eval_class_names: List[str],
          restrict_to_eval: bool) -> Dict:
    """Score predictions on a split whose labels lie in `eval_class_names`.

    restrict_to_eval=False: argmax over all model classes. Predicting a class that
        this split cannot contain counts as an error.
    restrict_to_eval=True: argmax over only the classes present in the split.
        This is the optimistic 'closed-set' number and is reported separately.
    Macro metrics average only over `eval_class_names`.
    """
    idx = {n: i for i, n in enumerate(class_names)}
    eval_ids = [idx[n] for n in eval_class_names]
    if restrict_to_eval:
        sub = logits[:, eval_ids]
        y_pred = np.asarray(eval_ids)[sub.argmax(1)]
    else:
        y_pred = logits.argmax(1)

    p, r, f, s = _prf(y_true, y_pred, eval_ids)
    in_scope = np.isin(y_pred, eval_ids)
    per_class = {
        n: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k]),
            "low_support": bool(s[k] < config.LOW_SUPPORT_THRESHOLD)}
        for k, n in enumerate(eval_class_names)
    }
    top5 = None
    if not restrict_to_eval and logits.shape[1] >= 5:
        top = np.argsort(-logits, axis=1)[:, :5]
        top5 = float((top == y_true[:, None]).any(1).mean())
    return {
        "n_images": int(len(y_true)),
        "n_classes_scored": len(eval_ids),
        "accuracy": float((y_pred == y_true).mean()),
        "top5_accuracy": top5,
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(np.average(f, weights=s)),
        "predicted_outside_scored_classes": float((~in_scope).mean()),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": eval_class_names,
            "matrix": _confusion(y_true, y_pred, eval_ids).tolist(),
            "note": "rows=true, cols=predicted; predictions outside the scored classes are excluded from columns",
        },
    }


def _confusion(y_true, y_pred, label_ids):
    pos = {l: i for i, l in enumerate(label_ids)}
    m = np.zeros((len(label_ids), len(label_ids)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in pos and p in pos:
            m[pos[t], pos[p]] += 1
    return m


def evaluate_split(model, loader, device, class_names, eval_class_names, split_name: str) -> Dict:
    logits, y = predict_logits(model, loader, device)
    return {
        "split": split_name,
        "open_set_all_classes": score(logits, y, class_names, eval_class_names, restrict_to_eval=False),
        "closed_set_split_classes": score(logits, y, class_names, eval_class_names, restrict_to_eval=True),
    }
