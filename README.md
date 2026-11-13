# AI-based food detection and classification

Classifies photos of Indian dishes using models trained on
[`SohlHealth/enhanced-indian-food-classification`](https://huggingface.co/datasets/SohlHealth/enhanced-indian-food-classification).
Includes dataset inspection, a training and evaluation pipeline, and a website.

**No model is included, and no accuracy figure is claimed anywhere in this repository.**
You train the models on your machine; the website then shows only numbers that
`train.py` computed from those models' real predictions.

## Run order

```bash
pip install -r requirements.txt

python inspect_dataset.py                  # writes reports/dataset_report.json
python inspect_dataset.py --check-leakage  # optional: exact/near-duplicate images across splits (slow, decodes all images)

python train.py --arch mobilenetv3_large_100 --mode linear_probe --epochs 8
python train.py --arch efficientnet_b0 --mode finetune --epochs 12   # a GPU is strongly recommended
python train.py --arch convnext_tiny   --mode finetune --epochs 10   # GPU recommended

python evaluate.py --run-id <run id printed by train.py>   # optional recomputation from the checkpoint
python app.py                                              # http://127.0.0.1:5000
```

Each `train.py` run writes `models/<run-id>/model.pt` and `metrics.json` and adds an entry to
`models/registry.json`. The website lists every valid entry and can compare them.
Linear-probe training is practical on a CPU; full fine-tuning is not.

## Dataset facts (checked 2026-10-11)

| | |
|---|---|
| Licence | **CC BY 4.0**: reuse, including commercial, requires attribution and a note of changes |
| Images | 15,404 = train 13,311 + validation 1,067 + test 1,026 |
| Classes | train 43; validation 22; test 22 (the same 22) |
| Without official val/test images | 21 classes |
| Smallest eval class | `modak`: 5 test images (21 validation) |

`reports/dataset_snapshot.json` stores the per-class counts read from the Hugging Face dataset API
on that date. `inspect_dataset.py` regenerates them from the real data and flags any difference.
The dataset card gives no author names, collection method or citation, so attribution
credits the Hugging Face dataset page. It also does not say whether the "enhanced" images include
augmented copies of originals. Run `--check-leakage` and read the result before trusting any number.

## How inconsistent class coverage is handled

- The model has **43 outputs**, fixed from the train split. Validation/test labels must be a subset of
  them; otherwise training stops.
- **Model selection** uses an internal stratified holdout cut from train (10% per class). It is the
  only held-out data covering all 43 classes. The test split is used once, after the best epoch is chosen.
- **Validation and test** are scored only on the 22 classes they contain, and macro metrics average over
  those 22, never over 43. Two variants are reported separately:
  - *open-set*: argmax over all 43 outputs; predicting a class the split cannot contain is an error;
  - *closed-set*: argmax over the 22 outputs only. Easier, so it is never the headline number.
- The 21 classes without validation/test images are listed in every `metrics.json` as
  `unevaluated_on_official_splits`; their only measurement is the internal holdout, which can be
  optimistic if train contains near-duplicates.
- Per-class results with fewer than 10 images are flagged `low_support`.
- The website shows, for each prediction, which kind of evidence exists for that dish.

## What the website refuses to do

- Without a valid trained model, `/api/predict` returns 503 and the page shows the training commands.
- A registry entry is rejected if its checksum does not match, it was not trained on the Hub dataset,
  it used a partial subset (`--max-train-samples`), or it used random weights (`--no-pretrained`).
- The model always names one of its 43 dishes. A confidence below 0.5 (`FOODCLF_UNCERTAIN_BELOW`) is shown as a warning,
  but this is not an out-of-distribution detector and non-food photos can still get high confidence.

## Tests

```bash
pip install pytest && python -m pytest -q tests
```

The tests train a tiny model on synthetic noise images to exercise the code paths (coverage handling,
metrics, registry, API rules). They write to temporary directories and their output is never a result.

## Attribution

Dataset: SohlHealth/enhanced-indian-food-classification (Hugging Face), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Models trained here are derived from it;
changes made: resizing, augmentation and an internal holdout split. Pretrained backbones come from
`timm` (ImageNet weights); check each backbone's own licence before redistributing a trained model.
