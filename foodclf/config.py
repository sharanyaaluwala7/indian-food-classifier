from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REGISTRY_PATH = MODELS_DIR / "registry.json"
DATASET_REPORT_PATH = REPORTS_DIR / "dataset_report.json"
DATASET_SNAPSHOT_PATH = REPORTS_DIR / "dataset_snapshot.json"

DATASET_ID = "SohlHealth/enhanced-indian-food-classification"
DATASET_LICENSE = "CC-BY-4.0"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
ATTRIBUTION = (
    "Dataset: SohlHealth/enhanced-indian-food-classification (Hugging Face), "
    "licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
    "Models in this project were trained on it; changes were made (resizing, "
    "augmentation, holdout split)."
)

TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Classes with fewer held-out images than this get a "low support" flag in reports.
LOW_SUPPORT_THRESHOLD = 10
