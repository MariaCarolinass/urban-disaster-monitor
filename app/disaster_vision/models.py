"""Download and cache model instances."""

from functools import lru_cache

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .config import MODEL_CACHE_DIR, RFDETR_CHECKPOINT_FILE, RFDETR_REPO_MAP, YOLO_MODEL_MAP


def get_model_path(model_variant):
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    repo_id = YOLO_MODEL_MAP[model_variant]
    return hf_hub_download(
        repo_id=repo_id,
        filename=f"{model_variant}.pt",
        cache_dir=MODEL_CACHE_DIR,
    )


@lru_cache(maxsize=1)
def get_yolo_model(variant):
    return YOLO(get_model_path(variant))


@lru_cache(maxsize=1)
def get_rfdetr_model(variant="nano"):
    try:
        if variant == "nano":
            from rfdetr import RFDETRNano as model_class
        else:
            from rfdetr import RFDETRMedium as model_class
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed. Run `pip install -r app/requirements.txt` first."
        ) from exc

    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    weights_path = hf_hub_download(
        repo_id=RFDETR_REPO_MAP[variant],
        filename=RFDETR_CHECKPOINT_FILE,
        cache_dir=MODEL_CACHE_DIR,
    )
    return model_class(pretrain_weights=weights_path)
