"""Evaluate COCO and custom YOLO26m on one contiguous five-class taxonomy.

Merge class scores BEFORE the shared NMS, then use one DetectionValidator.
Historical results stay separate; output directories can be chosen by the caller.
"""

import argparse
import hashlib
import json
import math
import platform
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import nms
from ultralytics.utils.metrics import ConfusionMatrix

from .paths import ROOT
COMMON_NAMES = {0: "person", 1: "cat", 2: "dog", 3: "horse", 4: "cow"}
ALIASES = {"civilian": "person", "rescuer": "person"}
CUSTOM = ROOT / "models/yolov26m/weights/best.pt"
PRETRAINED = ROOT / "notebooks/generated/weights/yolo26m.pt"
RESULT = ROOT / "models/comparisons/coco-vs-custom-yolo26m-common-val.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_scores(raw, source_names):
    """Map class scores using max (not sum) for civilian/rescuer -> person."""
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    if raw.ndim != 3 or raw.shape[1] != 4 + len(source_names):
        raise ValueError("Expected native one-to-many YOLO output [B, 4+nc, N]; use nms=True")
    groups = {name: [] for name in COMMON_NAMES.values()}
    for class_id, name in source_names.items():
        name = ALIASES.get(name, name)
        if name in groups:
            groups[name].append(class_id)
    if any(not ids for ids in groups.values()):
        raise ValueError(f"Checkpoint cannot represent all five common classes: {source_names}")
    scores = [raw[:, [4 + i for i in groups[name]], :].amax(dim=1, keepdim=True)
              for name in COMMON_NAMES.values()]
    return torch.cat([raw[:, :4, :], *scores], dim=1)


class CommonTaxonomyValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        if self.end2end:
            raise ValueError("Use the one-to-many head with nms=True for both checkpoints")
        self.source_names = dict(model.names)
        self.names = dict(COMMON_NAMES)
        self.nc = len(self.names)
        self.class_map = list(range(1, self.nc + 1))
        self.metrics.names = dict(self.names)
        self.confusion_matrix = ConfusionMatrix(names=self.names)
        self.fixed_counts = {"tp": 0, "fp": 0, "fn": 0}

    def postprocess(self, preds):
        outputs = nms.non_max_suppression(
            common_scores(preds, self.source_names), conf_thres=self.args.conf,
            iou_thres=self.args.iou, multi_label=True, agnostic=False,
            max_det=self.args.max_det, nc=5, end2end=False, max_time_img=10.0,
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
                for x in outputs]

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        # Recompute matching AFTER applying the common operating threshold.
        for index, pred in enumerate(preds):
            filtered = {key: values[pred["conf"] >= 0.25] for key, values in pred.items()}
            target = self._prepare_batch(index, batch)
            tp = int(self._process_batch(filtered, target)["tp"][:, 0].sum())
            self.fixed_counts["tp"] += tp
            self.fixed_counts["fp"] += len(filtered["cls"]) - tp
            self.fixed_counts["fn"] += len(target["cls"]) - tp


def prepare_dataset(split="val", limit=None):
    source_split = {"val": "valid", "test": "test"}[split]
    source = ROOT / "dataset" / source_split
    config = yaml.safe_load((ROOT / "dataset/data.yaml").read_text())
    names = config["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names
    reverse = {name: i for i, name in COMMON_NAMES.items()}
    mapping = {int(i): reverse[ALIASES.get(name, name)] for i, name in names.items()}
    images = sorted(p for p in (source / "images").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        images = images[:limit]
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError("Empty image set or ambiguous label filenames")
    generated = ROOT / "notebooks/generated/coco_common_5class"
    generated.mkdir(parents=True, exist_ok=True)
    destination = generated / (split + (f"_smoke_{limit}" if limit else "_full"))
    manifest, counts = [], Counter()
    with tempfile.TemporaryDirectory(dir=generated) as staging:
        staging = Path(staging)
        (staging / "images").mkdir()
        (staging / "labels").mkdir()
        for image in images:
            label = source / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing annotation, cannot assume background: {label}")
            mapped = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Expected YOLO detection annotations: {label}")
                class_id = int(parts[0])
                coordinates = list(map(float, parts[1:]))
                if class_id not in mapping or any(not math.isfinite(x) or not 0 <= x <= 1 for x in coordinates):
                    raise ValueError(f"Invalid annotation: {label}: {line}")
                if min(coordinates[2:]) <= 0:
                    raise ValueError(f"Empty bounding box: {label}")
                mapped.append(f"{mapping[class_id]} {' '.join(parts[1:])}")
                counts[COMMON_NAMES[mapping[class_id]]] += 1
            if len(mapped) != len(set(mapped)):
                raise ValueError(f"Duplicate ground-truth boxes after class merging; review {label}")
            mapped_text = "\n".join(mapped) + ("\n" if mapped else "")
            (staging / "labels" / label.name).write_text(mapped_text)
            shutil.copy2(image, staging / "images" / image.name)
            manifest.append({"image": str(image.relative_to(ROOT)), "image_sha256": sha256(image),
                             "source_label_sha256": sha256(label),
                             "mapped_label_sha256": hashlib.sha256(mapped_text.encode()).hexdigest(),
                             "boxes": len(mapped)})
        # The validator requires train/val keys even for evaluation-only data.
        data = {"path": str(destination), "train": None, "val": "images",
                "names": COMMON_NAMES, "nc": 5}
        if split == "test":
            data["test"] = "images"
        (staging / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        if destination.exists():
            shutil.rmtree(destination)  # Only this helper's generated copy.
        shutil.copytree(staging, destination)
    return destination / "data.yaml", {"split": split, "image_count": len(images),
            "annotation_count": sum(counts.values()), "annotations_per_class": dict(counts),
            "images": manifest, "smoke_test": limit is not None}


def run(split="val", device="cpu", batch=8, limit=None):
    for checkpoint in (PRETRAINED, CUSTOM):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Supply the checkpoint locally before evaluation: {checkpoint}")
    models = {"coco_pretrained": YOLO(str(PRETRAINED)), "custom": YOLO(str(CUSTOM))}
    # Same scale and layer specification; only the output class count differs.
    configs = [model.model.yaml for model in models.values()]
    for key in ("scale", "backbone", "head", "reg_max", "channels"):
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"Different model architectures: {key}")
    torch.set_num_threads(8)
    data_path, manifest = prepare_dataset(split, limit)
    args = dict(data=str(data_path), split=split, imgsz=640, batch=batch, device=device,
                workers=0, conf=0.001, iou=0.7, max_det=300, rect=False, nms=True,
                augment=False, quantize=32, channels_last=False, plots=False,
                save_json=False, save_txt=False, single_cls=False, agnostic_nms=False,
                classes=None, project=str(ROOT / "notebooks/generated/common-taxonomy-runs"),
                exist_ok=True, verbose=False)
    results = {}
    for name, model in models.items():
        path = PRETRAINED if name == "coco_pretrained" else CUSTOM
        holder = {}

        def make_validator(**kwargs):
            holder["validator"] = CommonTaxonomyValidator(**kwargs)
            return holder["validator"]

        metrics = model.val(validator=make_validator, name=name, **args)
        validator = holder["validator"]
        assert validator.seen == manifest["image_count"]
        counts = validator.fixed_counts
        assert counts["tp"] + counts["fn"] == manifest["annotation_count"]
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        results[name] = {"precision_macro_best_f1": float(metrics.box.mp),
                         "recall_macro_best_f1": float(metrics.box.mr),
                         "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                         "ap50_per_class": {COMMON_NAMES[int(i)]: float(ap)
                                            for i, ap in zip(metrics.box.ap_class_index, metrics.box.all_ap[:, 0])},
                         "ap50_95_per_class": {COMMON_NAMES[int(i)]: float(ap)
                                               for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap)},
                         "fixed_operating_point": {"confidence": 0.25, "matching_iou": 0.5,
                                                   "aggregation": "micro", "precision": precision,
                                                   "recall": recall, **counts},
                         "checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": sha256(path)}
    payload = {"protocol": "common five-class evaluation; max class-score merge before shared class-aware NMS",
               "measured_at_utc": datetime.now(timezone.utc).isoformat(),
               "ultralytics_version": ultralytics_version, "torch_version": torch.__version__,
               "python_version": platform.python_version(), "platform": platform.platform(),
               "torch_num_threads": torch.get_num_threads(), "class_names": COMMON_NAMES,
               "parameters": {k: v for k, v in args.items() if k not in {"project", "data"}},
               "dataset": manifest, "models": results}
    # Smoke results have their own name and cannot replace a complete comparison.
    output = RESULT.with_name(f"coco-vs-custom-yolo26m-common-{split}" + ("-smoke.json" if limit else ".json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("Saved:", output, flush=True)
    if limit is None:
        plot_comparison(payload)
    return payload


def plot_comparison(payload, output_dir=None, filename=None):
    """Plot only a complete common-taxonomy run, never a legacy/smoke cache."""
    if (not payload.get("dataset") or payload["dataset"].get("smoke_test")
            or len(payload.get("class_names", {})) != 5):
        raise ValueError("Use a complete five-class evaluation to export comparison figures")
    import matplotlib.pyplot as plt
    import numpy as np
    from .plotting.style import publication_style as style, COLORS

    style()
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(4)
    for index, (key, label) in enumerate((("coco_pretrained", "COCO-pretrained YOLO26m"),
                                          ("custom", "Custom YOLO26m"))):
        row = payload["models"][key]
        values = [row["fixed_operating_point"]["precision"], row["fixed_operating_point"]["recall"],
                  row["map50"], row["map50_95"]]
        bars = ax.bar(positions + (index - 0.5) * 0.36, values, width=0.36,
                      color=COLORS[index], label=label)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=9)
    ax.set_xticks(positions, ["Micro precision\n(conf. 0.25)", "Micro recall\n(conf. 0.25)",
                             "mAP@0.5", "mAP@0.5:0.95"])
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
    ax.grid(axis="y", color="#D9DEE4", alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    folder = Path(output_dir) if output_dir is not None else ROOT / "notebooks/generated/figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (filename or f"coco-vs-custom-yolo26m-common-{payload['dataset']['split']}")
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=300, bbox_inches="tight", pad_inches=0.12)
    print("Figure:", path, flush=True)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int)
    options = parser.parse_args()
    run(**vars(options))
