# Model evaluation

`src/evaluation/` contains the reusable code for evaluating Urban Disaster Monitor models, reading saved results, and generating figures. The [notebook guide](../../notebooks/README.md) covers interactive workflows and environment setup.

## Package layout

- [`paths.py`](paths.py) resolves the repository root and shared input and output paths. Set `UDM_PROJECT_ROOT` to use artifacts from another checkout when running a module.
- [`results.py`](results.py) reads YOLO and RF-DETR training logs and selects the epoch with the highest validation mAP@0.5. Historical YOLOv11 logs come from the `yolo11-v1.0.0` Git tag.
- [`checkpoint_validation.py`](checkpoint_validation.py) fills missing entries in `models/comparisons/checkpoint-validation.json` by validating the corresponding checkpoints. The cache includes per-class AP and checkpoint hashes.
- [`coco_comparison.py`](coco_comparison.py) evaluates the COCO-pretrained and custom YOLO26m checkpoints on the same images and five categories. Both models use the same scoring and postprocessing protocol. The evaluation combines civilians and rescuers into the `person` category.
- [`external_inference.py`](external_inference.py) measures inference latency and records detections for the external image and rescue video. The video benchmark uses `ffprobe` to read stream metadata.

### Figures

- [`plotting/style.py`](plotting/style.py) contains shared Matplotlib settings, palettes, number and axis formatting, and figure export helpers.
- [`plotting/comparison_figures.py`](plotting/comparison_figures.py) generates model comparison charts. It also provides the notebook functions for plotting global and per-class mAP and comparing YOLOv11 with YOLOv26.
- [`plotting/timing_figures.py`](plotting/timing_figures.py) plots YOLOv26 training durations and measured inference latencies. It checks the benchmark summary against the saved raw samples.
- [`plotting/yolo_curves.py`](plotting/yolo_curves.py) and [`plotting/rfdetr_curves.py`](plotting/rfdetr_curves.py) plot validation mAP across the recorded epochs.
- [`plotting/detection_figures.py`](plotting/detection_figures.py) previews validation images, runs YOLO26m inference, and exports original and annotated mosaics, an external image example, and video frame examples.

## Regenerate charts from saved results

Set up the environment using the notebook guide, then run any of these commands from the repository root. These commands read saved CSV and JSON artifacts. The comparison and YOLO curve modules also read the historical YOLOv11 Git tag.

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.plotting.comparison_figures
PYTHONPATH=src .venv/bin/python -m evaluation.plotting.timing_figures
PYTHONPATH=src .venv/bin/python -m evaluation.plotting.yolo_curves
PYTHONPATH=src .venv/bin/python -m evaluation.plotting.rfdetr_curves
```

Figures are written to `notebooks/generated/figures/` by default. To export elsewhere, pass an output directory to the module's `generate()` function.

## Run an evaluation

Use the validation split during development. Use the test split once the checkpoints and protocol are fixed. The following command runs inference and writes `models/comparisons/coco-vs-custom-yolo26m-common-test.json`.

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.coco_comparison --split test --batch 8
```

The shared five-class results measure person detection across both civilian and rescuer roles. The per-class checkpoint cache uses the project's six-class labels for its custom YOLO models. Use the shared protocol JSON for comparisons with the COCO-pretrained model.

The [detection examples notebook](../../notebooks/evaluation/qualitative/detection-examples.ipynb) provides a model-free preview before you run inference. Its generation step records the seed, selected image paths, confidence scores, and checkpoint hash in `models/comparisons/qualitative-mosaic.json`.
