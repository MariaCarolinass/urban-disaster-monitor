# Notebooks and evaluation utilities

The notebooks are entry points for running experiments and inspecting results. Reusable Python code lives in `src/evaluation/`.

## Workflow

- `data-generation/`: [synthetic images with Gemini](data-generation/generative-images-synthetic-gemini.ipynb).
- `training/`: [YOLO training](training/training-yolo-dataset.ipynb) and [RF-DETR training](training/training-rfdetr-dataset.ipynb).
- `evaluation/accuracy/`: [YOLO variant metrics](evaluation/accuracy/metrics-and-comparison-yolo-models.ipynb), [YOLO11 vs YOLO26](evaluation/accuracy/yolo11-vs-yolo26-comparison.ipynb), and [COCO vs custom YOLO26m](evaluation/accuracy/coco-vs-yolo-comparison.ipynb).
- `evaluation/performance/`: [training and inference times](evaluation/performance/model-training-inference-times.ipynb).
- `evaluation/qualitative/`: [detection examples](evaluation/qualitative/detection-examples.ipynb), including a fast image lottery independent of inference.
- `video/`: [annotated video simulation](video/simulation-video-yolo.ipynb).

Each notebook locates the scientific project root from its working directory and adds `src/` to its import path. Open notebooks from the repository root or any nested notebook directory.

The notebooks use English for instructions and chart labels. Run the relevant cells to populate their outputs. The evaluation notebooks can regenerate comparison charts from saved results.

## Reusable code

See the [evaluation package guide](../src/evaluation/README.md) for module responsibilities and commands to regenerate figures from saved results.

- `src/evaluation/paths.py`: shared paths; `UDM_PROJECT_ROOT` optionally selects a different scientific checkout.
- `results.py`: reads preserved YOLO and RF-DETR training artifacts.
- `checkpoint_validation.py`: fills missing historical per-class validation cache entries. Its old COCO entry is historical and is not the final five-class comparison.
- `coco_comparison.py`: common-taxonomy COCO/custom evaluation, with identical scoring/NMS protocol.
- `external_inference.py`: measurements on the external image and rescue video; loading this module does not run inference.
- `plotting/style.py`: shared Matplotlib styles, palettes, and figure formatting.
- `plotting/comparison_figures.py`, `timing_figures.py`, `yolo_curves.py`, `rfdetr_curves.py`: recreate figures from saved artifacts.
- `plotting/detection_figures.py`: fast image selection, annotated mosaics and external examples.

Figure generators write to `notebooks/generated/figures/` by default. Use their `output_dir` argument to export elsewhere.

## Local setup

Use the single virtual environment at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r app/requirements.txt -r notebooks/requirements.txt
python -m ipykernel install --user --name urban-disaster-monitor --display-name "Python (Urban Disaster Monitor)"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. In VS Code or Jupyter, select the **Python (Urban Disaster Monitor)** kernel. Notebook dependency-install cells are optional with the environment prepared.

## Commands from the repository root

Run the common five-class evaluation (performs inference; defaults to validation unless `--split test` is given):

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.coco_comparison --split test --batch 8
```

Generate model comparison charts:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.plotting.comparison_figures
```

Other module commands follow the same pattern, for example `evaluation.external_inference`, `evaluation.plotting.yolo_curves`, and `evaluation.plotting.timing_figures`. These command examples use POSIX shell syntax.

## Inputs, outputs and interpretation

- `dataset/` and `models/` hold the scientific inputs; historical YOLO11 analysis also needs the preserved Git tags.
- `models/comparisons/` holds recorded metrics, measurements, hashes and manifests. The JSON files preserve measured values and input hashes for later analysis.
- COCO comparison uses `notebooks/generated/coco_common_5class/` and caches pretrained weights in `notebooks/generated/weights/`.
- `static/images/qualitative/external-disaster.jpg` is the input image for qualitative evaluation and inference timing. The rescue video remains in `static/video/`.
- `notebooks/generated/` is ignored by Git and holds derived datasets, figures and other generated artifacts. Gemini writes generated images there and requires `GOOGLE_API_KEY`.
- Standalone quick-chart cells use saved artifacts without repeating checkpoint evaluation or timing measurements.
- The final COCO notebook defaults to the test split after weights and protocol were fixed. It merges civilian/rescuer into person for both models, so its five-class metrics do not assess civilian/rescuer discrimination or directly match six-class validation results.
- The timing benchmark records its hardware, resolutions and sample count; comparisons between model families retain those methodological limits.
