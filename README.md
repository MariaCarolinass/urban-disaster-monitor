<div align="center">

<img src="static/images/logo1.png" alt="Project logo" width="600"/><br>

[![YOLOv11](https://img.shields.io/badge/YOLOv11-ultralytics-green)](https://docs.ultralytics.com/models/yolov8/#yolov8-usage-examples) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-blue.svg)

[Dataset](./dataset) | [Models](./models) | [Notebooks](./notebooks) | [Kaggle](https://www.kaggle.com/datasets/mariacsoares/urban-disaster-object-detection-dataset/data) | [Roboflow](https://universe.roboflow.com/ufrnprojects-xlut9/urban-disaster-monitor) | [App](https://huggingface.co/spaces/carolinasoares/urban-disaster-monitor-v2)

[Português](./README-pt.md) | English

</div>

# Urban Disaster Monitor

### Detection and classification of civilians, animals and rescuers in urban disaster scenarios using YOLOv11

In urban disaster situations, every second counts. This project offers a computer vision tool to help rescue teams act with greater precision and speed.

<div align="center">

<img src="static/images/capa1.webp" alt="People in flood" width="700"/> 

</div><br>

## Folder Structure

```
.
├── app
│   ├── examples
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── notebooks
│   ├── coco-vs-yolo-comparison.ipynb
│   ├── generative-images-synthetic-gemini.ipynb
│   ├── metrics-and-comparison-yolo-models.ipynb
│   ├── simulation-video-yolo.ipynb
│   └── training-yolo-dataset.ipynb
├── dataset
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── valid
│   │   ├── images
│   │   └── labels
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
├── models
│   ├── yolov11l
│   ├── yolov11m
│   ├── yolov11n
│   └── yolov11s
└── static
    ├── gif
    ├── images
    └── video
```

## Table of Contents

- [About the project](#about-the-project)
  - [Features](#features)
  - [Institutional context](#institutional-context)
- [Dataset](#dataset)
  - [Pre-processing and augmentations](#pre-processing-and-augmentations)
  - [Class composition](#class-composition)
  - [Why a custom dataset?](#why-a-custom-dataset)
  - [Data acquisition](#data-acquisition)
    - [Public images](#public-images)
    - [Synthetic images (Gemini 2.5 Flash Image)](#synthetic-images-gemini-25-flash-image)
- [Annotations and Model](#annotations-and-model)
  - [Classes (Roboflow)](#classes-roboflow)
  - [Architecture](#architecture)
  - [Training pipeline](#training-pipeline)
  - [Training environment (Colab + T4 GPU)](#training-environment-colab--t4-gpu)
- [Metrics and results](#metrics-and-results)
  - [Results by variant](#results-by-variant-map05)
  - [Results by class](#results-by-class)
  - [Custom training vs. COCO pre-trained](#custom-training-vs-coco-pre-trained)
  - [Video simulation](#video-simulation)
- [Conclusions and recommendations by model](#conclusions-and-recommendations-by-model)
  - [Future work](#future-work)
- [Interactive Interface](#interactive-interface)
  - [Test online](#test-online)
  - [Features](#features)
  - [Technologies](#technologies)
  - [Run locally](#run-locally)
- [Team](#team)
- [License](#license)
- [References](#references)

## About the project

**Urban Disaster Monitor** is a computer vision system for detecting and classifying **individuals** (Civilians and Rescuers) and **animals** (Cows, Horses, Dogs and Cats) in urban disaster scenarios. In situations of structural collapse, floods or landslides, quickly identifying civilians and rescuers in real time is crucial to optimize resources and reduce fatalities. The system uses **YOLOv11** to support emergency teams during disaster response.

### Features

- Detection and classification of **Civilians**, **Rescuers**, **Cows**, **Horses**, **Dogs** and **Cats**
- **Metrics** and **bounding boxes** visualization
- Interactive **Gradio** interface for uploading and testing images and videos

### Institutional context

The project started during the Computer Vision course taught by **Prof. Dr. [Helton Maia](https://heltonmaia.com/)** at **ECT/UFRN**. It was developed voluntarily for **Smart Metropolis** (IMD/UFRN) as part of **Project SPICI (Integrated Public Security in Smart Cities)**, constituting an intelligent platform for real-time image collection, processing and analysis for crisis and disaster management.

<a href="https://smlab.imd.ufrn.br/"><img src="static/images/smartmetropolislab.png" alt="Smart Metropolis logo" width="150"/></a> <a href="https://smlab.imd.ufrn.br/projeto-spici/"><img src="static/images/spici.png" alt="SPICI logo" width="100"/></a> <a href="https://imd.ufrn.br/"><img src="static/images/imd-logo.7784f1db.webp" alt="IMD/UFRN logo" width="150"/></a> <a href="https://www.ect.ufrn.br/"><img src="static/images/cet.png" alt="ECT/UFRN logo" width="100"/></a>

## Dataset

The dataset was developed for **detection and counting of survivors** (civilians, rescuers and animals) in emergency scenarios, including floods, landslides, structural collapses and flooded rural areas. The initial set had 1,903 images; after pre-processing and augmentations, the final dataset totals **3,240 images**, distributed across **6 classes** selected based on real operational demands, particularly in the Brazilian context. The dataset follows a **multi-label** distribution (a single image can contain multiple classes simultaneously).

### Pre-processing and augmentations

- **Auto-orientation:** consistent image alignment
- **Resizing:** 640×640 px (stretch)
- **Contrast:** stretching to improve details in low-visibility scenes
- **Augmentations:** 2 variants generated per original sample
  - Rotation: ±15°
  - Shear: ±10° horizontal and vertical
  - Grayscale: 15% of images
  - Saturation: ±25%
  - Sparse noise: up to 0.89% of pixels

### Class composition

| Class | Annotated objects | Images containing the class |
|--------|------------------|---------------------------|
| civilian | 3,150 | 1,060 |
| rescuer | 1,074 | 397 |
| dog | 531 | 373 |
| cat | 637 | 207 |
| horse | 811 | 279 |
| cow | 749 | 180 |

*Annotated objects* indicates the total bounding boxes per class; *images containing the class* indicates in how many distinct images each class appears. The **rescuer** class was introduced to avoid alerts in areas occupied only by emergency teams, reducing false positives. The distinction criterion between rescuers and civilians is based on visual attributes such as helmets, reflective vests or red-yellow uniforms typical of rescue teams.

- **Annotation:** [Roboflow](https://universe.roboflow.com/ufrnprojects-xlut9/urban-disaster-monitor/)
- **Partitioning:** 83% train (2,674) / 12% validation (383) / 6% test (183) 
- **Partitioned dataset:** [dataset](./dataset)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

### Why a custom dataset?

Large-scale datasets such as **COCO** do not perform well in disaster scenarios. COCO does not distinguish *rescuer* from *civilian* and represents all as "person", which compromises alert systems—areas occupied exclusively by rescue teams would generate false positives. In addition, models pre-trained on COCO struggle in underrepresented scenarios, such as people or animals partially submerged in water. A domain-specific dataset was necessary.

### Data acquisition

The dataset combines **real-world images from public repositories** and **AI-generated synthetic images**, mitigating the scarcity of high-quality real data in disaster environments.

#### Public images

Collected from sources with open or shareable licenses:

- [Wikimedia Commons](https://commons.wikimedia.org/)
- [Flickr](https://www.flickr.com/)
- [Google Images](https://images.google.com)
- [Roboflow Universe](https://universe.roboflow.com/)

<div align="center">

<img src="static/images/230714-india-flooding-mb-0831-d3a66d.jpg" alt="People rescuing flooded cows" width="400"/> 
<img src="static/images/230714-india-flooding-mb-0831-d3a66d_annotated.webp" alt="People rescuing flooded cows - annotated" width="400"/> 

</div>

#### Synthetic images (Gemini 2.5 Flash Image)

**832 synthetic images** were generated with the [Gemini 2.5 Flash Image](https://developers.googleblog.com/introducing-gemini-2-5-flash-image/) model to reduce class imbalance and increase contextual variability, especially in animal classes. The prompts follow a documentary/photojournalistic style, with:

- **Authentic disaster scenarios:** flooded streets, inundated fields, submerged houses, animals in rescue or isolation situations.
- **Species diversity:** horses, cows, dogs and cats in different flood contexts.
- **Quality and resolution variation:** from low-resolution images to sharper photos, simulating real field records.
- **Visual details:** realistic proportions, visible expressions, full or partially submerged bodies, debris, overcast sky, natural lighting.
- **Varied situations:** animals alone, in groups, being rescued, trapped on rooftops, balconies, fences or floating objects.
- **Urban and rural environments:** streets, houses, pastures and farms affected by floods.

Each image was manually reviewed. The synthetic samples complement the real ones and expand the intra-class variability of *dog*, *cat*, *cow* and *horse*.

<div align="center">

<img src="static/images/gemini.jpg" alt="Generated image" width="400"/>
<img src="static/images/gemini_result.webp" alt="Generated image classified" width="400"/>

</div><br>

Example prompts:

1. "Authentic documentary photo of a brown horse standing in a flooded street after heavy rain, realistic proportions, clear body visible, imperfect disaster photo"

2. "Documentary style photo of a white horse trapped in a pasture with floodwater, overcast sky, realistic proportions, authentic disaster scene"

3. "Low-resolution documentary photo of two wet cats stranded on a rooftop surrounded by floodwater, realistic proportions, visible faces, authentic urban flood event"

4. "Low quality documentary style photo of a stray dog on a small wooden raft during flood, rescue scene, realistic body intact"

5. "Documentary flood photo of cows standing in a partially submerged field, water up to their legs, cloudy sky, authentic proportions"

6. "Realistic low-resolution documentary photo of a cow isolated in floodwater reflecting sunset colors, authentic disaster photo"

[Synthetic image generation notebook](./notebooks/generative-images-synthetic-gemini.ipynb)

## Annotations and Model

Annotations were performed on the **Roboflow** platform and converted to YOLO format. The YOLOv11 model is trained on the six classes defined below, with continuous validation and evaluation on an independent test set.

### Classes (Roboflow)

| Class | Description |
|--------|-----------|
| `civilian` | Individuals not identified as rescue teams |
| `rescuer` | Emergency teams with uniforms, PPE (helmets, vests) |
| `cat`, `dog` | Domestic animals in urban scenarios |
| `cow`, `horse` | Large animals in rural/peripheral urban areas |

### Architecture

The model uses **YOLO** (You Only Look Once), a reference in real-time object detection: it formulates the task as a single regression problem that predicts bounding boxes and class probabilities directly from the image, enabling end-to-end optimization and high inference speed.

**YOLOv11** (Ultralytics) is the latest version, with compatibility for conversion between frameworks. Trained variants: YOLOv11n (nano), YOLOv11s (small), YOLOv11m (medium) and YOLOv11l (large).

### Training pipeline

1. **Conversion:** Roboflow annotations to YOLO format (`.txt`)
2. **Iterative training:** parameters above; Adam or SGD optimizers
3. **Continuous validation:** real-time metrics, *overfitting* detection, early stopping
4. **Final evaluation:** independent test set to verify generalization

### Training environment (Colab + T4 GPU)

Training was performed on **Google Colab** using **NVIDIA T4** GPUs. Example call with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')
model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolov11n'
)
```

[Training notebook](./notebooks/training-yolo-dataset.ipynb)

## Metrics and results

Model evaluation was performed with standard object detection metrics: **mAP@0.5** (mean average precision at IoU 0.5), **mAP@0.5:0.95** (multiple IoU thresholds, more stringent), **Precision**, **Recall** and **confusion matrix** for per-class error analysis.

- [Model comparison notebook](./notebooks/metrics-and-comparison-yolo-models.ipynb)
- [Model vs. COCO comparison notebook](./notebooks/coco-vs-yolo-comparison.ipynb)
- [Trained model results](./models)

### Results by variant (mAP@0.5)

The four trained variants (nano, small, medium, large) show similar performance: YOLOv11n reached 85.27%, YOLOv11s 86.88%, YOLOv11m 86.18% and YOLOv11l 86.12%. The gap between the best (small) and worst (nano) did not exceed 1.61%, indicating that even the most compact version maintains good accuracy, which is relevant for hardware-constrained scenarios.

<div align="center">
<img src="static/images/metricas0.5.png" alt="mAP@0.5 by variant" width="700"/>
</div>

### Results by class

The chart below compares mAP@0.5 per class across the different variants. The `dog` class had the lowest performance overall but remained above 70% with the small model. Person classes (`civilian`, `rescuer`) and large animals (`cow`, `horse`) tend to perform better, possibly due to greater visibility and area occupied in the image. 

<div align="center">
<img src="static/images/metricas-classes.png" alt="mAP@0.5 by class" width="900"/>
</div>

### Custom training vs. COCO pre-trained

To validate the impact of domain-specific training, **custom YOLOv11m** was compared with the **COCO pre-trained** version. COCO evaluation was restricted to the 4 animal classes in our dataset plus one aggregated *person* class (civilian + rescuer):

| Metric | Custom | COCO pre-trained |
|---------|-------------|-------------------|
| Precision | 0.87 | 0.87 |
| Recall | 0.77 | 0.71 |
| mAP@0.5 | 0.86 | 0.80 |
| mAP@0.5–0.95 | 0.52 | 0.42 |

The custom model outperforms the pre-trained one in recall and in both mAP metrics, highlighting gains in localization quality (stricter IoU) and in detecting people and animals in flood scenarios. Qualitative analysis points to bottlenecks such as confusion between `civilian` and `background` and difficulty with smaller animals (`dog`, `cat`).

<div align="center">
<img src="static/images/metricas-yolo.png" alt="Custom model result" width="700"/>
</div>

*Qualitative comparison:* left, output of the COCO pre-trained model; right, custom model. Domain-specific training improves detections in disaster scenes.

<div align="center">
<img src="static/images/modelo-coco.png" alt="COCO model" width="400"/>
<img src="static/images/modelo-customizado.png" alt="Custom model" width="400"/>
</div>

### Video simulation

A [public YouTube video](https://www.youtube.com/watch?v=QnFwDqzCwRU) was used to simulate a real urban disaster scenario, with scenes of firefighters (`rescuer`) in flood areas. The video also contains an animal (goat) not included in the model's classes and therefore not identified.

**YOLOv11m** was applied to the video with a minimum confidence of **0.75**, to evaluate detection and classification of individuals in motion under different lighting conditions and camera angles.

<div align="center">
<img src="static/gif/rescuer.gif" alt="Training video example" width="600"/>
</div><br>

**Results:** Performance was satisfactory; most `rescuers` were correctly identified, demonstrating the effectiveness of Urban Disaster Monitor in dynamic scenarios. At times the model did not detect all rescuers, possibly due to partial occlusions or unfavorable angles. Occasional false positives were observed (e.g., object classified as `dog` with confidence up to 0.8), indicating the need to diversify the dataset and refine hyperparameters for greater robustness in video.

[Video simulation notebook](./notebooks/simulation-video-yolo.ipynb)

## Conclusions and recommendations by model

- **YOLOv11n:** Good performance with lower computational cost; suitable for drones and edge devices with resource constraints. Priority on speed and energy efficiency.
- **YOLOv11s:** Significant improvement over Nano, especially in learning stability and F1 score (~0.83). Inference times suitable for real-time applications on drones and smart urban cameras.
- **YOLOv11m:** Superior metrics and better generalization; recommended for production and critical applications requiring high precision.
- **YOLOv11l:** Maximum precision among variants; compatible with environments that have robust infrastructure.

The analysis reinforces the role of augmentations, data balancing and temporal modeling to increase robustness. For extended training and more complex experiments, dedicated GPU infrastructure is recommended (e.g., Google Colab Pro, NVIDIA A100).

### Future work

- **Temporal modeling:** ConvLSTM and Transformers for video stability
- **Embedded:** optimization on drones and urban cameras for real-time detection
- **New classes:** debris, rescue vehicles, barriers
- **Diversified dataset:** night scenes, low visibility, smoke, rain
- **More epochs and hyperparameters:** extended training in a dedicated environment; fine-tuning for video
- **Continual learning:** pipeline for adaptation to new scenarios and classes

## Interactive Interface

The interface was built with **Gradio** to allow testing the trained models without setting up a training environment. You can upload images or videos and view detections (civilians, rescuers and animals) with _bounding boxes_ and labels in real time.

### Test online

The application is available on **Hugging Face Spaces**. No installation required to try it:

**👉 [Open Urban Disaster Monitor](https://huggingface.co/spaces/carolinasoares/urban-disaster-monitor-v2)**

<div align="center">
<img src="static/images/app_gradio.png" alt="App interface on Gradio" width="900"/>
</div>

*Interface hosted on [Hugging Face](https://huggingface.co).*

### Features

- **Model:** choose among YOLOv11n, YOLOv11s, YOLOv11m or YOLOv11l
- **Upload:** images or videos
- **Confidence:** adjust the _confidence threshold_ to filter detections
- **Visualization:** _bounding boxes_ and labels by class (civilian, rescuer, dog, cat, horse, cow)
- **Download:** processed images or videos with detections

### Technologies

[Python](https://www.python.org/) · [Gradio](https://gradio.app/) · [Ultralytics YOLO](https://docs.ultralytics.com) · [PyTorch](https://pytorch.org/) · [OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/)

### Run locally

Clone the repository:

```bash
git clone https://github.com/MariaCarolinass/urban-disaster-monitor.git
```

Navigate to the `app` folder:

```bash
cd urban-disaster-monitor/app
```

Create a virtual environment with venv:

```bash
python3 -m venv venv
```

Activate the virtual environment (Linux/macOS or Windows):

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python app.py
```

## Team

| [![](https://github.com/jagaldino.png?size=80)](https://github.com/jagaldino) | [![](https://github.com/MariaCarolinass.png?size=80)](https://github.com/MariaCarolinass) |
| :---------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|                           **João Galdino**                           |                             **Carolina Soares**                | 

## License

- Code: MIT License
- Dataset: CC BY 4.0 (see dataset source for details)

## References

- T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, C. L. Zitnick, Microsoft COCO: Common Objects in Context, in: D. Fleet, T. Pajdla, B. Schiele, T. Tuytelaars (Eds.), Computer Vision – ECCV 2014, Vol. 8693 of Lecture Notes in Computer Science, Springer Cham, Switzerland, 2014, pp. 740–755. [doi:10.1007/978-3-319-10602-1_48](https://doi.org/10.1007/978-3-319-10602-1_48).

- J. Redmon, S. Divvala, R. Girshick, A. Farhadi, You Only Look Once: Unified, real-time object detection, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, USA, 2016, pp. 779–788. [doi:10.1109/CVPR.2016.91](https://doi.org/10.1109/CVPR.2016.91).

- M. S. Z. Pattankudi, S. Uppin, A. R. Attar, K. Bhoomaraddi, R. Kolhar, S. Varur, Human detection in disaster scenarios for enhanced emergency response using YOLO11, in: Proceedings of the 3rd International Conference on Futuristic Technology, Vol. 2, SciTePress, Portugal, 2025, pp. 739–746. [doi:10.5220/0013601000004664](https://doi.org/10.5220/0013601000004664).

- D. H. Sai, K. Vidhya, K. A. Jenefa, C. R. Joy, T. M. Thiyagu, S. S. Kirubakaran, Enhancing emergency response with real-time video analytics for natural disaster management, in: 2025 Fourth International Conference on Smart Technologies, Communication and Robotics (STCR), IEEE, USA, 2025. [doi:10.1109/stcr62650.2025.11018905](https://doi.org/10.1109/stcr62650.2025.11018905).

- S. S. Vibhuti, S. Sutar, B. Marigoudar, A. Gopal, S. Varur, C. Muttal, YOLO11: Flood victim detection and rescue alert system, in: Proceedings of the 3rd International Conference on Futuristic Technology, Vol. 2, SciTePress, Portugal, 2025, pp. 804–811. [doi:10.5220/0013603000004664](https://doi.org/10.5220/0013603000004664).

- B. V. B. Prabhu, R. Lakshmi, R. Ankitha, M. S. Prateeksha, N. C. Priya, RescueNet: YOLO-based object detection model for detection and counting of flood survivors, Modeling Earth Systems and Environment 8 (4) (2022) 4509–4516. [doi:10.1007/s40808-022-01414-6](https://doi.org/10.1007/s40808-022-01414-6).
