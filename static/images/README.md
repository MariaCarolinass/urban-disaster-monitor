# Figure and image credits

## Image folders

- [Branding](branding/) contains the project and institutional logos.
- [Dataset](dataset/) contains the synthetic flood example and examples of training image augmentation.
- [Evaluation](evaluation/) contains model comparison charts, validation curves, and inference benchmarks. The [evaluation guide](../../src/evaluation/README.md) explains how to regenerate the plots.
- [Interface](interface/) contains the Gradio screenshot.
- [Qualitative examples](qualitative/) contains validation mosaics, the external image and its predictions, and annotated rescue video frames.

## Sources

- [The synthetic flood example](dataset/synthetic-flood-example.jpg) was generated with Gemini 2.5 Flash Image for the project dataset.
- The validation mosaics use four dataset images. The annotated version displays YOLOv26m predictions at confidence 0.70. The Gradio screenshot shows a local run on a validation image at confidence 0.30. See the [dataset attribution notes](../../dataset/SOURCES.md) for known source projects and reuse terms.
- The external image is outside the annotated dataset. Its path and SHA-256 are recorded in the [external inference results](../../models/comparisons/external-inference.json). The source photo appeared in the [Jornal Gazeta Regional flood-rescue article](https://gazetaregional.com.br/enchente-do-rs-o-resgate-de-animais-requer-calma-e-cuidado/).
- The two annotated rescue video stills use frames 240 and 336, indexed from zero, from the [Santa Catarina Military Fire Brigade video](https://www.youtube.com/shorts/6PuaEPbOhJU).

Reuse terms for third-party photos and footage are governed by their sources.
