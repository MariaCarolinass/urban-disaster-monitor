"""Gradio entry point for the Urban Disaster Monitor Space."""

import gradio as gr

if __package__:
    from .disaster_vision.config import APP_DIR, MAX_UPLOAD_BYTES
    from .disaster_vision.examples import load_example_video
    from .disaster_vision.inference import image_detection, video_detection
else:
    from disaster_vision.config import APP_DIR, MAX_UPLOAD_BYTES
    from disaster_vision.examples import load_example_video
    from disaster_vision.inference import image_detection, video_detection


def _model_selectors():
    backend = gr.Radio(["YOLOv26", "RF-DETR"], label="Model family", value="YOLOv26")
    yolo_variant = gr.Dropdown(
        ["yolov26n", "yolov26s", "yolov26m", "yolov26l"],
        label="YOLOv26 variant",
        info="Used when the model family is YOLOv26.",
        value="yolov26m",
    )
    rfdetr_variant = gr.Dropdown(
        ["nano", "medium"],
        label="RF-DETR variant",
        info="Used when the model family is RF-DETR.",
        value="nano",
        visible=False,
    )
    backend.change(
        fn=lambda selected: (
            gr.update(visible=selected == "YOLOv26"),
            gr.update(visible=selected != "YOLOv26"),
        ),
        inputs=backend,
        outputs=[yolo_variant, rfdetr_variant],
    )
    return backend, yolo_variant, rfdetr_variant


with gr.Blocks(delete_cache=(3600, 86400)) as app:
    gr.HTML("""
        <h1 style='text-align: center'>Urban Disaster Monitor</h1>
        <h2 style='text-align: center'><a href='https://github.com/MariaCarolinass/urban-disaster-monitor' target='_blank'>GitHub</a></h2>
    """)

    with gr.Tab("Image"):
        gr.Markdown("JPEG, PNG, or WebP; up to 10 MiB and 12 megapixels.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Upload an Image", type="filepath", image_mode=None)
                image_conf_threshold = gr.Slider(
                    label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.30
                )
                image_backend, image_yolo_variant, image_rfdetr_variant = _model_selectors()
                image_button = gr.Button("Process Image", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Processed Image")
                image_predictions = gr.Textbox(
                    label="Predictions", placeholder="Predictions will appear here..."
                )

        image_button.click(
            fn=image_detection,
            inputs=[image, image_conf_threshold, image_backend, image_yolo_variant, image_rfdetr_variant],
            outputs=[output_image, image_predictions],
            concurrency_id="model_inference",
            concurrency_limit=1,
        )

        gr.Examples(
            examples=[[str(path)] for path in sorted((APP_DIR / "examples").glob("*"))
                      if path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            inputs=image,
            label="Example Images",
        )

    with gr.Tab("Video"):
        gr.Markdown("Up to 50 MiB, 1280×720 (either orientation), 60 seconds, 60 FPS, and 1,800 frames.")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Upload a Video", autoplay=True)
                video_conf_threshold = gr.Slider(
                    label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.30
                )
                video_backend, video_yolo_variant, video_rfdetr_variant = _model_selectors()
                video_button = gr.Button("Process Video", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="Processed Video", autoplay=True)
                video_predictions = gr.Textbox(
                    label="Predictions", placeholder="Predictions will appear here..."
                )

        video_button.click(
            fn=video_detection,
            inputs=[video, video_conf_threshold, video_backend, video_yolo_variant, video_rfdetr_variant],
            outputs=[output_video, video_predictions],
            concurrency_id="model_inference",
            concurrency_limit=1,
        )

        with gr.Row():
            rescue_example = gr.Button("Load rescue simulation")
        rescue_example.click(fn=lambda: load_example_video("rescue_simulation.mp4"), outputs=video)
        
app.queue(max_size=10, api_open=False)

if __name__ == "__main__":
    app.launch(max_file_size=MAX_UPLOAD_BYTES)
