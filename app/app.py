import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

model_image = YOLO("models/best50p.pt")
model_video = YOLO("models/best30p.pt")

def costum_bounding_box(image, results):
    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    class_names = results[0].names

    class_colors = {
        0: (255, 0, 0),      # vermelho
        1: (255, 255, 0),    # amarelo
    }

    font_colors = {
        0: (255, 255, 255),  # branco
        1: (0, 0, 0),        # preto
    }

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls]} {conf:.2f}"

        box_color = class_colors.get(cls, (0, 255, 0))     # fallback: verde
        text_color = font_colors.get(cls, (255, 255, 255)) # fallback: branco
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = x1
        text_y = y1 if y1 - text_h < 0 else y1

        bg_tl = (text_x, text_y - text_h)
        bg_br = (text_x + text_w, text_y)

        cv2.rectangle(annotated_image, bg_tl, bg_br, box_color, -1)

        cv2.putText(annotated_image, label, (text_x, text_y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)

    return annotated_image


def image_detection(image, conf_threshold):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model_image(image_bgr, conf=conf_threshold)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotated_image = costum_bounding_box(image_rgb, results)

    detected_classes = set(int(box.cls[0]) for box in results[0].boxes)
    class_names = results[0].names
    predictions = ", ".join(sorted([class_names[i] for i in detected_classes]))

    return annotated_image, predictions

def video_detection(video_path, conf_threshold, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_classes = set()
    frame_count = 0
    last_annotated_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            results = model_video(frame, conf=conf_threshold)
            annotated_frame = costum_bounding_box(frame, results)
        
            for c in results[0].boxes.cls:
                class_name = results[0].names[int(c)]
                all_classes.add(class_name)
                
            last_annotated_frame = annotated_frame
        else:
            if last_annotated_frame is not None:
                annotated_frame = last_annotated_frame
            else:
                annotated_frame = frame
        
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()

    predictions = ", ".join(sorted(all_classes)) if all_classes else "Nenhuma classe detectada"
    return temp_video_path, predictions

with gr.Blocks() as app:
    gr.HTML("""
        <h1 style='text-align: center'>Urban Disaster Monitor</h1>
        <p style='text-align: center'>This app is designed to monitor urban disasters using AI.</p>
        <p style='text-align: center'><a href='https://github.com/MariaCarolinass/urban-disaster-monitor' target='_blank'>GitHub</a></p>      
    """)

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Upload an Image", type="pil")
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                btn = gr.Button("Process Image", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Processed Image")
                output_predictions = gr.Textbox(label="Predictions", placeholder="Predictions will appear here...")
        
        btn.click(fn=image_detection, inputs=[image, conf_threshold], outputs=[output_image, output_predictions])
    
        gr.Examples(
            examples=[
                ["examples/3FF53711-E6A2-4581-BF42745FCF3CB0FA_source_jpg.rf.c0621d72c8985d028f877917a33ac72f.jpg"],
                ["examples/AAA960_jpg.rf.8389818b70436876a784d52ffa733e01.jpg"],
                ["examples/1013_jpg.rf.8504152d4e8d767ea8c0c4f892c5d30b.jpg"],
                ["examples/1424_jpg.rf.f5a76dff0779f8d132ca4443d25f80eb.jpg"],
                ["examples/1521_jpg.rf.5bd1ea1f52ea2990dddbe4a0bbbe5140.jpg"],
                ["examples/Flood-7_jpg.rf.a71bfe309c707883299f283ca207306b.jpg"]
            ],
            inputs=[image, conf_threshold],
            outputs=[output_image, output_predictions],
            label="Example Images"
        )
        
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Upload a Video", autoplay=True)
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                btn = gr.Button("Process Video", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="Processed Video", autoplay=True)
                output_predictions = gr.Textbox(label="Predictions", placeholder="Predictions will appear here...")

        btn.click(fn=video_detection, inputs=[video, conf_threshold], outputs=[output_video, output_predictions])

        gr.Examples(
            examples=[["examples/video5143238584093902399.mp4"]],
            inputs=[video, conf_threshold],
            outputs=[output_video, output_predictions],
            label="Example Videos"
        )

app.launch()
