import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile

model_image = YOLO("models/best50p.pt")
model_video = YOLO("models/best30p.pt")

def image_detection(image, conf_threshold):    
    results = model_image(image, conf=conf_threshold) 
    annotated_image = results[0].plot()
    predictions = results[0].names 
    return annotated_image, predictions

def video_detection(video_path, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_classes = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model_video(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()

        for c in results[0].boxes.cls:
            class_name = results[0].names[int(c)]
            all_classes.add(class_name)

        out.write(annotated_frame)

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

app.launch()
