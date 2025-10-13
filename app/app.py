import os
import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from huggingface_hub import hf_hub_download

def get_model_path(model_variant):
    os.makedirs("models", exist_ok=True)
    model_map = {
        "yolov11n": "carolinasoares/yolov11n",
        "yolov11s": "carolinasoares/yolov11s",
        "yolov11m": "carolinasoares/yolov11m",
        "yolov11l": "carolinasoares/yolov11l"
    }
    repo_id = model_map[model_variant]
    return hf_hub_download(repo_id=repo_id, filename=f"{model_variant}.pt", cache_dir="models")

def costum_bounding_box(image, results):
    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    class_names = results[0].names

    class_colors = {
        0: (255, 0, 0),      # vermelho
        1: (255, 255, 0),    # amarelo
        2: (0, 255, 0),      # verde
        3: (0, 0, 255),      # azul
        4: (255, 0, 255),    # magenta
        5: (0, 255, 255),    # ciano
    }

    font_colors = {
        0: (255, 255, 255),  # branco
        1: (0, 0, 0),        # preto
        2: (0, 0, 0),        # preto
        3: (255, 255, 255),  # branco
        4: (255, 255, 255),  # branco
        5: (0, 0, 0),        # preto
    }

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_color = class_colors.get(int(box.cls[0]), (255, 255, 255))  # fallback: branco
        text_color = font_colors.get(int(box.cls[0]), (0, 0, 0))        # fallback: preto
        label = f"{class_names[int(box.cls[0])]} {float(box.conf[0]):.2f}"

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = x1
        text_y = y1 if y1 - text_h < 0 else y1

        bg_tl = (text_x, text_y - text_h)
        bg_br = (text_x + text_w, text_y)

        cv2.rectangle(annotated_image, bg_tl, bg_br, box_color, -1)

        cv2.putText(annotated_image, label, (text_x, text_y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)

    return annotated_image


def image_detection(image, conf_threshold, model_variant):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    weights_path = get_model_path(model_variant)
    model_image = YOLO(weights_path)
    
    results = model_image(image_bgr, conf=conf_threshold)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotated_image = costum_bounding_box(image_rgb, results)

    detected_classes = set(int(box.cls[0]) for box in results[0].boxes)
    class_names = results[0].names
    predictions = ", ".join(sorted([class_names[i] for i in detected_classes]))

    return annotated_image, predictions

def video_detection(video_path, conf_threshold, model_variant, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_classes = set()
    frame_count = 0
    last_annotated_frame = None

    weights_path = get_model_path(model_variant)
    model_video = YOLO(weights_path)

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
        <h2 style='text-align: center'><a href='https://github.com/MariaCarolinass/urban-disaster-monitor' target='_blank'>GitHub</a></h2>
    """)

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Upload an Image", type="pil")
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                output_model = gr.Dropdown(["yolov11n", "yolov11s", "yolov11m", "yolov11l"], label="Select Model", info="Select the YOLOv11 model variant to use.", value="yolov11m")
                btn = gr.Button("Process Image", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Processed Image")
                output_predictions = gr.Textbox(label="Predictions", placeholder="Predictions will appear here...")

        btn.click(fn=image_detection, inputs=[image, conf_threshold, output_model], outputs=[output_image, output_predictions])
    
        gr.Examples(
            examples=[
                ["examples/1019715_jpg.rf.58a43da4e0959d4e75f1eceb0d288bd0.jpg"],
                ["examples/20250924_1153_Vacas_em_Alagamento_simple_compose_01k5y3bzjee4sbbf02c30c2phm1_png.rf.1caa0a0ff7a605e8b84669b0cc6fc364.jpg"],
                ["examples/230714-india-flooding-mb-0831-d3a66d_jpg.rf.3e607c4f8f121834224f95ab0d44ddd6.jpg"],
                ["examples/754_jpg.rf.47e7b8cdcfa1ffb020bb1b0588890f78.jpg"],
                ["examples/775_jpg.rf.d2c4a77e35dd329df2478517c42c1176.jpg"],
                ["examples/f-banglafloods-a-20190725_jpg.rf.db7b95e9eb7d8294b89644a27cc18166.jpg"],
                ["examples/Flood-25_jpg.rf.92d30a193fb4f368a8d92f65f9669244.jpg"],
                ["examples/Flood-30_jpg.rf.a9d21f122ddb98ee863989f552c0adc4.jpg"],
                ["examples/Flood-46_jpg.rf.1b3bd9e0e51798a4f61a51de0a694c6d.jpg"],
                ["examples/Flood-7_jpg.rf.a71bfe309c707883299f283ca207306b.jpg"],
                ["examples/image_123f58f43036403cb7aab908fe5fc69d_png.rf.b94dda710e99cc3ab85dbbd7f0d196f0.jpg"],
                ["examples/image23_jpeg.rf.20eca34e2be7c8a452a1ab682e1254cc.jpg"],
                ["examples/image_24d9705c165d4c818b9d10631d0ce48e_png.rf.5e504a35a21ec0f7adaba4a76a4edf09.jpg"],
                ["examples/image_2d402afa0296407d953e3fb2a46167a7_png.rf.f1aea2fc84dcf5428764241fe5843d53.jpg"],
                ["examples/image_4269233d29ec4a55941013d8660768db_png.rf.310026b583728ef0fc05a95e1fbffd42.jpg"],
                ["examples/image_536b176558764282b5dcfb33115db7bb_png.rf.0b1dfb32c8b26ed9520324d7e0123683.jpg"],
                ["examples/image_55dad25f64af4067be760720adfb3372_png.rf.5941414d221b13d9902e4005b5852a0c.jpg"],
                ["examples/image_572fd077c88b46bbbb3c6c5a74a93652_png.rf.f30b9b6d9e8abf00accb625882d0fdf9.jpg"],
                ["examples/image_674fd25133f64fd6bb6ddcfe36168583_png.rf.99e3cf8ca1ed9dc5b849b70394c6f545.jpg"],
                ["examples/image_6990bbbf052a4ae199b59e0151d1ce34_png.rf.c2d6327ad2ce2a66e2fdf6ab73882c91.jpg"],
                ["examples/image_91ff6fbe98c6465988897977f9a7a3ac_png.rf.e7925b19b659948901c256abc271b318.jpg"],
                ["examples/image_9c9a10f736f04d15a407c16e8eddd2b5_png.rf.89b86afb45331b0705a17f70369e0f3d.jpg"],
                ["examples/image_9c9d9969450e4db7ad86219f535c79c5_png.rf.23949fc7e161024b5d62c98a2279091e.jpg"],
                ["examples/image_9e67cb7ca8634c199296e5360aff9d52_png.rf.8331d74fd27b7369d7e9f7ad0d26caa4.jpg"],
                ["examples/image_a0e220e5d36b4253a0abf3db8e56c696_png.rf.1ba0ad185f497aca83d5c74087c181aa.jpg"],
                ["examples/image_b014c660bf2d439785cf1ffdfa9b5c55_png.rf.a306e8b69be0e029f98b52809649037e.jpg"],
                ["examples/image_b32be24ecb5f4af6ab4afb8f18f24f11_png.rf.2f605c0712858483d925480bda9c815c.jpg"],
                ["examples/image_b43bdbd062914dbdb513e2ef5f2b5d1d_png.rf.a7dfcd6f380c8ff12360459e06d67744.jpg"],
                ["examples/image_b4ec2525bfee4e8b8c154be463f7255e_png.rf.8b3986c3f6b2da8fa33f266734e57098.jpg"],
                ["examples/image_b7701fbd19444453a79356cae619bac7_png.rf.d0dd742a003c9ed23577eb367fd5ad92.jpg"],
                ["examples/image_de8817d6c699457bbee71252f69b83d2_png.rf.f00e820c7e9f340e16078d12722423c9.jpg"],
                ["examples/image_fed89c6e1d1c4f04a92ad4aca9f85f10_png.rf.799a7361fa4833d0b056c2e736c4048b.jpg"],
                ["examples/imagem_008_232_png.rf.7fc9f7b2b426747ebca6453e5e6ee2a6.jpg"],
                ["examples/imagem_032_png.rf.2611a927a635635b644206943646bc49.jpg"],
                ["examples/imagem_058-copia-_png.rf.22689b7c998b76dd1af82e218bb0ad7c.jpg"],
                ["examples/imagem_064-copia-_png.rf.2e109916a38a8cfe3b158303c3bfa95f.jpg"],
                ["examples/images18_jpg.rf.c6874bfb0609dc6d52defda4e161d25e.jpg"],
                ["examples/images219_jpg.rf.793784542da37f5a78a3837688314c97.jpg"],
                ["examples/ph_17939_63492_jpg.rf.bf0e962767adc644290645db26ab9e26.jpg"]
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
                output_model = gr.Dropdown(["yolov11n", "yolov11s", "yolov11m", "yolov11l"], label="Select Model", info="Select the YOLOv11 model variant to use.", value="yolov11m")
                btn = gr.Button("Process Video", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="Processed Video", autoplay=True)
                output_predictions = gr.Textbox(label="Predictions", placeholder="Predictions will appear here...")

        btn.click(fn=video_detection, inputs=[video, conf_threshold, output_model], outputs=[output_video, output_predictions])

        video_path = hf_hub_download("carolinasoares/urban-disaster-examples", "rescuer.mp4")

        gr.Examples(
            examples=[[video_path]],
            inputs=[video, conf_threshold, output_model],
            outputs=[output_video, output_predictions],
            label="Example Videos"
        )

app.launch(allowed_paths=["/home/carol/.cache/huggingface/hub"])
