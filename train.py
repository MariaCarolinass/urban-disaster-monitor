from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolov8-desastre-v1.01'
)