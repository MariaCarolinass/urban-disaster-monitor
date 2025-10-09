from ultralytics import YOLO

# yolo11n.pt yolo11m.pt yolo11l.pt
model = YOLO('yolo11n.pt')

model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolo11n_results'
)