from ultralytics import YOLO

# Load a pretrained YOLOv8 model (you can choose 'yolov8n', 'yolov8s', etc.)
model = YOLO('yolov8s.pt')

# Train the model
model.train(data=r'copy your path\Desktop\PPE\data.yaml', epochs=20, imgsz=640)


