import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the trained YOLOv8 model
model = YOLO(r'copy your path\runs\detect\train3\weights\best.pt')

# Path to validation images
val_images_path = 'C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/PPE/Validation/images'

# Get sample images
sample_images = [os.path.join(val_images_path, img) for img in os.listdir(val_images_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]

# Run inference and visualize
for image_path in sample_images:
    results = model.predict(source=image_path, save=False, conf=0.25)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Predictions for {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

