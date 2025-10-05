from ultralytics import YOLO

# 1. Load base yolov8n model (no fine-tuning)
model = YOLO("yolov8n.pt")

# 2. Run validation on your dataset
metrics = model.val(data="batch1_data.yaml", split="val")

# 3. Print results
print("📊 Base Model Performance on Validation Set:")
print(metrics)  # dictionary with metrics like mAP50, mAP50-95, precision, recall
