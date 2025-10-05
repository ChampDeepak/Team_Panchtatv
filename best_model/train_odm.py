from ultralytics import YOLO

if __name__ == "__main__":
    # === CHOOSE YOUR STARTING MODEL (from Step 3) ===
    
    # Option A: Continue from your last training run (Recommended)
    model = YOLO("/home/deepak/cml_project/runs/train/batch1_experiment13/weights/best.pt")

    # Option B: Start fresh from the original YOLOv8n model
    # model = YOLO("yolov8n.pt") 
    
    # === TRAIN ON THE NEW DATASET ===
    results = model.train(
        data='batch2_data.yaml',        # <-- CHANGE: Use the new .yaml file
        epochs=100,
        imgsz=416,
        batch=2,
        workers=0,
        project='runs/train',
        name='batch2_experiment1',      # <-- CHANGE: Give it a new name to stay organized
        verbose=True
    )