import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import time
import os

def main():
    """
    Main function to run video inference.
    """
    # --- 1. Hardcoded Configuration ---
    MODEL_PATH = "path/to/your_model.pt"  # <<< IMPORTANT: SET THE PATH TO YOUR MODEL
    VIDEO_PATH = "path/to/your_video.mp4"  # <<< IMPORTANT: SET THE PATH TO YOUR VIDEO
    OUTPUT_PATH = "output/result.mp4"       # Optional: Path to save output video. Set to None to disable.
    CONFIDENCE_THRESHOLD = 0.5

    CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
        'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
        'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
        'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
        'hair drier', 'toothbrush', 'hair brush'
    ]

    IGNORE_CLASSES = [
        "Boat", "Fire Hydrant", "Parking Meter", "Bird", "Horse", "Sheep", "Cow",
        "Elephant", "Bear", "Zebra", "Giraffe", "Umbrella",
        "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite",
        "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket",
        "Wine Glass", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
        "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Couch", "Potted Plant",
        "Dining Table", "Tv", "Laptop", "Mouse", "Remote", "Keyboard", "Microwave",
        "Toaster", "Sink", "Book", "Clock", "Vase", "Scissors", "Teddy Bear",
        "Hair Drier", "Refrigerator", "Airplane", "Oven", "Traffic Light"
    ]
    # Create a set of lowercase class names for efficient, case-insensitive lookup
    ignore_classes_set = {name.lower() for name in IGNORE_CLASSES}


    # --- 2. Setup and Configuration ---
    # Select device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a color map for each class
    colors = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

    # --- 3. Load Model ---
    try:
        # Load the model and map it to the selected device
        model = torch.load(MODEL_PATH, map_location=device)
        # Set the model to evaluation mode
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 4. Video Processing Setup ---
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup video writer if an output path is provided
    out = None
    if OUTPUT_PATH:
        output_dir = os.path.dirname(OUTPUT_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
        print(f"Output video will be saved to: {OUTPUT_PATH}")

    # --- 5. Inference Loop ---
    frame_count = 0
    total_fps = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        
        # Preprocess the frame
        # 1. Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. Convert to tensor
        # NOTE: You might need to add more transforms like resizing or normalization
        # depending on how your model was trained.
        # Example: F.resize(img, (h, w))
        input_tensor = F.to_tensor(rgb_frame).to(device)
        # 3. Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = model(input_tensor)

        # The prediction format depends on the model.
        # We assume a standard torchvision object detection model output:
        # predictions[0] = {'boxes': tensor, 'labels': tensor, 'scores': tensor}
        pred = predictions[0]

        # --- 6. Visualization ---
        for i in range(len(pred['scores'])):
            score = pred['scores'][i].item()
            if score > CONFIDENCE_THRESHOLD:
                # Get class label
                label_idx = pred['labels'][i].item()
                class_name = CLASS_NAMES[label_idx]

                # Check if the class should be ignored (case-insensitive)
                if class_name.lower() in ignore_classes_set:
                    continue

                # Get bounding box coordinates
                box = pred['boxes'][i].cpu().numpy().astype(int)
                (startX, startY, endX, endY) = box

                # Get color for the class
                color = colors[label_idx]

                # Draw bounding box and label
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                text = f"{class_name}: {score:.2f}"
                cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate and display FPS
        end_time = time.time()
        current_fps = 1 / (end_time - start_time)
        total_fps += current_fps
        frame_count += 1
        avg_fps = total_fps / frame_count
        fps_text = f"Avg FPS: {avg_fps:.2f}"
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Video Inference', frame)
        
        # Write to output file if specified
        if out:
            out.write(frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 7. Cleanup ---
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Inference complete.")

if __name__ == "__main__":
    main()

