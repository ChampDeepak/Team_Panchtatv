# test_model.py

from ultralytics import YOLO

# --- Step 1: Specify the paths ---

# Path to the dataset configuration file
# This file tells YOLO where to find the test images.
data_config_path = 'batch1_data.yaml'

# Path to your trained model weights.
# You need to replace this with the correct path to YOUR 'best.pt' file.
# It's usually inside a directory like 'runs/train/batch1_experiment1/weights/'.
model_weights_path = 'runs/train/batch1_experiment1/weights/best.pt'


# --- Step 2: Load the Model ---

# Load your custom-trained model
model = YOLO(model_weights_path)


# --- Step 3: Run the Evaluation ---

# Evaluate the model's performance on the 'test' dataset split.
# The `split='test'` argument is crucial here. It tells YOLO to use the
# 'test: images/test' path from your 'batch1_data.yaml' file.
print("Starting evaluation on the test set...")
metrics = model.val(data=data_config_path, split='test')
print("Evaluation complete.")

# --- Step 4: Review the Results ---

# The results, including mAP50, mAP50-95, precision, and recall for each class,
# will be printed to your console.
# A new directory will also be created (e.g., 'runs/detect/val2') with detailed plots and results.
print(f"\nResults saved to {metrics.save_dir}")