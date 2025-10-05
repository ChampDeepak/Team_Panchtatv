import os
import glob
import shutil
from PIL import Image

# ✅ CONFIG
# DATASET_ROOT = "/home/deepak/Downloads/Batch1" 
DATASET_ROOT = "/home/deepak/Downloads/Batch2"
NUM_CLASSES = 35  # must match your .yaml file
CORRUPT_DIR = os.path.join(DATASET_ROOT, "corrupt")

os.makedirs(CORRUPT_DIR, exist_ok=True)

def move_corrupt(file_path, reason):
    """Move bad files into /corrupt folder with reason noted."""
    base = os.path.basename(file_path)
    new_path = os.path.join(CORRUPT_DIR, base)
    print(f"🗑️ Moving {file_path} → {new_path} ({reason})")
    try:
        shutil.move(file_path, new_path)
    except Exception as e:
        print(f"⚠️ Could not move {file_path}: {e}")

def check_and_fix_split(split="train"):
    image_dir = os.path.join(DATASET_ROOT, "images", split)
    label_dir = os.path.join(DATASET_ROOT, "labels", split)

    print(f"\n🔎 Checking split: {split}")
    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")

    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))

    problems = []
    max_class_id = -1

    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")

        # 1️⃣ Check if image can be opened
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            problems.append(f"❌ Corrupt image: {img_path}")
            move_corrupt(img_path, "Corrupt image")
            continue

        # 2️⃣ Check if label exists
        if not os.path.exists(label_path):
            problems.append(f"⚠️ Missing label for {img_path}")
            move_corrupt(img_path, "Missing label")
            continue

        # 3️⃣ Check label content
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            if len(lines) == 0:
                problems.append(f"⚠️ Empty label file: {label_path}")
                move_corrupt(label_path, "Empty label")
                continue

            valid_lines = []
            for ln in lines:
                parts = ln.strip().split()
                if len(parts) != 5:
                    problems.append(f"❌ Wrong format in {label_path}: '{ln.strip()}'")
                    move_corrupt(label_path, "Bad format")
                    break

                cid = int(parts[0])
                if cid > max_class_id:
                    max_class_id = cid

                if cid < 0 or cid >= NUM_CLASSES:
                    problems.append(f"❌ Invalid class ID {cid} in {label_path}")
                    move_corrupt(label_path, "Invalid class ID")
                    break

                # Check normalized coords
                coords = list(map(float, parts[1:]))
                if not all(0 <= c <= 1 for c in coords):
                    problems.append(f"❌ Out-of-range bbox in {label_path}: {coords}")
                    move_corrupt(label_path, "Bad bbox")
                    break

        except Exception as e:
            problems.append(f"❌ Could not read label {label_path}: {e}")
            move_corrupt(label_path, "Unreadable label")

    print(f"✅ Checked {len(image_files)} images in {split}")
    if problems:
        print(f"⚠️ Found {len(problems)} issues in {split}")
    else:
        print("🎉 No problems in this split!")

    print(f"📊 Max class ID found: {max_class_id}")
    return problems, max_class_id


if __name__ == "__main__":
    all_problems = []
    max_ids = []
    for split in ["train", "val", "test"]:
        if os.path.exists(os.path.join(DATASET_ROOT, "images", split)):
            probs, max_id = check_and_fix_split(split)
            all_problems.extend(probs)
            max_ids.append(max_id)

    if all_problems:
        print("\n🚨 Dataset had issues → bad files moved to:")
        print(CORRUPT_DIR)
    else:
        print("\n✅ Dataset is clean!")

    if max_ids:
        print(f"📊 Maximum class ID across all splits: {max(max_ids)}")
        print(f"⚖️ Expected maximum ID: {NUM_CLASSES-1}")
