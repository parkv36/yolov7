import os
import shutil
import numpy as np
import joblib
from annotation_utils import preprocess_image, extract_features
import re

# ---- CONFIGURATION ----
rgb_folder = '/Users/BrendanLeahey/Desktop/CS Classes/Thesis/data/yolo-labelled/dronevehicle/rgb'
ir_folder = '/Users/BrendanLeahey/Desktop/CS Classes/Thesis/data/yolo-labelled/dronevehicle/tir'  # Same level as rgb_folder
output_base = '/Users/BrendanLeahey/Desktop/CS Classes/Thesis/data/sorted'
clf_path = 'runs/time_of_day_classifier.tz'
file_prefix = "dronevehicle" # e.g., "dronevehicle"

time_of_day_classes = {
    0: "pre_sunrise_or_post_sunset",
    1: "post_sunrise_or_pre_sunset",
    2: "noon"
}

# ---- CLASSIFY & MOVE ----
def classify_and_move_images():
    clf = joblib.load(clf_path)
    os.makedirs(output_base, exist_ok=True)

    rgb_filenames = [f for f in os.listdir(rgb_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in rgb_filenames:
        rgb_path = os.path.join(rgb_folder, fname)

        # Extract numeric part of the RGB filename
        base_num = extract_leading_number(fname)

        # Find matching IR file
        if file_prefix != "dronevehicle":
            ir_base_num = base_num + 100
        else:
            ir_base_num = base_num
        ir_fname = None
        for candidate in os.listdir(ir_folder):
            candidate_base = os.path.splitext(candidate)[0]
            match = re.match(r'^(\d+)', candidate_base)
            if match and int(match.group(1)) == ir_base_num and candidate.lower().endswith(('.jpg', '.jpeg', '.png')):
                ir_fname = candidate
                break

        if ir_fname is None:
            print(f"[SKIPPED] {fname} — IR counterpart {ir_base_num} not found.")
            continue

        ir_path = os.path.join(ir_folder, ir_fname)

        # Label paths
        rgb_label_path = os.path.splitext(rgb_path)[0] + ".txt"
        ir_label_path = os.path.join(ir_folder, os.path.splitext(ir_fname)[0] + ".txt")

        img, hsv = preprocess_image(rgb_path)
        features = extract_features(img, hsv)
        prediction = clf.predict([features])[0]
        label = time_of_day_classes[prediction]

        # Create target directory
        target_dir = os.path.join(output_base, label)
        os.makedirs(target_dir, exist_ok=True)

        # Build output filenames with prefix
        output_rgb_image = file_prefix + fname
        output_ir_image = file_prefix + "IR_" + fname

        output_rgb_label = file_prefix + os.path.splitext(fname)[0] + ".txt"
        output_ir_label = file_prefix + "IR_" + os.path.splitext(fname)[0] + ".txt"

        # Copy RGB image
        shutil.copy(rgb_path, os.path.join(target_dir, output_rgb_image))

        # Copy IR image
        shutil.copy(ir_path, os.path.join(target_dir, output_ir_image))

        # Copy RGB label if it exists
        if os.path.exists(rgb_label_path):
            shutil.copy(rgb_label_path, os.path.join(target_dir, output_rgb_label))
        else:
            print(f"[WARN] RGB label missing for {fname}")

        # Copy IR label if it exists
        if os.path.exists(ir_label_path):
            shutil.copy(ir_label_path, os.path.join(target_dir, output_ir_label))
        else:
            print(f"[WARN] IR label missing for {ir_fname}")

        print(f"[OK] {fname} ➜ {label}")

def extract_leading_number(filename):
    """Extract leading number from a filename like '101person.jpg'."""
    base_name = os.path.splitext(filename)[0]
    match = re.match(r'^(\d+)', base_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not find leading number in filename: {filename}")

# ---- MAIN ----
if __name__ == "__main__":
    if not os.path.exists(rgb_folder):
        raise FileNotFoundError(f"RGB folder not found: {rgb_folder}")
    if not os.path.exists(ir_folder):
        raise FileNotFoundError(f"IR folder not found: {ir_folder}")
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classifier not found: {clf_path}")

    classify_and_move_images()