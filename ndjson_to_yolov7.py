"""
This script processes NDJSON datasets for YOLOv7 model training.
python3 ndjson_to_yolov7.py /notebooks/labware_correct_validated2.ndjson /notebooks/yolov7/dataset
"""

import argparse
import hashlib
import json
import multiprocessing
import shutil
from functools import partial
from pathlib import Path

import requests
from google.cloud import storage
from PIL import Image
from tqdm import tqdm

# Save files to /notebooks/tmp directory use this to ln to download directory
TMP_DIR = Path("/notebooks/tmp")


def load_ndjson(ndjson_path):
    """Load ndjson dataset."""
    data = []
    with open(ndjson_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:50]}...")
    return data


def get_boundingbox_classes(data):
    """Extract and sort all classes that have bounding boxes."""
    classes = set()
    for item in data:
        # Navigate to the correct location for bounding boxes
        if "projects" in item:
            for project_id, project in item["projects"].items():
                if "labels" in project:
                    for label in project["labels"]:
                        if "annotations" in label and "objects" in label["annotations"]:
                            for obj in label["annotations"]["objects"]:
                                if "bounding_box" in obj and (
                                    "name" in obj or "value" in obj
                                ):
                                    # Use either name or value as the class label
                                    class_name = obj.get(
                                        "value", obj.get("name", "unknown")
                                    )
                                    classes.add(class_name)
    sorted_classes = sorted(list(classes))
    class_dict = {cls: idx for idx, cls in enumerate(sorted_classes)}
    return class_dict


def download_image(url, local_path, max_attempts=3):
    """Download image from URL and verify it can be opened with PIL."""
    local_path = Path(local_path)
    for attempt in range(max_attempts):
        try:
            # Remove file if it exists (for redownload attempts)
            if attempt > 0 and local_path.exists():
                local_path.unlink()

            if url.startswith("gs://"):
                # Google Cloud Storage URL
                client = storage.Client()
                bucket_name = url.split("/")[2]
                blob_path = "/".join(url.split("/")[3:])
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(str(local_path))
            elif url.startswith("http"):
                # HTTP/HTTPS URL
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                else:
                    continue  # Try again if response wasn't successful
            else:
                print(f"Unsupported URL scheme: {url}")
                return False

            # Verify the image can be opened with PIL
            try:
                img = Image.open(local_path)
                img.verify()  # Verify that it's a valid image
                img.close()  # Close the file
                return True  # Success if we reach here
            except Exception as e:
                print(f"Downloaded file isn't a valid image ({local_path}): {e}")
                # Continue to next attempt

        except Exception as e:
            print(f"Error downloading {url} (attempt {attempt+1}/{max_attempts}): {e}")

    # If we get here, all attempts failed
    print(f"Failed to download valid image from {url} after {max_attempts} attempts")
    return False


def download_single_image(item, image_dir):
    """Download a single image and return its path."""
    # Get image URL from the correct location
    image_url = None
    if "data_row" in item and "row_data" in item["data_row"]:
        image_url = item["data_row"]["row_data"]

    # Skip if no image URL
    if not image_url:
        return None, None

    # Get item ID from the correct location
    item_id = item.get("data_row", {}).get("id", "")
    if not item_id:
        return None, None

    # Generate MD5 hash of the URL
    md5_hash = hashlib.md5(image_url.encode()).hexdigest()
    local_path = Path(image_dir) / f"{md5_hash}.png"

    # Skip if already downloaded and valid
    if local_path.exists():
        try:
            # Verify the image can be opened with PIL
            img = Image.open(local_path)
            img.verify()  # Verify that it's a valid image
            img.close()  # Close the file
            return item_id, local_path
        except Exception as e:
            print(f"Existing image is invalid ({local_path}): {e}")
            # Continue to download since verification failed
            local_path.unlink()

    # Download image
    if download_image(image_url, local_path):
        return item_id, local_path

    return None, None


def download_images(data, image_dir, num_processes=None):
    """Download images from gs:// or https:// urls using multiprocessing."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        image_dir.mkdir(parents=True, exist_ok=True)

    if num_processes is None:
        # Use number of CPU cores minus 1, or at least 1
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    print(f"Using {num_processes} processes for downloading images")

    # Create a partial function with fixed image_dir parameter
    worker_fn = partial(download_single_image, image_dir=image_dir)

    # Create a pool of workers
    image_paths = {}
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap_unordered to process results as they come
        for item_id, local_path in tqdm(
            pool.imap_unordered(worker_fn, data),
            total=len(data),
            desc="Downloading images",
        ):
            if item_id is not None and local_path is not None:
                image_paths[item_id] = local_path

    return image_paths


def convert_to_yolo_format(
    data, image_paths, class_dict, output_dir, train_ratio=0.7, val_ratio=0.2
):
    """Convert annotations to YOLOv7 format."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset structure
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            path = output_dir / split / subdir
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    # Create classes.txt
    with open(output_dir / "classes.txt", "w") as f:
        for cls in sorted(class_dict.keys(), key=lambda x: class_dict[x]):
            f.write(f"{cls}\n")

    # Track split distribution
    split_counts = {"train": 0, "val": 0, "test": 0}

    # Process annotations
    for item in tqdm(data, desc="Converting annotations"):
        item_id = item.get("data_row", {}).get("id", "")
        if item_id not in image_paths:
            continue

        image_path = image_paths[item_id]
        image_basename = Path(image_path).name

        # Determine split based on ID rather than using metadata
        # This ensures a 70/20/10 train/val/test split
        split = determine_split(item_id, train_ratio, val_ratio)
        split_counts[split] += 1

        # Copy image to appropriate split directory
        target_image_path = output_dir / split / "images" / image_basename
        shutil.copy(image_path, target_image_path)

        # Create label file
        label_basename = Path(image_basename).stem + ".txt"
        label_path = output_dir / split / "labels" / label_basename

        # Get image dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Extract and convert annotations
        yolo_annotations = []

        if "projects" in item:
            for project_id, project in item["projects"].items():
                if "labels" in project:
                    for label in project["labels"]:
                        if "annotations" in label and "objects" in label["annotations"]:
                            for obj in label["annotations"]["objects"]:
                                if "bounding_box" in obj:
                                    bbox = obj["bounding_box"]
                                    # Use either name or value as the class label
                                    class_name = obj.get(
                                        "value", obj.get("name", "unknown")
                                    )

                                    if class_name in class_dict:
                                        class_idx = class_dict[class_name]

                                        # Extract bounding box coordinates
                                        x_min = float(bbox.get("left", 0))
                                        y_min = float(bbox.get("top", 0))
                                        width = float(bbox.get("width", 0))
                                        height = float(bbox.get("height", 0))

                                        # Normalize coordinates for YOLO format (center_x, center_y, width, height)
                                        center_x = (x_min + width / 2) / img_width
                                        center_y = (y_min + height / 2) / img_height
                                        norm_width = width / img_width
                                        norm_height = height / img_height

                                        # Append YOLO format annotation
                                        yolo_annotations.append(
                                            f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                                        )

        # Write annotations to file
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))

    # Print split distribution
    total = sum(split_counts.values())
    if total > 0:
        print("Dataset split distribution:")
        for split, count in split_counts.items():
            percentage = (count / total) * 100
            print(f"  {split}: {count} images ({percentage:.1f}%)")


def create_yolov7_config(class_dict, output_dir):
    """Create YOLOv7 config file."""
    output_dir = Path(output_dir)
    config_path = output_dir / "yolov7_custom.yaml"

    # Create config content
    config = {
        "train": str(output_dir / "train" / "images"),
        "val": str(output_dir / "val" / "images"),
        "test": str(output_dir / "test" / "images"),
        "nc": len(class_dict),
        "names": [
            cls for cls in sorted(class_dict.keys(), key=lambda x: class_dict[x])
        ],
    }

    # Write config to file
    with open(config_path, "w") as f:
        f.write("# YOLOv7 custom configuration\n")
        for key, value in config.items():
            if key == "names":
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"Created YOLOv7 config at: {config_path}")
    return config_path


def determine_split(item_id, train_ratio=0.7, val_ratio=0.2):
    """
    Determine which split (train, val, test) an item belongs to based on its ID.
    Uses hash of the ID to ensure consistent but random-like assignment.

    Args:
        item_id: Unique identifier for the item
        train_ratio: Proportion of data for training (default: 0.7 or 70%)
        val_ratio: Proportion of data for validation (default: 0.2 or 20%)

    Returns:
        str: 'train', 'val', or 'test'
    """
    # Use hash of ID to get a deterministic but random-like value
    hash_value = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 100

    # Determine split based on hash value and ratios
    if hash_value < train_ratio * 100:
        return "train"
    elif hash_value < (train_ratio + val_ratio) * 100:
        return "val"
    else:
        return "test"


def main(ndjson_path, output_dir, num_processes=None, train_ratio=0.7, val_ratio=0.2):
    """Main function to process the ndjson dataset for YOLOv7."""
    print(f"Processing NDJSON dataset: {ndjson_path}")
    print(
        f"Using split ratios: {train_ratio*100:.0f}% train, {val_ratio*100:.0f}% val, {(1-train_ratio-val_ratio)*100:.0f}% test"
    )

    # Load data
    data = load_ndjson(ndjson_path)
    print(f"Loaded {len(data)} items from NDJSON file")

    # Get bounding box classes
    class_dict = get_boundingbox_classes(data)
    print(f"Found {len(class_dict)} classes with bounding boxes:")
    for cls, idx in class_dict.items():
        print(f"  {idx}: {cls}")

    # Download images
    image_dir = Path(output_dir) / "raw_images"
    image_paths = download_images(data, image_dir, num_processes=num_processes)
    print(f"Downloaded {len(image_paths)} images to {image_dir}")

    # Convert to YOLOv7 format with specified split ratios
    convert_to_yolo_format(
        data,
        image_paths,
        class_dict,
        output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    print(f"Converted annotations to YOLOv7 format in {output_dir}")

    # Create YOLOv7 config
    config_path = create_yolov7_config(class_dict, output_dir)

    print("\nDataset preparation complete!")
    print(
        f"To train YOLOv7 on this dataset, use: python3 train.py --batch 64 --epochs 20 --data {config_path} --cfg /notebooks/yolov7/cfg/training/yolov7-tiny.yaml  --weights /notebooks/yolov7/yolov7-tiny.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NDJSON dataset for YOLOv7.")
    parser.add_argument("ndjson_path", help="Path to NDJSON dataset")
    parser.add_argument("output_dir", help="Directory to save the processed dataset")
    parser.add_argument(
        "--num-processes",
        "-j",
        type=int,
        default=None,
        help="Number of parallel download processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training (default: 0.7 or 70%)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion of data for validation (default: 0.2 or 20%)",
    )
    args = parser.parse_args()

    main(
        args.ndjson_path,
        args.output_dir,
        num_processes=args.num_processes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
