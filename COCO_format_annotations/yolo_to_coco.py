import os
import json
import cv2

# Define paths
yolo_labels_dir = "labels"  # Directory containing YOLO .txt files
image_dir = "images"  # Directory containing images
output_coco_file = "output_coco.json"  # Output JSON file

# Initialize COCO format
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Define categories (Update this based on your dataset)
category_mapping = {i: f"category_{i}" for i in range(26)}
for class_id, class_name in category_mapping.items():
    coco_data["categories"].append({"id": class_id, "name": class_name})

annotation_id = 1  # Unique ID for each annotation
image_id = 1  # Unique ID for each image

# Process each image and its corresponding YOLO annotation
for txt_file in os.listdir(yolo_labels_dir):
    if txt_file.endswith(".txt"):
        image_file = txt_file.replace(".txt", ".jpg")  # Assuming images are .jpg
        image_path = os.path.join(image_dir, image_file)

        # Read image dimensions
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_height, img_width, _ = image.shape

        # Add image info to COCO JSON
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": img_width,
            "height": img_height
        })

        # Read YOLO annotations
        with open(os.path.join(yolo_labels_dir, txt_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert YOLO format to COCO format
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                abs_width = int(width * img_width)
                abs_height = int(height * img_height)

                # Add annotation info to COCO JSON
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, abs_width, abs_height],
                    "area": abs_width * abs_height,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

# Save the converted annotations as a JSON file
with open(output_coco_file, "w") as json_file:
    json.dump(coco_data, json_file, indent=4)

print(f"Conversion complete! COCO annotations saved in {output_coco_file}")
