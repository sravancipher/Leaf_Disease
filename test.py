# import cv2
# import numpy as np

# img = cv2.imread("00f2e69a-1e56-412d-8a79-fdce794a17e4___JR_B.Spot 3132.JPG")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower = np.array([25, 40, 40])
# upper = np.array([90, 255, 255])

# mask = cv2.inRange(hsv, lower, upper)
# mask = cv2.medianBlur(mask, 7)

# cv2.imwrite("mask.png", mask)

import os
import cv2
import numpy as np

INPUT_DIR = "PlantVillage"
OUTPUT_DIR = "PlantVillage_Masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

def generate_leaf_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

for disease in os.listdir(INPUT_DIR):
    disease_path = os.path.join(INPUT_DIR, disease)

    if not os.path.isdir(disease_path):
        continue

    output_disease_dir = os.path.join(OUTPUT_DIR, disease)
    os.makedirs(output_disease_dir, exist_ok=True)

    for img_name in os.listdir(disease_path):
        if not img_name.endswith(VALID_EXTS):
            continue

        img_path = os.path.join(disease_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        mask = generate_leaf_mask(image)

        # 🔥 Preserve EXACT extension
        base, ext = os.path.splitext(img_name)
        mask_name = base + ext
        mask_path = os.path.join(output_disease_dir, mask_name)

        cv2.imwrite(mask_path, mask)

        print(f"Saved mask: {mask_path}")

# import os
# import pandas as pd

# IMAGE_ROOT = "PlantVillage"
# OUTPUT_CSV = "labels.csv"

# rows = []
# label_map = {}

# for idx, disease in enumerate(sorted(os.listdir(IMAGE_ROOT))):
#     disease_path = os.path.join(IMAGE_ROOT, disease)

#     if not os.path.isdir(disease_path):
#         continue

#     label_map[disease] = idx

#     for img_name in os.listdir(disease_path):
#         if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue

#         img_rel_path = f"{disease}/{img_name}"
#         rows.append([img_rel_path, idx, disease])

# df = pd.DataFrame(rows, columns=["image", "label", "label_name"])
# df.to_csv(OUTPUT_CSV, index=False)

# print("labels.csv generated successfully")
# print("Label Mapping:")
# for k, v in label_map.items():
#     print(f"{v} -> {k}")
