import cv2
import numpy as np
import random
import os

INPUT_DIR = "/Users/omarelshobky/Downloads/ECG/"
OUTPUT_DIR = "/Users/omarelshobky/Downloads/ECG/"
angles = [15, -15, 30, -30, 45, -45, 60, -60,
          75, -75, 90, -90, 105, -105, 130, -130, 150, -150, 180]

# Higher weight for smaller angles
weights = [
    7, 7,       # ±15
    5, 5,       # ±30
    4, 4,       # ±45
    3, 3,       # ±60
    2, 2,       # ±75
    1, 1,       # ±90
    1, 1,       # ±105
    1, 1,       # ±130
    1, 1,       # ±150
    1           # 180
]

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute rotated bounding box to avoid cropping
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust for translation
    mat[0, 2] += (new_w / 2) - center[0]
    mat[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return rotated

def zoom_image(img, zoom_factor):
    """Zoom in or zoom out while keeping final size same."""
    h, w = img.shape[:2]

    # Zoom in (crop + resize back)
    if zoom_factor > 1:
        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)
        y1 = (h - new_h) // 2
        x1 = (w - new_w) // 2
        cropped = img[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (w, h))

    # Zoom out (pad + resize back)
    else:
        small = cv2.resize(img, (int(w * zoom_factor), int(h * zoom_factor)))
        new_img = np.zeros_like(img)
        y_start = (h - small.shape[0]) // 2
        x_start = (w - small.shape[1]) // 2
        new_img[y_start:y_start+small.shape[0], x_start:x_start+small.shape[1]] = small
        return new_img



def augment_image(img_path, save_dir):
    img = cv2.imread(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 1) Redundant original copies
    for i in range(1, 4):
        cv2.imwrite(f"{save_dir}/{base_name}_orig{i}.png", img)

    # 2) Zoom in / out
    zoom_in = zoom_image(img, 1.15)
    zoom_out = zoom_image(img, 0.85)

    cv2.imwrite(f"{save_dir}/{base_name}_zoom_in.jpg", zoom_in)
    cv2.imwrite(f"{save_dir}/{base_name}_zoom_out.jpg", zoom_out)

    # 3) 3 rotations w/ weighted probability
    for i in range(1, 4):
        angle = random.choices(angles, weights=weights, k=1)[0]
        rotated = rotate_image(img, angle)
        cv2.imwrite(f"{save_dir}/{base_name}_rot_{angle}_{i}.png", rotated)

# Example usage

for image_path in os.listdir(INPUT_DIR):
    augment_image(os.path.join(INPUT_DIR,image_path), OUTPUT_DIR)
print("Augmentation completed!")
