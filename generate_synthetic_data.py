import os
import cv2
import numpy as np

NUM_TRAIN = 200
NUM_VAL = 50
IMAGE_SIZE = 256


def generate_synthetic_pair(size=256):
    clean = np.random.randint(20, 200, (size, size, 3), dtype=np.uint8)

    num_polygons = np.random.randint(8, 20)
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(num_polygons):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.randint(10, 60)
        pts = []
        for i in range(np.random.randint(5, 10)):
            angle = 2 * np.pi * i / np.random.randint(5, 10)
            pts.append([
                int(cx + r * np.random.uniform(0.5, 1.0) * np.cos(angle)),
                int(cy + r * np.random.uniform(0.5, 1.0) * np.sin(angle)),
            ])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

    cloud_intensity = np.random.randint(180, 256)
    cloud_color = np.full_like(clean, cloud_intensity, dtype=np.uint8)
    cloudy = np.where(mask[..., None] > 0, cloud_color, clean).astype(np.uint8)

    cloudy = cv2.GaussianBlur(cloudy, (5, 5), 0)

    return cloudy, clean


def generate_dataset(output_dir, num_pairs):
    cloudy_dir = os.path.join(output_dir, "cloudy")
    clean_dir = os.path.join(output_dir, "clean")
    os.makedirs(cloudy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    for i in range(num_pairs):
        cloudy, clean = generate_synthetic_pair(IMAGE_SIZE)
        cv2.imwrite(os.path.join(cloudy_dir, f"{i:04d}.png"), cloudy)
        cv2.imwrite(os.path.join(clean_dir, f"{i:04d}.png"), clean)

    print(f"Generated {num_pairs} pairs in {output_dir}")


if __name__ == "__main__":
    generate_dataset("data/train", NUM_TRAIN)
    generate_dataset("data/val", NUM_VAL)
    print("Synthetic data generation complete.")
