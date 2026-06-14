import os
import shutil
import cv2
import numpy as np

NUM_TRAIN = 1000
NUM_VAL = 200
IMAGE_SIZE = 256


def setup_base_assets():
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)

    # Path where Antigravity saved the generated images
    brain_dir = r"C:\Users\itzdh\.gemini\antigravity-ide\brain\70088804-d8df-43d2-b8c3-ef099aced3c7"

    # Map of filenames
    files_to_copy = {
        "satellite_clean_forest_1781418811513.png": "forest.png",
        "satellite_clean_coastal_1781418832631.png": "coastal.png",
        "satellite_clean_mountains_1781418895700.png": "mountains.png",
        "satellite_clean_urban_1781418973164.png": "urban.png",
    }

    copied_any = False
    for src_name, dst_name in files_to_copy.items():
        src_path = os.path.join(brain_dir, src_name)
        dst_path = os.path.join(assets_dir, dst_name)
        if not os.path.exists(dst_path):
            if os.path.exists(src_path):
                print(f"Copying {src_name} to local assets/{dst_name}...")
                shutil.copy(src_path, dst_path)
                copied_any = True
            else:
                print(f"Warning: Source asset not found at {src_path}")

    return assets_dir


def generate_fractal_noise(size, scale=0.05, octaves=4, persistence=0.5):
    noise = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        grid_size = int(size * scale * frequency)
        grid_size = max(4, min(grid_size, size))
        grid = np.random.uniform(0, 1, (grid_size, grid_size)).astype(np.float32)
        smooth = cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)
        noise += smooth * amplitude
        amplitude *= persistence
        frequency *= 2.0

    noise_min = noise.min()
    noise_max = noise.max()
    if noise_max > noise_min:
        noise = (noise - noise_min) / (noise_max - noise_min)
    else:
        noise = np.zeros_like(noise)
    return noise


def generate_cloud_mask(size=256):
    scale = np.random.uniform(0.02, 0.08)
    octaves = np.random.randint(3, 5)
    noise = generate_fractal_noise(size, scale=scale, octaves=octaves)

    cloud_threshold = np.random.uniform(0.3, 0.5)
    mask = (noise - cloud_threshold) / (1.0 - cloud_threshold)
    mask = np.clip(mask, 0.0, 1.0)

    # Warp the mask with low-frequency noise displacement for wind-blown realism
    warp_scale = np.random.uniform(0.01, 0.03)
    warp_noise_x = generate_fractal_noise(size, scale=warp_scale, octaves=2) * 20.0
    warp_noise_y = generate_fractal_noise(size, scale=warp_scale, octaves=2) * 20.0

    grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
    map_x = (grid_x + warp_noise_x).astype(np.float32)
    map_y = (grid_y + warp_noise_y).astype(np.float32)

    mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return mask


def generate_procedural_terrain(size=256):
    heights = generate_fractal_noise(size, scale=0.02, octaves=4, persistence=0.5)
    moisture = generate_fractal_noise(size, scale=0.04, octaves=4, persistence=0.5)

    terrain = np.zeros((size, size, 3), dtype=np.uint8)

    h_rgb = np.expand_dims(heights, axis=-1)
    m_rgb = np.expand_dims(moisture, axis=-1)

    water_deep = np.array([120, 50, 10])  # BGR
    water_shallow = np.array([180, 120, 30])
    sand = np.array([145, 195, 210])
    plains = np.array([60, 140, 110])
    forest = np.array([20, 80, 25])
    rock = np.array([90, 95, 100])
    snow = np.array([250, 245, 240])

    water_factor = heights / 0.4
    water_factor_rgb = np.expand_dims(water_factor, axis=-1)
    water_color = (water_deep * (1.0 - water_factor_rgb) + water_shallow * water_factor_rgb).astype(np.uint8)

    veg_color = (plains * (1.0 - m_rgb) + forest * m_rgb).astype(np.uint8)

    terrain = np.where(h_rgb < 0.4, water_color, terrain)
    terrain = np.where((h_rgb >= 0.4) & (h_rgb < 0.43), sand, terrain)
    terrain = np.where((h_rgb >= 0.43) & (h_rgb < 0.75), veg_color, terrain)
    terrain = np.where((h_rgb >= 0.75) & (h_rgb < 0.85), rock, terrain)
    terrain = np.where(h_rgb >= 0.85, snow, terrain)

    texture_noise = np.random.normal(0, np.random.uniform(1, 5), terrain.shape).astype(np.float32)
    terrain = np.clip(terrain + texture_noise, 0, 255).astype(np.uint8)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(75, 95)]
    _, jpeg_data = cv2.imencode(".jpg", terrain, encode_param)
    terrain = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)

    if np.random.rand() > 0.3:
        river_pts = []
        start_x = np.random.randint(0, size)
        current_x = start_x
        for y in range(size):
            offset = int(15 * np.sin(y * 0.05 + start_x))
            current_x = np.clip(current_x + np.random.randint(-1, 2) + int(offset * 0.05), 0, size - 1)
            river_pts.append([current_x, y])

        cv2.polylines(
            terrain,
            [np.array(river_pts, dtype=np.int32)],
            isClosed=False,
            color=(160, 100, 20),
            thickness=np.random.randint(2, 5),
        )

    return terrain.astype(np.uint8)


def get_clean_patch(assets_dir, size=256):
    if not os.path.exists(assets_dir):
        return generate_procedural_terrain(size)

    files = [f for f in os.listdir(assets_dir) if f.endswith(".png")]
    if not files:
        return generate_procedural_terrain(size)

    filename = np.random.choice(files)
    img_path = os.path.join(assets_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        return generate_procedural_terrain(size)

    h, w = img.shape[:2]
    if h < size or w < size:
        img = cv2.resize(img, (max(size, w), max(size, h)))
        h, w = img.shape[:2]

    y = np.random.randint(0, h - size + 1)
    x = np.random.randint(0, w - size + 1)
    patch = img[y : y + size, x : x + size]

    # Augmentations
    if np.random.rand() > 0.5:
        patch = cv2.flip(patch, 1)
    if np.random.rand() > 0.5:
        patch = cv2.flip(patch, 0)

    rot = np.random.choice([0, 90, 180, 270])
    if rot == 90:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        patch = cv2.rotate(patch, cv2.ROTATE_180)
    elif rot == 270:
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return patch


def generate_synthetic_pair(assets_dir, size=256):
    clean = get_clean_patch(assets_dir, size)
    mask = generate_cloud_mask(size)

    cloud_type = np.random.choice(["cumulus", "cirrus", "stratus", "mixed"])
    if cloud_type == "cumulus":
        base_intensity = np.random.randint(235, 256)
    elif cloud_type == "cirrus":
        base_intensity = np.random.randint(200, 230)
    elif cloud_type == "stratus":
        base_intensity = np.random.randint(210, 240)
    else:
        base_intensity = np.random.randint(200, 256)

    cloud_color = np.full_like(clean, base_intensity, dtype=np.uint8)

    cloud_texture = generate_fractal_noise(size, scale=0.08, octaves=2)
    cloud_texture = 0.7 + 0.3 * cloud_texture
    cloud_color = (cloud_color * cloud_texture[..., None]).astype(np.uint8)

    opacity = np.random.uniform(0.6, 1.0)
    effective_mask = mask * opacity

    cloudy = (clean * (1.0 - effective_mask[..., None]) + cloud_color * effective_mask[..., None]).astype(np.uint8)

    sensor_noise = np.random.normal(0, np.random.uniform(1, 4), cloudy.shape).astype(np.float32)
    cloudy = np.clip(cloudy + sensor_noise, 0, 255).astype(np.uint8)

    return cloudy, clean


def generate_dataset(output_dir, num_pairs, assets_dir):
    cloudy_dir = os.path.join(output_dir, "cloudy")
    clean_dir = os.path.join(output_dir, "clean")
    os.makedirs(cloudy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    for i in range(num_pairs):
        cloudy, clean = generate_synthetic_pair(assets_dir, IMAGE_SIZE)
        cv2.imwrite(os.path.join(cloudy_dir, f"{i:04d}.png"), cloudy)
        cv2.imwrite(os.path.join(clean_dir, f"{i:04d}.png"), clean)

    print(f"Generated {num_pairs} pairs in {output_dir}")


if __name__ == "__main__":
    assets_dir = setup_base_assets()
    generate_dataset("data/train", NUM_TRAIN, assets_dir)
    generate_dataset("data/val", NUM_VAL, assets_dir)
    print("Synthetic data generation complete.")
