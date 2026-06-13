import cv2
import numpy as np
import torch

from models.unet import UNet


def predict(image_path, checkpoint_path="models/checkpoints/unet_final.pth", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    img_resized = cv2.resize(img_rgb, (256, 256))
    img_tensor = (
        torch.from_numpy(img_resized.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    output = cv2.resize(output, (w, h))
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output_bgr
