import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from models.unet import UNet


class CloudDataset(Dataset):
    def __init__(self, root_dir, image_size=256, augment=False):
        self.cloudy_dir = os.path.join(root_dir, "cloudy")
        self.clean_dir = os.path.join(root_dir, "clean")
        self.image_size = image_size
        self.augment = augment
        self.filenames = sorted(os.listdir(self.cloudy_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cloudy_path = os.path.join(self.cloudy_dir, self.filenames[idx])
        clean_path = os.path.join(self.clean_dir, self.filenames[idx])

        cloudy = cv2.imread(cloudy_path)
        cloudy = cv2.cvtColor(cloudy, cv2.COLOR_BGR2RGB)
        clean = cv2.imread(clean_path)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        cloudy = cv2.resize(cloudy, (self.image_size, self.image_size))
        clean = cv2.resize(clean, (self.image_size, self.image_size))

        cloudy = cloudy.astype(np.float32) / 255.0
        clean = clean.astype(np.float32) / 255.0

        if self.augment:
            brightness = np.random.uniform(0.85, 1.15)
            contrast = np.random.uniform(0.85, 1.15)
            cloudy = np.clip((cloudy - 0.5) * contrast + 0.5 + (brightness - 1.0), 0, 1)

            if np.random.rand() > 0.5:
                noise = np.random.normal(0, np.random.uniform(0.005, 0.02), cloudy.shape).astype(np.float32)
                cloudy = np.clip(cloudy + noise, 0, 1)

        cloudy = torch.from_numpy(cloudy).permute(2, 0, 1)
        clean = torch.from_numpy(clean).permute(2, 0, 1)

        return cloudy, clean


def train_model(
    data_dir="data",
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    image_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="models/checkpoints",
    results_dir="results/training_curves",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_dataset = CloudDataset(os.path.join(data_dir, "train"), image_size, augment=True)
    val_dataset = CloudDataset(os.path.join(data_dir, "val"), image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    train_losses = []
    val_losses = []

    print(f"Training on {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        for cloudy, clean in train_loader:
            cloudy, clean = cloudy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(cloudy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for cloudy, clean in val_loader:
                cloudy, clean = cloudy.to(device), clean.to(device)
                output = model(cloudy)
                loss = criterion(output, clean)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )

    final_path = os.path.join(checkpoint_dir, "unet_final.pth")
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        },
        final_path,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "loss_curves.png"))
    plt.close()

    print(f"Training complete. Model saved to {final_path}")
    return train_losses, val_losses


if __name__ == "__main__":
    train_model()
