import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import make_loaders

# --- IMPORTANT ---
# !! Adjust these paths to point to your data !!
DATA_ROOT = "data"  # Path to your 'data' folder (containing 'raw' and 'marked')
SPLITS_DIR = "data/splits" # Path to folder containing 'train.txt' and 'val.txt'

# --- Augmentation Stats ---
# These MUST match the values in your augmentations.py file
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def un_normalize(tensor, mean, std):
    """Reverses the normalization on a tensor for plotting."""
    # Clone to avoid changing the original tensor
    tensor = tensor.clone() 
    # Loop over each channel
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) # t = t * s + m
    return tensor

def plot_batch(images, masks):
    """Plots a batch of images and their masks."""
    batch_size = len(images)
    fig, ax = plt.subplots(nrows=2, ncols=batch_size, figsize=(batch_size * 4, 8))

    # Ensure 'ax' is always a 2D array for easy indexing
    if batch_size == 1:
        ax = ax.reshape(2, 1)

    print("Plotting batch... (Close window to exit)")

    for i in range(batch_size):
        # --- Process Image ---
        img_tensor = images[i]
        img = un_normalize(img_tensor, MEAN, STD)
        # Move from [C, H, W] to [H, W, C] for matplotlib
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1) # Clip values to [0, 1] range

        # --- Process Mask ---
        # Move from [1, H, W] to [H, W]
        mask_np = masks[i].squeeze().numpy()

        # Plot image
        ax[0, i].imshow(img_np)
        ax[0, i].set_title(f"Image {i} (Augmented)")
        ax[0, i].axis("off")

        # Plot mask
        ax[1, i].imshow(mask_np, cmap="gray")
        ax[1, i].set_title(f"Mask {i} (Augmented)")
        ax[1, i].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    print("Loading data... (This may take a moment)")
    # Get the loaders
    train_loader, _ = make_loaders(
        data_root=DATA_ROOT,
        splits_dir=SPLITS_DIR,
        batch_size=4, # Let's look at 4 images
        num_workers=0
    )

    # Get just one batch
    images, masks, _ = next(iter(train_loader))

    print(f"Batch loaded.")
    print(f"Image tensor shape: {images.shape}") # Should be [4, 3, H, W]
    print(f"Mask tensor shape: {masks.shape}")   # Should be [4, 1, H, W]

    plot_batch(images, masks)

if __name__ == "__main__":
    main()