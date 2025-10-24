"""
Test script for the classification data loader.
Located at: /test_classifier_loader.py (in your root folder)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- IMPORTANT ---
# This should be the path to the root 'data' folder
# that contains the 'processed' folder.
DATA_ROOT = "data" 

# --- Augmentation Stats ---
# These MUST match the values in your augmentations.py file
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def un_normalize(tensor, mean, std):
    """Reverses the normalization on a tensor for plotting."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) # t = t * s + m
    return tensor

def plot_batch(images, labels, split_name):
    """Plots a batch of images and their labels."""
    batch_size = len(images)
    # Adjust for single image plotting
    if batch_size == 1:
        fig, ax = plt.subplots(ncols=1, figsize=(5, 6))
        ax = [ax] # Make it iterable
    else:
        # Plot up to 4 images
        plot_count = min(batch_size, 4)
        fig, ax = plt.subplots(ncols=plot_count, figsize=(plot_count * 4, 5))
        if plot_count == 1:
            ax = [ax]
        
    print(f"Plotting {plot_count} images from {split_name} batch... (Close window to continue)")
    
    for i in range(plot_count):
        # --- Process Image ---
        img_tensor = images[i]
        img = un_normalize(img_tensor, MEAN, STD)
        # Move from [C, H, W] to [H, W, C] for matplotlib
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1) # Clip values to [0, 1] range
        
        # --- Process Label ---
        label_val = labels[i].item()
        label = f"Positive (1.0)" if label_val == 1 else f"Negative (0.0)"
        
        # Plot image
        ax[i].imshow(img_np)
        ax[i].set_title(f"Image {i}\nLabel: {label}")
        ax[i].axis("off")

    plt.suptitle(f"Sample from {split_name} set", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    # --- Check for src/data/dataset.py ---
    # This is a common point of failure if the script is run from the wrong place
    dataset_py_path = os.path.join('src', 'data', 'dataset.py')
    if not os.path.exists(dataset_py_path):
        print(f"ERROR: Could not find '{dataset_py_path}'")
        print("Please make sure you are running this script from your ROOT project folder")
        print("(The folder that contains 'src', 'data', and 'test_classifier_loader.py')")
        sys.exit(1)
        
    # --- Dynamically add src to Python path ---
    # This allows the script to import from 'src.data.dataset'
    sys.path.insert(0, os.path.abspath(os.getcwd()))
    
    try:
        from src.data.dataset import make_loaders
    except ImportError as e:
        print(f"ERROR: Failed to import 'make_loaders' from 'src.data.dataset'.")
        print(f"Details: {e}")
        print("Make sure your virtual environment is active and all packages are installed.")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("This likely means a file *inside* src/data/dataset.py is failing to import.")
        print("Check 'from .augmentations import ...' inside dataset.py")
        sys.exit(1)


    print("Loading data... (This may take a moment)")
    # Get the loaders
    train_loader, val_loader, test_loader = make_loaders(
        data_root=DATA_ROOT,
        batch_size=4, # Let's look at 4 images
        num_workers=0 # Simpler for testing
    )
    
    if train_loader is None:
        print("Failed to create data loaders. Exiting.")
        return

    # --- Test Train Loader ---
    try:
        images, labels = next(iter(train_loader))
        print(f"\nBatch loaded from TRAIN set.")
        print(f"Image tensor shape: {images.shape}") # Should be [4, 3, H, W]
        print(f"Label tensor shape: {labels.shape}")   # Should be [4, 1]
        plot_batch(images, labels, "TRAIN")
    except StopIteration:
        print("ERROR: Training loader is empty. Check your 'data/processed' folder.")
        return
    except Exception as e:
        print(f"An error occurred while testing the train loader: {e}")

    # --- Test Validation Loader ---
    try:
        images, labels = next(iter(val_loader))
        print(f"\nBatch loaded from VALIDATION set.")
        print(f"Image tensor shape: {images.shape}")
        print(f"Label tensor shape: {labels.shape}")
        plot_batch(images, labels, "VALIDATION")
    except StopIteration:
        print("ERROR: Validation loader is empty.")
    except Exception as e:
        print(f"An error occurred while testing the validation loader: {e}")

    print("\nTest complete. If you saw the plots, your data loader is working!")

if __name__ == "__main__":
    main()
