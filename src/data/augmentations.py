"""
Image augmentation pipelines for CLASSIFICATION using Albumentations.
Located at: src/data/augmentations.py
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_HEIGHT = 224 
IMG_WIDTH = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_train_transforms():
    """
    Augmentations for the training set.
    Includes random geometric and color transformations.
    """
    return A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),
            A.Rotate(limit=90, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),

            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
                p=0.8
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

def get_val_transforms():
    """
    Augmentations for the validation/testing set.
    Only includes resizing and normalization.
    """
    return A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )