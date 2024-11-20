import numpy as np
import albumentations as A
import tensorflow as tf
from typing import Tuple, List


def augment_single (image, mask) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies data augmentation to a single image and its corresponding mask.

    The augmentation pipeline includes:
        - Vertical flip with a 50% probability.
        - Horizontal flip with a 50% probability.
        - Random brightness and contrast adjustments with a 50% probability.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        mask (np.ndarray): The corresponding binary mask as a NumPy array.

    Returns:
        tuple: Augmented image and mask as NumPy arrays.
    """
    augmentation_pipeline = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        )
    ])
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented['image'], augmented['mask']


def augment_dataset (images, masks) -> tf.data.Dataset:
    """
    Augments a dataset of images and masks, combines the original and augmented data,
    and prepares a TensorFlow dataset for model training.

    Augmentation is applied to each image-mask pair using `augment_single`. The augmented
    data is then combined with the original dataset, shuffled, and batched.

    Args:
        images (np.ndarray): Array of input images with shape (N, H, W, C).
        masks (np.ndarray): Array of corresponding binary masks with shape (N, H, W).

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing both original and augmented
                         image-mask pairs, batched for training.
    """
    aug_images: List = []
    aug_masks: List = []
    for img, mask in zip(images, masks):
        aug_img, aug_mask = augment_single(img, mask)
        aug_images.append(aug_img)
        aug_masks.append(aug_mask)
    
    combined_images: np.array = np.concatenate((images, np.array(aug_images)), axis=0)
    combined_masks: np.array = np.concatenate((masks, np.array(aug_masks)), axis=0)
    
    dataset = tf.data.Dataset.from_tensor_slices((combined_images, combined_masks))
    dataset = dataset.shuffle(buffer_size=len(combined_images), seed=42)
    batch_size: int = 4
    batched_dataset = dataset.batch(batch_size)
    
    return batched_dataset
