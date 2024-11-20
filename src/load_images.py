import cv2
from typing import Tuple, List
import numpy as np
import os
import PIL.Image as Image

def load_images_masks(
    image_dir: str,
    mask_dir: str,
    img_size: Tuple[int, int] = (128, 128),
    remove_black_masks: bool = False
) -> Tuple[np.array, np.array]:
    """
    Loads and preprocesses images and their corresponding masks from directories.

    Images are resized to the specified size, normalized to a range of [0, 1], and
    optionally filtered to remove masks that are completely black.

    Args:
        image_dir (str): Directory containing input images.
        mask_dir (str): Directory containing corresponding masks.
        img_size (Tuple[int, int], optional): Target size for resizing images and masks.
            Defaults to (128, 128).
        remove_black_masks (bool, optional): If True, excludes pairs where the mask
            contains no non-zero pixels. Defaults to False.

    Returns:
        Tuple[np.array, np.array]: A tuple containing:
            - `images` (np.array): Preprocessed images with shape (N, H, W, C).
            - `masks` (np.array): Corresponding binary masks with shape (N, H, W, 1).
    """
    image_list: List = []
    mask_list: List = []

    for image_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name[:-4] + "_vis.tif")

        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0

        if not remove_black_masks or np.any(mask > 0):
            image_list.append(img)
            mask_list.append(mask)

    images = np.array(image_list)
    masks = np.array(mask_list).reshape(-1, img_size[0], img_size[1], 1)

    return images, masks


def load_single_image(image_path: str) -> np.array:
    """
    Loads a single image from a file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: The loaded image as a NumPy array.
    """
    return np.array(Image.open(image_path))


def load_single_mask(mask_path: str) -> np.array:
    """
    Loads a single mask from a file path and converts it to grayscale.

    Args:
        mask_path (str): Path to the mask file.

    Returns:
        np.array: The loaded mask as a NumPy array in grayscale.
    """
    return np.array(Image.open(mask_path).convert('L'))
