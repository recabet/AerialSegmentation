import cv2
from typing import Tuple,List
import numpy as np
import os
import PIL.Image as Image

def load_images_masks(
    image_dir: str,
    mask_dir: str,
    img_size: Tuple[int, int] = (128, 128),
    remove_black_masks: bool = False
) -> Tuple[np.array, np.array]:

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


def load_single_image (image_path: str) -> np.array:
    return np.array(Image.open(image_path))


def load_single_mask (mask_path: str) -> np.array:
    return np.array(Image.open(mask_path).convert('L'))