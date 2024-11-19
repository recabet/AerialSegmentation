import numpy as np
import albumentations as A
import tensorflow as tf

def augment_single(image, mask):
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

def augment_dataset(images, masks):
    aug_images = []
    aug_masks = []
    for img, mask in zip(images, masks):
        aug_img, aug_mask = augment_single(img, mask)
        aug_images.append(aug_img)
        aug_masks.append(aug_mask)

    combined_images = np.concatenate((images, np.array(aug_images)), axis=0)
    combined_masks = np.concatenate((masks, np.array(aug_masks)), axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((combined_images, combined_masks))
    dataset = dataset.shuffle(buffer_size=len(combined_images), seed=42)
    batch_size = 4
    batched_dataset = dataset.batch(batch_size)

    return batched_dataset

