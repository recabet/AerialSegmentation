import os
import shutil
from sklearn.model_selection import train_test_split


def resplit_dataset (
        dataset_path,
        output_path,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_state=42
):
    assert round(train_ratio + val_ratio + test_ratio) == 1

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)

    image_files = []
    label_files = []
    for split in ["train", "val", "test"]:
        image_dir = os.path.join(dataset_path, split, "image")
        label_dir = os.path.join(dataset_path, split, "label")
        for file in os.listdir(image_dir):
            if file.endswith(".tif"):
                label_file = file.replace(".tif", "_vis.tif")
                label_path = os.path.join(label_dir, label_file)
                if os.path.exists(label_path):
                    image_files.append(os.path.join(image_dir, file))
                    label_files.append(label_path)

    _train_images, _temp_images, _train_labels, _temp_labels = train_test_split(
        image_files, label_files, test_size=(1 - train_ratio), random_state=random_state
    )
    _val_images, _test_images, _val_labels, _test_labels = train_test_split(
        _temp_images, _temp_labels, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state
    )

    def copy_files (files, dest_dir):
        for f in files:
            shutil.copy(f, dest_dir)

    copy_files(_train_images, os.path.join(output_path, "train", "images"))
    copy_files(_train_labels, os.path.join(output_path, "train", "labels"))
    copy_files(_val_images, os.path.join(output_path, "val", "images"))
    copy_files(_val_labels, os.path.join(output_path, "val", "labels"))
    copy_files(_test_images, os.path.join(output_path, "test", "images"))
    copy_files(_test_labels, os.path.join(output_path, "test", "labels"))


