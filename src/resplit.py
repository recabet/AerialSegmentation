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
)->None:
    """
    Splits a dataset into training, validation, and test sets, and copies the images and labels into corresponding directories.

    Parameters:
    dataset_path (str): Path to the original dataset. It should contain subdirectories 'train', 'val', and 'test',
                         each containing 'image' and 'label' directories with the corresponding data files.
    output_path (str): Path where the split dataset will be saved. Subdirectories for 'train', 'val', and 'test' will be created
                       along with 'images' and 'labels' directories within each.
    train_ratio (float, optional): Proportion of the data to be used for training. Defaults to 0.7.
    val_ratio (float, optional): Proportion of the data to be used for validation. Defaults to 0.2.
    test_ratio (float, optional): Proportion of the data to be used for testing. Defaults to 0.1.
    random_state (int, optional): Seed for the random number generator used in splitting the dataset. Defaults to 42.

    Returns:
    None: The function splits the dataset and saves the split files to the specified output path.
    """
    assert round(train_ratio + val_ratio + test_ratio) == 1, "Ratios must sum to 1."
    
    # Create necessary directories for the split data
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)
    
    image_files = []
    label_files = []
    
    # Collect all image and label files
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
    
    def __copy_files (files, dest_dir):
        """
        Copies the specified files to the given destination directory.

        Parameters:
        files (list): List of file paths to be copied.
        dest_dir (str): The destination directory where the files will be copied.

        Returns:
        None
        """
        for f in files:
            shutil.copy(f, dest_dir)
    

    __copy_files(_train_images, os.path.join(output_path, "train", "images"))
    __copy_files(_train_labels, os.path.join(output_path, "train", "labels"))
    __copy_files(_val_images, os.path.join(output_path, "val", "images"))
    __copy_files(_val_labels, os.path.join(output_path, "val", "labels"))
    __copy_files(_test_images, os.path.join(output_path, "test", "images"))
    __copy_files(_test_labels, os.path.join(output_path, "test", "labels"))
