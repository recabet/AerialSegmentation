import numpy as np
import matplotlib.pyplot as plt


def display_image_and_mask (image: np.array, mask: np.array) -> None:
    """
    Displays an image and its corresponding mask side by side.

    Parameters:
    image (np.array): The image to be displayed.
    mask (np.array): The binary mask to be displayed, where white pixels represent the mask area.

    Returns:
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    plt.show()


def plot_metrics (iou: np.array,
                  val_iou: np.array,
                  accuracy: np.array,
                  val_accuracy: np.array,
                  precision: np.array,
                  val_precision: np.array,
                  recall: np.array,
                  val_recall: np.array,
                  f1: np.array,
                  val_f1: np.array,
                  loss: np.array,
                  val_loss: np.array) -> None:
    """
    Plots the training and validation metrics over epochs, including IoU, accuracy, precision,
    recall, F1 score, and loss.

    Parameters:
    iou (np.array): Array of training IoU scores over epochs.
    val_iou (np.array): Array of validation IoU scores over epochs.
    accuracy (np.array): Array of training accuracy scores over epochs.
    val_accuracy (np.array): Array of validation accuracy scores over epochs.
    precision (np.array): Array of training precision scores over epochs.
    val_precision (np.array): Array of validation precision scores over epochs.
    recall (np.array): Array of training recall scores over epochs.
    val_recall (np.array): Array of validation recall scores over epochs.
    f1 (np.array): Array of training F1 scores over epochs.
    val_f1 (np.array): Array of validation F1 scores over epochs.
    loss (np.array): Array of training loss values over epochs.
    val_loss (np.array): Array of validation loss values over epochs.

    Returns:
    None
    """
    plt.figure(figsize=(12, 10))
    
    if iou is not None and val_iou is not None:
        plt.subplot(2, 3, 1)
        plt.plot(iou, label='Training IoU')
        plt.plot(val_iou, label='Validation IoU')
        plt.title('IoU Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
    
    if accuracy is not None and val_accuracy is not None:
        plt.subplot(2, 3, 2)
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    
    if precision is not None and val_precision is not None:
        plt.subplot(2, 3, 3)
        plt.plot(precision, label='Training Precision')
        plt.plot(val_precision, label='Validation Precision')
        plt.title('Precision Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
    
    if recall is not None and val_recall is not None:
        plt.subplot(2, 3, 4)
        plt.plot(recall, label='Training Recall')
        plt.plot(val_recall, label='Validation Recall')
        plt.title('Recall Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
    
    if f1 is not None and val_f1 is not None:
        plt.subplot(2, 3, 5)
        plt.plot(f1, label='Training F1 Score')
        plt.plot(val_f1, label='Validation F1 Score')
        plt.title('F1 Score Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
    
    if loss is not None and val_loss is not None:
        plt.subplot(2, 3, 6)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
