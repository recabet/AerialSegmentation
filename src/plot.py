import numpy as np
import matplotlib.pyplot as plt

def display_image_and_mask (image: np.array, mask: np.array) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    plt.show()
    
    
def plot_metrics(iou:np.array,
                 val_iou:np.array,
                 accuracy:np.array,
                 val_accuracy:np.array,
                 precision:np.array,
                 val_precision:np.array,
                 recall:np.array,
                 val_recall:np.array,
                 f1:np.array,
                 val_f1:np.array,
                 loss:np.array,
                 val_loss:np.array)->None:
    
    plt.figure(figsize=(12, 10))
    if iou and val_iou:
        plt.subplot(2, 3, 1)
        plt.plot(iou, label='Training IoU')
        plt.plot(val_iou, label='Validation IoU')
        plt.title('IoU Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
    if accuracy and val_accuracy:
        plt.subplot(2, 3, 2)
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    if precision and val_precision:
        plt.subplot(2, 3, 3)
        plt.plot(precision, label='Training Precision')
        plt.plot(val_precision, label='Validation Precision')
        plt.title('Precision Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
    if recall and val_recall:
        plt.subplot(2, 3, 4)
        plt.plot(recall, label='Training Recall')
        plt.plot(val_recall, label='Validation Recall')
        plt.title('Recall Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
    if f1 and val_f1:
        plt.subplot(2, 3, 5)
        plt.plot(f1, label='Training F1 Score')
        plt.plot(val_f1, label='Validation F1 Score')
        plt.title('F1 Score Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
    if loss and val_loss:
        plt.subplot(2, 3, 6)
        plt.plot(loss, label='Training Loss Score')
        plt.plot(val_loss, label='Validation Loss Score')
        plt.title('Loss Score Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
