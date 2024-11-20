import numpy as np


def f1_score(precision: np.array, recall: np.array) -> np.array:
    """
    Computes the F1 score for each pair of precision and recall values.

    The F1 score is the harmonic mean of precision and recall. It is given by the formula:
    F1 = 2 * (precision * recall) / (precision + recall)
    If both precision and recall are zero, the F1 score is set to zero to avoid division by zero.

    Parameters:
    precision (np.array): Array of precision values.
    recall (np.array): Array of recall values.

    Returns:
    np.array: An array of F1 scores, one for each pair of precision and recall values.
    """
    return [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]


def calculate_pixel_accuracy(predictions, ground_truth):
    """
    Calculates pixel-wise accuracy for white and black pixels.

    Args:
        predictions (np.ndarray): Predicted binary masks (0 or 1).
        ground_truth (np.ndarray): Ground truth binary masks (0 or 1).

    Returns:
        tuple: (white_pixel_accuracy, black_pixel_accuracy)
    """
    predicted_flat = predictions.flatten()
    ground_truth_flat = ground_truth.flatten()


    white_pixel_indices = (ground_truth_flat == 1)
    correct_white_pixels = np.sum(predicted_flat[white_pixel_indices] == ground_truth_flat[white_pixel_indices])
    total_white_pixels = np.sum(white_pixel_indices)
    white_pixel_accuracy = correct_white_pixels / total_white_pixels if total_white_pixels > 0 else 0

    black_pixel_indices = (ground_truth_flat == 0)
    correct_black_pixels = np.sum(predicted_flat[black_pixel_indices] == ground_truth_flat[black_pixel_indices])
    total_black_pixels = np.sum(black_pixel_indices)
    black_pixel_accuracy = correct_black_pixels / total_black_pixels if total_black_pixels > 0 else 0

    return white_pixel_accuracy, black_pixel_accuracy

def calculate_iou(predictions, ground_truth):
    """
    Calculates Intersection over Union (IoU) for white and black pixels.

    Args:
        predictions (np.ndarray): Predicted binary masks (0 or 1).
        ground_truth (np.ndarray): Ground truth binary masks (0 or 1).

    Returns:
        tuple: (iou_white, iou_black)
    """
    predicted_flat = predictions.flatten()
    ground_truth_flat = ground_truth.flatten()

    # White IoU
    true_positives = np.sum((predicted_flat == 1) & (ground_truth_flat == 1))
    false_positives = np.sum((predicted_flat == 1) & (ground_truth_flat == 0))
    false_negatives = np.sum((predicted_flat == 0) & (ground_truth_flat == 1))
    iou_white = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0

    # Black IoU
    true_negatives = np.sum((predicted_flat == 0) & (ground_truth_flat == 0))
    false_negatives_black = np.sum((predicted_flat == 1) & (ground_truth_flat == 0))
    false_positives_black = np.sum((predicted_flat == 0) & (ground_truth_flat == 1))
    iou_black = true_negatives / (true_negatives + false_negatives_black + false_positives_black) if (true_negatives + false_negatives_black + false_positives_black) > 0 else 0

    return iou_white, iou_black

def calculate_precision_recall(predictions, ground_truth):
    """
    Calculates precision and recall for white and black pixels.

    Args:
        predictions (np.ndarray): Predicted binary masks (0 or 1).
        ground_truth (np.ndarray): Ground truth binary masks (0 or 1).

    Returns:
        tuple: (precision_white, recall_white, precision_black, recall_black)
    """
    predicted_flat = predictions.flatten()
    ground_truth_flat = ground_truth.flatten()

    true_positives = np.sum((predicted_flat == 1) & (ground_truth_flat == 1))
    false_positives = np.sum((predicted_flat == 1) & (ground_truth_flat == 0))
    false_negatives = np.sum((predicted_flat == 0) & (ground_truth_flat == 1))
    precision_white = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall_white = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    true_negatives = np.sum((predicted_flat == 0) & (ground_truth_flat == 0))
    false_positives_black = np.sum((predicted_flat == 0) & (ground_truth_flat == 1))
    false_negatives_black = np.sum((predicted_flat == 1) & (ground_truth_flat == 0))
    precision_black = true_negatives / (true_negatives + false_positives_black) if (true_negatives + false_positives_black) > 0 else 0
    recall_black = true_negatives / (true_negatives + false_negatives_black) if (true_negatives + false_negatives_black) > 0 else 0

    return precision_white, recall_white, precision_black, recall_black
