import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score


# def class_report(test_images, test_labels, model):
#     y_pred = model.predict(test_images, batch_size=4)
    
#     # Decode the predictions and labels from one-hot encoding
#     y_pred_labels = np.argmax(y_pred, axis=-1)
#     y_true_labels = np.argmax(test_labels, axis=-1)
    
#     y_true_flattened = y_true_labels.flatten()
#     y_pred_flattened = y_pred_labels.flatten()

#     class_report = classification_report(y_true_flattened, y_pred_flattened, zero_division=0, target_names=['NBUA', 'NDUA', 'DUA'])
#     cm = confusion_matrix(y_true_flattened, y_pred_flattened)
    
#     return class_report, cm


def compute_iou(cm):
    """Compute IoU for each class from the confusion matrix."""
    intersection = np.diag(cm)  # True Positives
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection  # TP + FP + FN
    iou = intersection / np.maximum(union, 1)  # Avoid division by zero
    return iou

def compute_fwiou(cm):
    """Compute Frequency Weighted IoU (FWIoU)."""
    total_pixels = np.sum(cm)  # Total number of pixels
    class_freqs = np.sum(cm, axis=1) / total_pixels  # Frequency of each class
    iou_values = compute_iou(cm)
    fwiou = np.sum(class_freqs * iou_values)  # Weighted sum of IoU values
    return fwiou

def compute_mean_iou(cm):
    """Compute Mean IoU across all classes."""
    iou_values = compute_iou(cm)
    mean_iou = np.mean(iou_values)  # Average of IoU values
    return mean_iou

def class_report(test_images, test_labels, model):
    y_pred = model.predict(test_images, batch_size=4)
    y_pred_labels = np.argmax(y_pred.squeeze(), axis=-1)
    y_true_labels = np.argmax(test_labels.squeeze(), axis=-1)
    
    y_true_flattened = y_true_labels.flatten()
    y_pred_flattened = y_pred_labels.flatten()

    class_report = classification_report(y_true_flattened, y_pred_flattened, zero_division=0, target_names=['NBA', 'NDUA', 'DUA'])
    cm = confusion_matrix(y_true_flattened, y_pred_flattened)

    iou_values = compute_iou(cm)
    fwiou_value = compute_fwiou(cm)
    mean_iou_value = compute_mean_iou(cm)

    return class_report, cm, iou_values, mean_iou_value, fwiou_value

#For the multitask model
def class_report_mtcnn(test_loader, model):
    y_true = []
    y_pred = []

    for inputs, targets in test_loader:
        _, pred_seg = model.predict(inputs, verbose=0)

        pred_labels = np.argmax(pred_seg, axis=-1)
        true_labels = np.argmax(targets['seg'], axis=-1)

        y_pred.extend(pred_labels.reshape(-1))
        y_true.extend(true_labels.reshape(-1))

    class_report = classification_report(y_true, y_pred, zero_division=0, target_names=['NBUA', 'NDUA', 'DUA'])
    cm = confusion_matrix(y_true, y_pred)

    iou_values = compute_iou(cm)
    fwiou_value = compute_fwiou(cm)
    mean_iou_value = compute_mean_iou(cm)

    return class_report, cm, iou_values, mean_iou_value, fwiou_value

#  #generate classification report on big patches dividing them into smaller patches
def class_report_large_patches(test_images, test_masks, model, SIZE_X, SIZE_Y):
    PATCH_SIZE = (SIZE_X, SIZE_Y)
    print(PATCH_SIZE)
    # print(f'Shape of testing image and mask: {test_images.shape}, {test_masks.shape}')
    predicted_labels = np.zeros(test_masks.shape)
    # num_patches = (test_images.shape[1] // PATCH_SIZE[0]) * (test_images.shape[2] // PATCH_SIZE[1])
    for i in range(0, test_images.shape[1], PATCH_SIZE[0]):
        for j in range(0, test_images.shape[2], PATCH_SIZE[1]):
            patch = test_images[:, i:i+PATCH_SIZE[0], j:j+PATCH_SIZE[1], :]
            patch_pred = model.predict(patch, batch_size=4)
            patch_labels = np.argmax(patch_pred, axis=-1)
            patch_labels = np.expand_dims(patch_labels, axis=3)
            predicted_labels[:, i:i+PATCH_SIZE[0], j:j+PATCH_SIZE[1]] = patch_labels
    y_true_flattened = test_masks.flatten()
    y_pred_flattened = predicted_labels.flatten()
    class_report = classification_report(y_true_flattened, y_pred_flattened, zero_division=0, target_names = ['NBUA', 'NDUA', 'DUA'])
    cm = confusion_matrix(y_true_flattened, y_pred_flattened)
    return class_report, cm