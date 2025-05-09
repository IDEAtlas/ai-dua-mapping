import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

class IoU(tf.keras.metrics.Metric):
    def __init__(self, class_idx=0, name="IoU", **kwargs):
        super(IoU, self).__init__(name=name, **kwargs)
        self.class_idx = class_idx
        self.iou = self.add_weight(name="iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)

        y_pred = tf.cast(y_pred == self.class_idx, self.dtype)
        y_true = tf.cast(y_true == self.class_idx, self.dtype)

        axis = [_ for _ in range(1, len(y_pred.shape))]
        tp = tf.reduce_sum(y_true * y_pred, axis=axis)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=axis)
        epsilon = tf.keras.backend.epsilon()
        values = (tp + epsilon) / (tp + fp + fn + epsilon)

        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)

        self.iou.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.iou / self.count

    def reset_state(self):
        self.iou.assign(0.0)
        self.count.assign(0.0)


def class_report(test_images, test_labels, model):
    y_pred = model.predict(test_images, batch_size=4)
    
    # Decode the predictions and labels from one-hot encoding
    y_pred_labels = np.argmax(y_pred, axis=-1)
    y_true_labels = np.argmax(test_labels, axis=-1)
    
    y_true_flattened = y_true_labels.flatten()
    y_pred_flattened = y_pred_labels.flatten()

    class_report = classification_report(y_true_flattened, y_pred_flattened, zero_division=0, target_names=['NBUA', 'NDUA', 'DUA'])
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_flattened, y_pred_flattened)
    
    return class_report, cm


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
    """Generate classification report, IoU, Mean IoU, and confusion matrix for model predictions."""
    
    # assert len(test_images.shape) == 4, "Expected test_images to be 4D (batch, height, width, channels)"
    # assert len(test_labels.shape) == 4, "Expected test_labels to be 4D (batch, height, width, classes)"
    
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
def class_report_mtcnn(test_images, test_masks, model):
    # print(f'Shape of testing image and mask: {test_images.shape}, {test_masks.shape}')
    # Since model has two outputs, y_pred will be a list of two arrays
    # We are interested in the classification output, which is the second element
    y_pred = model.predict(test_images, batch_size=4)
    y_pred_classification = y_pred[1]
    y_pred_labels = np.argmax(y_pred_classification, axis=3)
    y_true_flattened = test_masks.flatten()
    y_pred_flattened = y_pred_labels.flatten()
    class_report = classification_report(y_true_flattened, y_pred_flattened, zero_division=0, target_names = ['NBUA', 'NDUA', 'DUA'])
    cm = confusion_matrix(y_true_flattened, y_pred_flattened)
    return class_report, cm

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