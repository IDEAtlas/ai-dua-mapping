import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric for multi-class segmentation.
    Computes F1 as the harmonic mean of precision and recall.
    """
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state with predictions and ground truth."""
        # Convert one-hot to class predictions
        y_pred_class = tf.argmax(y_pred, axis=-1)
        y_true_class = tf.argmax(y_true, axis=-1)
        
        # Flatten
        y_pred_flat = tf.reshape(y_pred_class, [-1])
        y_true_flat = tf.reshape(y_true_class, [-1])
        
        # Compute TP, FP, FN
        tp = tf.reduce_sum(tf.cast(
            tf.logical_and(
                tf.equal(y_pred_flat, y_true_flat),
                tf.not_equal(y_true_flat, 0)  # Exclude background class
            ), 
            tf.float32
        ))
        
        fp = tf.reduce_sum(tf.cast(
            tf.logical_and(
                tf.not_equal(y_pred_flat, y_true_flat),
                tf.not_equal(y_pred_flat, 0)
            ),
            tf.float32
        ))
        
        fn = tf.reduce_sum(tf.cast(
            tf.logical_and(
                tf.not_equal(y_pred_flat, y_true_flat),
                tf.not_equal(y_true_flat, 0)
            ),
            tf.float32
        ))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        """Compute and return F1 score."""
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1
    
    def reset_state(self):
        """Reset metric state."""
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


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

def report(test_loader, model):
    """Compute metrics from a tf.data.Dataset loader.
    
    Args:
        test_loader: tf.data.Dataset yielding (inputs, labels) tuples
        model: Trained model for prediction
    
    Returns:
        class_report, cm, iou_values, mean_iou_value, fwiou_value
    """
        
    # Collect all batches from dataset
    test_images_list = []
    test_labels_list = []
    
    for batch_inputs, batch_labels in test_loader:
        test_images_list.append(batch_inputs)
        test_labels_list.append(batch_labels)
    
    # Concatenate all batches into single arrays
    if isinstance(test_images_list[0], (list, tuple)):
        # Multi-input case (S2 + BD)
        test_images = [tf.concat([b[i] for b in test_images_list], axis=0).numpy() 
                       for i in range(len(test_images_list[0]))]
    else:
        # Single input case
        test_images = tf.concat(test_images_list, axis=0).numpy()
    
    test_labels = tf.concat(test_labels_list, axis=0).numpy()
    
    # Predict and compute metrics
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