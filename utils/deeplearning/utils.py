import tensorflow as tf
import numpy as np


class LRWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, target, start=0.0, verbose=0):
        super(LRWarmup, self).__init__()
        self.steps = 0
        self.warmup_steps = warmup_steps
        self.target = target
        self.start = start
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        self.steps = self.steps + 1

    def on_batch_begin(self, batch, logs=None):
        if self.steps <= self.warmup_steps:
            lr = (self.target - self.start) * (self.steps / self.warmup_steps) + self.start
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            if self.verbose > 0:
                print(f"\nLRWarmup callback: set learning rate to {lr}")


def gaussian_kernel(size, mu=0, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x**2 + y**2)
    kernel = np.exp(-(distance - mu)**2 / 2/ sigma**2) / np.sqrt(2 / np.pi) / sigma
    return kernel
                
                
def apply(features, model, patch_size=384, batch_size=4, n_outputs=2):
    height, width, _ = features.shape
    weighted_prob = np.zeros((height, width, n_outputs))
    weights = gaussian_kernel(patch_size)[..., np.newaxis]
    counts = np.zeros((height, width, 1))

    patches = []
    n_patches = 0
    rows_cols = []

    row = 0
    while row + patch_size <= height:
        col = 0 
        while col + patch_size <= width:
            patches.append(features[row:row + patch_size, col:col + patch_size, :])
            rows_cols.append((row, col))
            n_patches += 1

            if n_patches >= batch_size:
                patch_probs = model.predict(np.array(patches), verbose=0)

                for (row_pred, col_pred), patch_prob in zip(rows_cols, patch_probs):
                    weighted_prob[
                        row_pred:row_pred + patch_size, 
                        col_pred:col_pred + patch_size, :
                    ] += (weights * patch_prob)
                    counts[
                        row_pred:row_pred + patch_size, 
                        col_pred:col_pred + patch_size, :
                    ] += weights

                patches = []
                n_patches = 0
                rows_cols = []

            col += (patch_size // 2)
        row += (patch_size // 2)
    
    if n_patches > 0:
        patch_probs = model.predict(np.array(patches), verbose=0)

        for (row_pred, col_pred), patch_prob in zip(rows_cols, patch_probs):
            weighted_prob[
                row_pred:row_pred + patch_size, 
                col_pred:col_pred + patch_size, :
            ] += (weights * patch_prob)
            counts[
                row_pred:row_pred + patch_size, 
                col_pred:col_pred + patch_size, :
            ] += weights

    prob = weighted_prob / counts
    return prob

from tqdm import tqdm
from tensorflow.keras.callbacks import Callback

class training_progress(Callback):
    def __init__(self, total_epochs, steps_per_epoch):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.progress_bar = tqdm(total=total_epochs, desc="Training progress", position=0, leave=True)
    
    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix({
            'Epoch': f"{epoch+1}/{self.total_epochs}",
            'loss': f"{logs.get('loss'):.4f}",
            'IoU': f"{logs.get('f1'):.4f}",
            'val_loss': f"{logs.get('val_loss'):.4f}",
            'val_IoU': f"{logs.get('val_f1'):.4f}"
        })
    
    def on_train_end(self, logs=None):
        self.progress_bar.close()
