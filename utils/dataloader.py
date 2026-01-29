import tensorflow as tf
import os
import rasterio
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def normalize_s2(s2_data):
    """Normalize Sentinel-2 data to [0, 1] range"""
    if np.max(s2_data) > 3:
        s2_data = s2_data / 10000.0
    return np.clip(s2_data, 0, 1).astype(np.float32)


class MBCNNDataset(tf.keras.utils.Sequence):
    """Dataset for MBCNN model - loads patches for training"""
    
    def __init__(self, data_dir, s2_prefix="S2_", bd_prefix="BD_", rf_prefix="RF_", target_size=128, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.s2_prefix = s2_prefix
        self.bd_prefix = bd_prefix
        self.rf_prefix = rf_prefix
        self.target_size = target_size
        
        # Get all file pairs using S2 as reference
        self.s2_files = sorted([f for f in os.listdir(data_dir) if f.startswith(s2_prefix)])
        self.s2_paths = [os.path.join(data_dir, f) for f in self.s2_files]
        self.bd_paths = [os.path.join(data_dir, f.replace(s2_prefix, bd_prefix)) for f in self.s2_files]
        self.rf_paths = [os.path.join(data_dir, f.replace(s2_prefix, rf_prefix)) for f in self.s2_files]
        
    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.s2_paths)
    
    def __getitem__(self, idx):
        """Load and return a sample with multiple inputs"""
        s2_path = self.s2_paths[idx]
        bd_path = self.bd_paths[idx]
        rf_path = self.rf_paths[idx]
        return self.load_multi_input(s2_path, bd_path, rf_path)
    
    def load_multi_input(self, s2_path, bd_path, rf_path):
        """Load S2, BD as inputs and RF as mask - reference preprocessing"""
        with rasterio.open(s2_path) as src:
            s2_image = src.read()  
            s2_image = s2_image.astype(np.float32)
            s2_image = normalize_s2(s2_image) 
            
            # Handle NaN values - replace with 0 (valid after normalization to [0,1])
            if np.isnan(s2_image).any():
                s2_image = np.nan_to_num(s2_image, nan=0.0)
            
            # Transpose to (H, W, 10) THEN convert to tensor
            s2_image = np.transpose(s2_image, [1, 2, 0])
            s2_image = tf.convert_to_tensor(s2_image, dtype=tf.float32)
        
        # Load Built-up Density (BD) - assuming 1 band
        with rasterio.open(bd_path) as src:
            bd_image = src.read()  # (1, H, W)
            bd_image = bd_image.astype(np.float32)
            # Handle NaN values
            if np.isnan(bd_image).any():
                bd_image = np.nan_to_num(bd_image, nan=0.0)
            bd_image = np.transpose(bd_image, [1, 2, 0])  # (H, W, 1)
            bd_image = tf.convert_to_tensor(bd_image, dtype=tf.float32)
        
        # Load Reference mask (RF)
        with rasterio.open(rf_path) as src:
            rf_mask = src.read()  # (1, H, W)
            rf_mask = rf_mask.astype(np.int32)
            rf_mask = np.transpose(rf_mask, [1, 2, 0])  # (H, W, 1)
            rf_mask = tf.convert_to_tensor(rf_mask, dtype=tf.int32)
        
        # Resize/Crop/Pad all to target size
        s2_image = tf.image.resize_with_crop_or_pad(s2_image, self.target_size, self.target_size)
        bd_image = tf.image.resize_with_crop_or_pad(bd_image, self.target_size, self.target_size)
        rf_mask = tf.image.resize_with_crop_or_pad(rf_mask, self.target_size, self.target_size)
        
        # One-hot encode RF mask
        rf_mask = tf.squeeze(rf_mask, axis=-1)
        rf_mask = tf.one_hot(rf_mask, depth=3, dtype=tf.float32)
        
        # Return as [input1, input2], output
        return [s2_image, bd_image], rf_mask
    
    def compute_class_weights(self):
        """
        Compute class weights for RF masks
        Returns: tuple (class_weight_dict, class_weight_list)
        """
        all_labels = []
        
        # Collect all labels from the RF masks
        for rf_path in self.rf_paths:
            with rasterio.open(rf_path) as src:
                mask = src.read()  # (1, H, W)
                mask_flat = mask.flatten()  # Flatten to 1D array
                all_labels.extend(mask_flat)
        
        # Convert to numpy array
        all_labels = np.array(all_labels)
        
        # Compute class weights
        classes = np.unique(all_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=all_labels
        )
        
        # Convert to both formats
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        class_weight_list = class_weights.tolist()
        # print(f"Class weights list: {class_weight_list}")
        
        return class_weight_dict, class_weight_list


def mbcnn_loader(dataset, batch_size=16, shuffle=True):
    """
    Return a Keras Sequence for proper compatibility with model.fit().
    This ensures correct data handling and fixes NaN issues in training.
    """
    class MBCNNSequence(tf.keras.utils.Sequence):
        def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
            super().__init__(**kwargs)
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = np.arange(len(dataset))
            if self.shuffle:
                np.random.shuffle(self.indices)
        
        def __len__(self):
            """Return number of batches per epoch"""
            return int(np.ceil(len(self.dataset) / self.batch_size))
        
        def __getitem__(self, batch_idx):
            """Get a batch of data"""
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(self.dataset))
            batch_indices = self.indices[start_idx:end_idx]
            
            s2_batch = []
            bd_batch = []
            rf_batch = []
            
            for idx in batch_indices:
                (s2, bd), rf = self.dataset[int(idx)]
                s2_batch.append(s2)
                bd_batch.append(bd)
                rf_batch.append(rf)
            
            # Stack into batches
            s2_batch = tf.stack(s2_batch, axis=0)
            bd_batch = tf.stack(bd_batch, axis=0)
            rf_batch = tf.stack(rf_batch, axis=0)
            
            # Return as tuple of (inputs, targets) for model.fit()
            return (s2_batch, bd_batch), rf_batch
        
        def on_epoch_end(self):
            """Shuffle indices after each epoch"""
            if self.shuffle:
                np.random.shuffle(self.indices)
    
    return MBCNNSequence(dataset, batch_size, shuffle)
