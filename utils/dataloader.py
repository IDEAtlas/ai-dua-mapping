import tensorflow as tf
import os
import rasterio
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class CNNDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir, image_prefix="VHR_", mask_prefix="RF_", target_size=128):
        self.data_dir = data_dir
        self.image_prefix = image_prefix
        self.mask_prefix = mask_prefix
        self.target_size = target_size
        
        # Get all file pairs
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.startswith(image_prefix)])
        self.image_paths = [os.path.join(data_dir, f) for f in self.image_files]
        self.mask_paths = [os.path.join(data_dir, f.replace(image_prefix, mask_prefix)) for f in self.image_files]
        
    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and return a single sample"""
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        return self.load_single(image_path, mask_path)
    
    def load_single(self, image_path, mask_path):
        """Load single image-mask pair with rasterio"""
        # Load S2 image with rasterio
        with rasterio.open(image_path) as src:
            image = src.read()  # Reads all bands (10, H, W)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            image = tf.transpose(image, [1, 2, 0])  # Convert to (H, W, 10)
        
        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read()  # (1, H, W)
            mask = tf.convert_to_tensor(mask, dtype=tf.int32)
            mask = tf.transpose(mask, [1, 2, 0])  # Convert to (H, W, 1)
        
        # Normalize image
        # image = tf.cond(
        #     tf.reduce_max(image) > 3.0,
        #     lambda: image / 10000.0,
        #     lambda: image
        # )
        # image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Resize/Crop/Pad
        image = tf.image.resize_with_crop_or_pad(image, self.target_size, self.target_size)
        mask = tf.image.resize_with_crop_or_pad(mask, self.target_size, self.target_size)
        
        # One-hot encode mask
        mask = tf.squeeze(mask, axis=-1)
        mask = tf.one_hot(mask, depth=3, dtype=tf.float32)
        
        return image, mask
    
    def compute_class_weights(self):
        """
        Compute class weights for imbalanced data
        Returns: dict with class indices as keys and weights as values
        """
        all_labels = []
        
        # Collect all labels from the dataset
        for mask_path in self.mask_paths:
            with rasterio.open(mask_path) as src:
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
        
        # Convert to dictionary format (to be used directly in model.fit)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Convert to list format (to be used in some loss functions)
        class_weight_list = class_weights.tolist()
        print(f"Class weights list: {class_weight_list}")

        return class_weight_dict, class_weight_list

def loader(dataset, batch_size=4, shuffle=True):
    """Create tf.data.Dataset from RasterDataset"""
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: dataset,
        output_signature=(
            tf.TensorSpec(shape=(*dataset.target_size, 10), dtype=tf.float32),
            tf.TensorSpec(shape=(*dataset.target_size, 3), dtype=tf.float32)
        )
    )
    
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
    
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset



class MBCNNDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir, s2_prefix="S2_", bd_prefix="BD_", rf_prefix="RF_", target_size=128):
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
        """Load S2, BD as inputs and RF as mask"""
        # Load S2 image (10 bands)
        with rasterio.open(s2_path) as src:
            s2_image = src.read()  # (10, H, W)
            s2_image = tf.convert_to_tensor(s2_image, dtype=tf.float32)
            s2_image = tf.transpose(s2_image, [1, 2, 0])  # (H, W, 10)
        
        # Load Built-up Density (BD) - assuming 1 band
        with rasterio.open(bd_path) as src:
            bd_image = src.read()  # (1, H, W)
            bd_image = tf.convert_to_tensor(bd_image, dtype=tf.float32)
            bd_image = tf.transpose(bd_image, [1, 2, 0])  # (H, W, 1)
        
        # Load Reference mask (RF)
        with rasterio.open(rf_path) as src:
            rf_mask = src.read()  # (1, H, W)
            rf_mask = tf.convert_to_tensor(rf_mask, dtype=tf.int32)
            rf_mask = tf.transpose(rf_mask, [1, 2, 0])  # (H, W, 1)
        
        # Normalize S2 image
        s2_image = tf.cond(
            tf.reduce_max(s2_image) > 3.0,
            lambda: s2_image / 10000.0,
            lambda: s2_image
        )
        s2_image = tf.clip_by_value(s2_image, 0.0, 1.0)
        
        # Resize/Crop/Pad all to target size
        s2_image = tf.image.resize_with_crop_or_pad(s2_image, self.target_size, self.target_size)
        bd_image = tf.image.resize_with_crop_or_pad(bd_image, self.target_size, self.target_size)
        rf_mask = tf.image.resize_with_crop_or_pad(rf_mask, self.target_size, self.target_size)
        
        # One-hot encode RF mask
        rf_mask = tf.squeeze(rf_mask, axis=-1)
        rf_mask = tf.one_hot(rf_mask, depth=3, dtype=tf.float32)
        
        # Return as [input1, input2], output
        return (s2_image, bd_image), rf_mask
    
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
        print(f"Class weights list: {class_weight_list}")
        
        return class_weight_dict, class_weight_list

def mbcnn_loader(dataset, batch_size=16, shuffle=True):
    """Create tf.data.Dataset for multi-input model"""
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: dataset,
        output_signature=(
            # Use tuple for multiple inputs
            (
                tf.TensorSpec(shape=(*dataset.target_size, 10), dtype=tf.float32),
                tf.TensorSpec(shape=(*dataset.target_size, 1), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(*dataset.target_size, 3), dtype=tf.float32)
        )
    )
    
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
    
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset


class MTCNNDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir, target_size=None, input_modalities=None):
        self.data_dir = data_dir
        self.target_size = target_size
        self.input_modalities = input_modalities
        
        # Find all available files and use the first modality as reference
        if not self.input_modalities:
            raise ValueError("input_modalities cannot be empty")
        
        # Use the first modality as reference for file discovery
        reference_modality = self.input_modalities[0]
        reference_files = sorted([f for f in os.listdir(data_dir) 
                                if f.startswith(f'{reference_modality}_')])
        
        # Build file paths for all specified inputs
        self.file_paths = {}
        for modality in self.input_modalities:
            if modality == reference_modality:
                self.file_paths[modality] = [os.path.join(data_dir, f) for f in reference_files]
            else:
                # Derive other modality files from reference
                self.file_paths[modality] = [
                    os.path.join(data_dir, f.replace(f'{reference_modality}_', f'{modality}_')) 
                    for f in reference_files
                ]
        
        # Always include target paths (derive from reference modality)
        self.rf_paths = [
            os.path.join(data_dir, f.replace(f'{reference_modality}_', 'RF_')) 
            for f in reference_files
        ]
        self.bd_paths = [
            os.path.join(data_dir, f.replace(f'{reference_modality}_', 'BD_')) 
            for f in reference_files
        ]
        
        # Auto-detect image size from first file of reference modality
        if self.target_size is None:
            self.target_size = self._get_img_size(self.file_paths[reference_modality][0])
    
    def _get_img_size(self, file_path):
        """Detect image dimensions from file"""
        with rasterio.open(file_path) as src:
            height, width = src.shape
        return (height, width)
    
    def __len__(self):
        # Use reference modality for length
        reference_modality = self.input_modalities[0]
        return len(self.file_paths[reference_modality])
    
    def __getitem__(self, idx):
        """Get sample with inputs from config.DATASET"""
        inputs = []
        for modality in self.input_modalities:
            if modality == 'S1':
                inputs.append(self.load_s1(self.file_paths['S1'][idx]))
            elif modality == 'S2':
                inputs.append(self.load_s2(self.file_paths['S2'][idx]))
        
        rf = self.load_rf(self.rf_paths[idx])
        bd = self.load_bd(self.bd_paths[idx])
        
        return tuple(inputs), {'seg': rf, 'reg': bd}
    
    def load_s1(self, path):
        """S1 radar - 2 channels"""
        with rasterio.open(path) as src:
            img = src.read()
            img = tf.convert_to_tensor(img, dtype=tf.float32)
            img = tf.transpose(img, [1, 2, 0])
        
        if self.target_size:
            img = tf.image.resize(img, self.target_size)
        return img
    
    def load_s2(self, path):
        """S2 optical - 10 channels"""
        with rasterio.open(path) as src:
            img = src.read()
            img = tf.convert_to_tensor(img, dtype=tf.float32)
            img = tf.transpose(img, [1, 2, 0])
        
        # Normalize S2
        img = tf.cond(
            tf.reduce_max(img) > 3.0,
            lambda: img / 10000.0,
            lambda: img
        )
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        if self.target_size:
            img = tf.image.resize(img, self.target_size)
        return img
    
    def load_rf(self, path):
        """3-class segmentation mask"""
        with rasterio.open(path) as src:
            mask = src.read(1)
            mask = mask.astype(np.uint8)
            mask = tf.convert_to_tensor(mask, dtype=tf.uint8)
        
        mask = tf.expand_dims(mask, axis=-1)
        if self.target_size:
            mask = tf.image.resize(mask, self.target_size, method='nearest')
        mask = tf.squeeze(mask, axis=-1)
        mask = tf.cast(mask, tf.int32)
        mask = tf.one_hot(mask, depth=3, dtype=tf.float32)
        return mask
    
    def load_bd(self, path):
        """Building density mask"""
        with rasterio.open(path) as src:
            density = src.read(1)
            density = tf.convert_to_tensor(density, dtype=tf.float32)
        
        density = tf.expand_dims(density, axis=-1)
        if self.target_size:
            density = tf.image.resize(density, self.target_size)
        return density
    
    def _get_img_size(self, file_path):
        """Detect image dimensions from file"""
        with rasterio.open(file_path) as src:
            height, width = src.shape
        return (height, width)
    
    def __len__(self):
        reference_modality = self.input_modalities[0]
        return len(self.file_paths[reference_modality])
    
    def compute_class_weights(self):
        """Compute class weights for RF masks"""
        all_labels = []
        
        for rf_path in self.rf_paths:
            with rasterio.open(rf_path) as src:
                mask = src.read(1)
                mask = mask.astype(np.uint8)
                mask_flat = mask.flatten()
                all_labels.extend(mask_flat)
        
        all_labels = np.array(all_labels)
        classes = np.unique(all_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
        
        class_weight_dict = {int(cls): weight for cls, weight in zip(classes, class_weights)}
        class_weight_list = class_weights.tolist()
        print(f"Class weights list: {class_weight_list}")
        
        return class_weight_dict, class_weight_list
    
def create_dataloader(dataset, batch_size=8, shuffle=True):
    """Create tf.data pipeline - now dynamic based on dataset inputs"""
    
    # Determine output signature dynamically
    input_signature = []
    for modality in dataset.input_modalities:
        if modality == 'S1':
            input_signature.append(tf.TensorSpec(shape=(*dataset.target_size, 2), dtype=tf.float32))
        elif modality == 'S2':
            input_signature.append(tf.TensorSpec(shape=(*dataset.target_size, 10), dtype=tf.float32))
    
    output_signature = (
        tuple(input_signature),  # Dynamic inputs
        {
            'seg': tf.TensorSpec(shape=(*dataset.target_size, 3), dtype=tf.float32),
            'reg': tf.TensorSpec(shape=(*dataset.target_size, 1), dtype=tf.float32)
        }
    )
    
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: dataset,
        output_signature=output_signature
    )
    
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
    
    return tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



# load data on memory
import os
import numpy as np
import rasterio as rio
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from glob import glob


def norm_s2(s2):
    if np.max(s2) > 3:
        s2 = s2 / 10000.0
    return np.clip(s2, 0, 1)

def load_data(img_dir:str, patch_size:int, prefix:str, n_classes:int=None):
    images = []
    image_paths = sorted(glob(os.path.join(img_dir, f"{prefix}*.tif")))
    for img_path in image_paths:
        ds = rio.open(img_path)
        data = ds.read(window=((0, patch_size), (0, patch_size)))
        data = np.moveaxis(data, 0, -1)
        data = np.nan_to_num(data, nan=0.0)
                
        if 'S2' in img_path and np.max(data) > 3:
            data = norm_s2(data)
        if n_classes:
            data = to_categorical(data, n_classes)
        images.append(data)
        ds.close()
    return np.array(images)


def calculate_class_weights(one_hot_encoded_masks):
    # Convert one-hot encoded mask to single class labels
    mask_labels = np.argmax(one_hot_encoded_masks, axis=-1).flatten()
    class_labels = np.unique(mask_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=mask_labels)
    
    return class_weights



import os
import numpy as np
import tensorflow as tf
import rasterio
from tqdm import tqdm


class VHRDataset(tf.keras.utils.Sequence):
    """
    TensorFlow data loader for VHR (Pleiades) segmentation patches.
    Streams image/label pairs directly from disk, normalizes to [0,1],
    and can compute class weights from all labels in the folder.
    """

    def __init__(self, data_dir, batch_size=8, n_classes=3, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle

        # Pair VHR_ and RF_ files
        self.vhr_files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith("VHR_") and f.endswith(".tif")]
        )
        self.rf_files = [f.replace("VHR_", "RF_") for f in self.vhr_files]

        self.samples = [
            (os.path.join(data_dir, v), os.path.join(data_dir, r))
            for v, r in zip(self.vhr_files, self.rf_files)
            if os.path.exists(os.path.join(data_dir, r))
        ]

        if len(self.samples) == 0:
            raise ValueError(f"No valid VHR/RF pairs found in {data_dir}")

        # infer patch shape
        with rasterio.open(self.samples[0][0]) as src:
            self.target_size = (src.height, src.width, src.count)

        self.on_epoch_end()

    # ----------------------------------------------------------
    #  core keras sequence methods
    # ----------------------------------------------------------
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs, batch_lbls = [], []

        for i in batch_indices:
            vhr_path, rf_path = self.samples[i]

            with rasterio.open(vhr_path) as vsrc:
                img = vsrc.read().astype(np.float32)
                img = np.transpose(img, (1, 2, 0))
            img = np.clip(img / 10000.0, 0, 1)

            with rasterio.open(rf_path) as rsrc:
                label = rsrc.read(1).astype(np.uint8)

            label_oh = tf.one_hot(label, depth=self.n_classes, dtype=tf.float32)
            batch_imgs.append(img)
            batch_lbls.append(label_oh)

        X = np.stack(batch_imgs)
        y = np.stack(batch_lbls)
        return X, y

    # ----------------------------------------------------------
    #  class weight computation
    # ----------------------------------------------------------
    def compute_class_weights(self, normalize=True):
        """
        Compute class weights based on pixel frequencies
        across all reference patches in the dataset.

        Returns:
            dict  e.g. {0: w0, 1: w1, 2: w2}
        """
        print(f"Computing class weights from {len(self.samples)} label patches...")
        counts = np.zeros(self.n_classes, dtype=np.int64)

        for _, rf_path in tqdm(self.samples, desc="Scanning labels", ncols=90):
            with rasterio.open(rf_path) as src:
                label = src.read(1)
            for c in range(self.n_classes):
                counts[c] += np.count_nonzero(label == c)

        total = counts.sum()
        if total == 0:
            raise ValueError("No labeled pixels found.")

        # Inverse frequency weighting
        weights = total / (self.n_classes * counts.astype(np.float64))
        if normalize:
            weights = weights / weights.sum() * self.n_classes

        class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

        print("Pixel counts per class:", counts.tolist())
        print("Computed class weights:", class_weight_dict)
        return class_weight_dict
