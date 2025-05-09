import tensorflow as tf
import numpy as np
import rasterio
import os

class MyTFDataset(tf.data.Dataset):
    def _generator(data_dir, image_files, patch_size, apply_transform):
        for img_file in image_files:
            image_path = os.path.join(data_dir, img_file)
            mask_path = os.path.join(data_dir, img_file.replace('S2_', 'RF_'))

            with rasterio.open(image_path) as src:
                window = rasterio.windows.Window(0, 0, patch_size, patch_size)
                image = src.read(window=window).astype(np.float32)
                image = np.moveaxis(image, 0, -1)  # (H, W, C)

            with rasterio.open(mask_path) as src:
                window = rasterio.windows.Window(0, 0, patch_size, patch_size)
                mask = src.read(1, window=window).astype(np.uint8)

            if image.max() > 3:
                image = image / 10000.0

            # Apply transform (assumes albumentations or similar)
            if apply_transform:
                augmented = apply_transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            yield image.astype(np.float32), mask.astype(np.uint8)

    def __new__(cls, data_dir, transform=None, patch_size=256):
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.tif') and f.startswith('S2_')]
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(patch_size, patch_size, None), dtype=tf.float32),
                tf.TensorSpec(shape=(patch_size, patch_size), dtype=tf.uint8)
            ),
            args=(data_dir, image_files, patch_size, transform)
        )



# Example usage: LOAD TRAINING VALIDATION AND TEST DATA THAT IS ALREADY SPLIT

# TRAIN_DIR = '/path/to/train'
# VAL_DIR = '/path/to/val'
# TEST_DIR = '/path/to/test'
# batch_size = 32

# # Assuming you have a transform function defined

# buffer_size = 10 * batch_size
# dataset_train = MyTFDataset(TRAIN_DIR, transform=None, patch_size=256)
# dataset_val = MyTFDataset(VAL_DIR, transform=None, patch_size=256)
# dataset_test = MyTFDataset(TEST_DIR, transform=None, patch_size=256)

# TRAIN_LOADER = dataset_train.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# VAL_LOADER = dataset_val.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# TEST_LOADER = dataset_test.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
