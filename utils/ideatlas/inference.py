# import os
# import numpy as np
# from tqdm import tqdm
# import rasterio as rio
# from glob import glob
# import geopandas as gpd
# from utils.ideatlas.data_utils import *
# from tqdm import tqdm
# from rasterio.mask import mask

# def mtcnn_inference(N_CLASSES, raster_paths, ndbi_path, model, prediction_path, vector_path):
#     src = rio.open(raster_paths)
#     ndbi_src = rio.open(ndbi_path)

#     ds = src.read()
#     ndbi_ds = ndbi_src.read(1)  # Assuming NDBI is a single band raster

#     large_image = np.moveaxis(ds, 0, -1)
#     large_ndbi = ndbi_ds

#     patch_height, patch_width = model.inputs[0].shape[1], model.inputs[0].shape[2]
#     stride = int(patch_height / 4)  # Set the desired overlap between patches
#     image_height, image_width = large_image.shape[:2]
#     y_pred = np.zeros((image_height, image_width, N_CLASSES))
#     count_map = np.zeros((image_height, image_width, N_CLASSES))

#     total_iterations = ((image_height - patch_height + 1) // stride) * ((image_width - patch_width + 1) // stride)
#     pbar = tqdm(total=total_iterations, desc="Running Full Inference:")

#     for y in range(0, image_height - patch_height + 1, stride):
#         for x in range(0, image_width - patch_width + 1, stride):
#             patch = large_image[y:y + patch_height, x:x + patch_width]
#             ndbi_patch = large_ndbi[y:y + patch_height, x:x + patch_width]

#             # Expand dimensions for model input
#             input_patch = np.expand_dims(patch, axis=0)
#             input_ndbi_patch = np.expand_dims(ndbi_patch, axis=(0, -1))

#             # Predict using the model
#             patch_predictions = model.predict([input_patch, input_ndbi_patch], verbose=0)

#             # Assume patch_predictions[1] is for segmentation task
#             y_pred[y:y + patch_height, x:x + patch_width] += patch_predictions[1][0]
#             count_map[y:y + patch_height, x:x + patch_width] += 1

#             pbar.update(1)
    
#     pbar.close()
#     averaged_predictions = y_pred / count_map
#     final_pred = np.argmax(averaged_predictions, axis=-1)

#     # Clip the predicted raster using the vector boundary
#     def clip_raster_with_vector(raster_array, profile, vector_path, output_path):
#         with rio.Env():
#             with rio.open(output_path, 'w', **profile) as dst:
#                 dst.write(raster_array.astype(rio.int8), 1)
#             clipper = gpd.read_file(vector_path)
#             with rio.open(output_path) as src:
#                 clipped, clipped_transform = mask(src, clipper.geometry, crop=True)
#                 profile.update({
#                     'height': clipped.shape[1],
#                     'width': clipped.shape[2],
#                     'transform': clipped_transform
#                 })
#                 with rio.open(output_path, 'w', **profile) as dst:
#                     dst.write(clipped)

#     # Save the predicted array as a georeferenced raster and then clip
#     with rio.Env():
#         profile = src.profile
#         profile.update(
#             dtype=rio.int8,
#             count=1,
#             width=final_pred.shape[-1],
#             height=final_pred.shape[-2],
#             transform=src.transform,
#             compress='lzw'
#         )
#         temp_pred_path = 'temp_prediction.tif'
#         with rio.open(temp_pred_path, 'w', **profile) as dst:
#             dst.write(final_pred.astype(rio.int8), 1)

#         # Clip the temporary prediction raster using the city boundary
#         clip_raster_with_vector(final_pred, profile, vector_path, prediction_path)

#     print(f"Prediction map saved to {prediction_path}")


# def full_inference_mbcnn(N_CLASSES, image_sources, model, save_path):

#     # Load and preprocess input rasters
#     rasters = [rio.open(src).read().transpose(1, 2, 0) for src in image_sources]
#     input1 = norm_s2(rasters[0])  # Normalize first input
#     input2 = rasters[1]           # Second input as is
       
#     print(f'Input 1 shape is: {input1.shape}')
#     print(f'Input 2 shape is: {input2.shape}')
       
#     patch_height, patch_width = model.inputs[0].shape[1:3]
#     stride = patch_height // 2
#     image_height, image_width = input1.shape[:2]

#     y_pred = np.zeros((image_height, image_width, N_CLASSES))
#     count_map = np.zeros((image_height, image_width, N_CLASSES))
    
#     total_iterations = ((image_height - patch_height + 1) // stride) * ((image_width - patch_width + 1) // stride)
#     pbar = tqdm(total=total_iterations, desc="Running Full Inference:")
    
#     for y in range(0, image_height - patch_height + 1, stride):
#         for x in range(0, image_width - patch_width + 1, stride):
#             msi_patch = input1[y:y+patch_height, x:x+patch_width]
#             pbd_patch = input2[y:y+patch_height, x:x+patch_width]
#             input_patch = [np.expand_dims(msi_patch, axis=0), np.expand_dims(pbd_patch, axis=0)]
#             patch_predictions = model.predict(input_patch, verbose=0)
#             y_pred[y:y+patch_height, x:x+patch_width] += patch_predictions[0]
#             count_map[y:y+patch_height, x:x+patch_width] += 1
#             pbar.update(1)
    
#     pbar.close()
    
#     averaged_predictions = y_pred / (count_map + 1e-8)
#     final_pred = np.argmax(averaged_predictions, axis=-1)
#     final_pred = final_pred + 1
#     # Save the predicted array as a georeferenced raster
#     with rio.Env():
#         profile = rio.open(image_sources[0]).profile
#         profile.update(dtype=rio.float32, count=1, compress='lzw')
        
#         with rio.open(save_path + '_map.tif', 'w', **profile) as dst:
#             dst.write(final_pred.astype(rio.int8), 1)
    
#         with rio.open(save_path + '_prob.tif', 'w', **profile) as dst:
#             dst.write(averaged_predictions[:, :, 2], 1) #index 2 saves only slum class

#     print(f"Prediction map saved to {save_path}")



# from tensorflow.keras.utils import Sequence

# def hann_window(size):
#     hann = np.hanning(size)
#     window = np.outer(hann, hann)
#     return window

# class PatchGenerator(Sequence):
#     def __init__(self, input1, input2, patch_height, patch_width, stride, batch_size):
#         self.input1 = input1
#         self.input2 = input2
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.stride = stride
#         self.batch_size = batch_size
#         self.image_height, self.image_width = input1.shape[:2]
#         self.patches = self._create_patches()

#     def _create_patches(self):
#         patches = []
#         for y in range(0, self.image_height - self.patch_height + 1, self.stride):
#             for x in range(0, self.image_width - self.patch_width + 1, self.stride):
#                 msi_patch = self.input1[y:y+self.patch_height, x:x+self.patch_width]
#                 pbd_patch = self.input2[y:y+self.patch_height, x:x+self.patch_width]
#                 patches.append((msi_patch, pbd_patch, y, x))
#         return patches

#     def __len__(self):
#         return int(np.ceil(len(self.patches) / self.batch_size))

#     def __getitem__(self, idx):
#         batch_patches = self.patches[idx * self.batch_size:(idx + 1) * self.batch_size]
#         msi_patches = np.array([p[0] for p in batch_patches])
#         pbd_patches = np.array([p[1] for p in batch_patches])
#         ys = [p[2] for p in batch_patches]
#         xs = [p[3] for p in batch_patches]
#         return [msi_patches, pbd_patches], (ys, xs)

# def full_inference_mbcnn2(N_CLASSES, image_sources, model, save_path, batch_size=8):
#     # Load and preprocess input rasters
#     rasters = [rio.open(src).read().transpose(1, 2, 0) for src in image_sources]
#     input1 = norm_s2(rasters[0])  # Normalize first input
#     input2 = rasters[1]           # Second input as is
       
#     print(f'Input 1 shape is: {input1.shape}')
#     print(f'Input 2 shape is: {input2.shape}')
       
#     patch_height, patch_width = model.inputs[0].shape[1:3]
#     stride = patch_height // 2
#     image_height, image_width = input1.shape[:2]
    
#     y_pred = np.zeros((image_height, image_width, N_CLASSES))
#     count_map = np.zeros((image_height, image_width, N_CLASSES))
    
#     dataset = PatchGenerator(input1, input2, patch_height, patch_width, stride, batch_size)
    
#     window = hann_window(patch_height)
#     window = np.expand_dims(window, axis=-1)  # Expand dimensions to match the prediction shape
    
#     pbar = tqdm(total=len(dataset), desc="Running Full Inference:")
    
#     for batch in dataset:
#         input_patches, (ys, xs) = batch
#         batch_predictions = model.predict(input_patches, verbose=0)
        
#         for i in range(len(batch_predictions)):
#             y, x = ys[i], xs[i]
#             patch_prediction = batch_predictions[i] * window
#             y_pred[y:y+patch_height, x:x+patch_width] += patch_prediction
#             count_map[y:y+patch_height, x:x+patch_width] += window
#         pbar.update(1)
    
#     pbar.close()
#     # Safely divide to avoid division by zero
#     averaged_predictions = np.divide(y_pred, count_map, out=np.zeros_like(y_pred), where=(count_map != 0))
#     final_pred = np.argmax(averaged_predictions, axis=-1)
#     final_pred = final_pred + 1

#     # averaged_predictions = y_pred / (count_map + 1e-8)
#     # final_pred = np.argmax(averaged_predictions, axis=-1)
    
#     # aoi = gpd.read_file('/data/training_data/city_boundary/mexico_city_admin_boundary.geojson')

#     with rio.Env():
#         profile = rio.open(image_sources[0]).profile
#         profile.update(dtype=rio.float32, count=1, compress='lzw')
        
#         with rio.open(save_path + '_map.tif', 'w', **profile) as dst:
#             dst.write(final_pred.astype(rio.int8), 1)
    
#         with rio.open(save_path + '_prob.tif', 'w', **profile) as dst:
#             dst.write(averaged_predictions[:, :, 2], 1) #index 2 saves only slum class

#     print(f"Prediction map saved to {save_path}")

import os
import numpy as np
from tqdm import tqdm
import rasterio as rio
from glob import glob
import geopandas as gpd
from dataloaders.data_utils import *
from tqdm import tqdm
from rasterio.mask import mask


def save_and_clip_prediction_as_raster(final_pred, reference_image_path, save_path, aoi_path):
    """
    Save the predicted array as a georeferenced raster and clip it using a city boundary.

    Parameters:
    final_pred (numpy.ndarray): The final prediction array to be saved.
    reference_image_path (str): Path to the reference image to copy metadata.
    save_path (str): Path where the output raster will be saved.
    aoi_path (str): Path to the AOI boundary file (GeoJSON).

    """

    with rio.open(reference_image_path) as src:
        profile = src.profile.copy()
        transform = src.transform

    profile.update(
        dtype=np.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    aoi = gpd.read_file(aoi_path)
    aoi = aoi.to_crs(profile["crs"]) 

    with rio.MemoryFile() as memfile:
        with memfile.open(**profile) as mem_raster:
            mem_raster.write(final_pred, 1)
            clipped_image, clipped_transform = mask(mem_raster, aoi.geometry, crop=True)

    profile.update({
        "height": clipped_image.shape[1],
        "width": clipped_image.shape[2],
        "transform": clipped_transform
    })

    with rio.open(save_path, "w", **profile) as dst:
        dst.write(clipped_image[0], 1)

    print(f"Classified raster saved at: {save_path}")

class PatchGenerator:
    def __init__(self, input1, input2, patch_height, patch_width, stride, batch_size):
        self.input1 = input1
        self.input2 = input2
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride
        self.batch_size = batch_size
        self.coords = self._get_patch_coordinates()  # Store as a single list of (y, x) tuples

    def _get_patch_coordinates(self):
        """Generate top-left coordinates for patches."""
        ys = list(range(0, self.input1.shape[0] - self.patch_height + 1, self.stride))
        xs = list(range(0, self.input1.shape[1] - self.patch_width + 1, self.stride))
        return [(y, x) for y in ys for x in xs]  # Returns a list of (y, x) pairs

    def __len__(self):
        return int(np.ceil(len(self.coords) / self.batch_size))

    def __iter__(self):
        for i in range(0, len(self.coords), self.batch_size):
            batch_coords = self.coords[i : i + self.batch_size]
            patches1 = np.array([self.input1[y:y+self.patch_height, x:x+self.patch_width] for y, x in batch_coords])
            patches2 = np.array([self.input2[y:y+self.patch_height, x:x+self.patch_width] for y, x in batch_coords])
            yield [patches1, patches2], batch_coords  # Now returns patches + list of (y, x) coordinates


def hann_window(size):
    """Generates a 2D Hann window of given size."""
    hann_1d = np.hanning(size)  # 1D Hann window
    hann_2d = np.outer(hann_1d, hann_1d)  # Convert to 2D
    return hann_2d

def mtcnn_inference(N_CLASSES, raster_paths, pbd_path, model, prediction_path, vector_path):
    src = rio.open(raster_paths)
    pbd_src = rio.open(pbd_path)

    s2 = src.read()
    pbd = pbd_src.read(1)

    large_s2 = np.moveaxis(s2, 0, -1)
    large_pbd = pbd

    patch_height, patch_width = model.inputs[0].shape[1], model.inputs[0].shape[2]
    stride = int(patch_height / 4)  # Set the desired overlap between patches
    image_height, image_width = large_s2.shape[:2]
    y_pred = np.zeros((image_height, image_width, N_CLASSES))
    count_map = np.zeros((image_height, image_width, N_CLASSES))

    total_iterations = ((image_height - patch_height + 1) // stride) * ((image_width - patch_width + 1) // stride)
    pbar = tqdm(total=total_iterations, desc="Running Full Inference:")

    for y in range(0, image_height - patch_height + 1, stride):
        for x in range(0, image_width - patch_width + 1, stride):
            patch = large_s2[y:y + patch_height, x:x + patch_width]
            ndbi_patch = large_pbd[y:y + patch_height, x:x + patch_width]

            # Expand dimensions for model input
            input_patch = np.expand_dims(patch, axis=0)
            input_ndbi_patch = np.expand_dims(ndbi_patch, axis=(0, -1))

            # Predict using the model
            patch_predictions = model.predict([input_patch, input_ndbi_patch], verbose=0)

            # Assume patch_predictions[1] is for segmentation task
            y_pred[y:y + patch_height, x:x + patch_width] += patch_predictions[1][0]
            count_map[y:y + patch_height, x:x + patch_width] += 1

            pbar.update(1)
    
    pbar.close()
    averaged_predictions = y_pred / count_map
    final_pred = np.argmax(averaged_predictions, axis=-1)

    save_and_clip_prediction_as_raster(final_pred, raster_paths, prediction_path, vector_path)


from tqdm import tqdm
import numpy as np

def full_inference_mbcnn(N_CLASSES, image_sources, model, save_path, aoi_path, batch_size=32):
    """
    Perform full-image inference using a sliding window approach.

    Parameters:
    N_CLASSES (int): Number of output classes.
    image_sources (list): List of input image file paths.
    model (tf.keras.Model): Trained deep learning model.
    save_path (str): Output file path for the raster.
    aoi_path (str): Path to the AOI boundary (GeoJSON).
    batch_size (int): Number of patches processed per batch.

    Returns:
    None
    """
    # Load and preprocess input rasters
    rasters = [rio.open(src).read().transpose(1, 2, 0) for src in image_sources]
    input1 = norm_s2(rasters[0])  # Normalize Sentinel-2 input
    input2 = rasters[1]           # Second input as is

    # print(f'S2 shape: {input1.shape}')
    # print(f'PBD shape: {input2.shape}')
    
    patch_height, patch_width = model.inputs[0].shape[1:3]
    stride = patch_height // 2
    image_height, image_width = input1.shape[:2]

    y_pred = np.zeros((image_height, image_width, N_CLASSES), dtype=np.float32)
    count_map = np.zeros((image_height, image_width, N_CLASSES), dtype=np.float32)

    dataset = PatchGenerator(input1, input2, patch_height, patch_width, stride, batch_size)
    window = hann_window(patch_height)[..., np.newaxis]  # Expand dims to match prediction shape

    pbar = tqdm(total=len(dataset), desc="Running Full Inference:")

    for batch in dataset:
        input_patches, batch_coords = batch  # batch_coords is a list of (y, x) pairs

        batch_predictions = model.predict(input_patches, verbose=0)

        for i in range(len(batch_predictions)):
            y, x = batch_coords[i]  # Correctly extract (y, x)
            patch_prediction = batch_predictions[i] * window
            y_pred[y:y+patch_height, x:x+patch_width] += patch_prediction
            count_map[y:y+patch_height, x:x+patch_width] += window

        
        pbar.update(1)

    pbar.close()

    # Compute final classification map
    averaged_predictions = np.divide(y_pred, count_map, out=np.zeros_like(y_pred), where=(count_map != 0))
    final_pred = np.argmax(averaged_predictions, axis=-1) + 1  # Ensure class indexing starts at 1

    # Save and clip the raster
    save_and_clip_prediction_as_raster(final_pred, image_sources[0], save_path, aoi_path)
