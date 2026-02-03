import numpy as np
from tqdm import tqdm
import rasterio as rio
import geopandas as gpd
from utils.dataloader import normalize_s2
from tqdm import tqdm
from rasterio.mask import mask
from tqdm import tqdm
import numpy as np

def save_raster(final_pred, reference_image_path, save_path, aoi_path):
    """Save the final prediction raster, optionally clipped to an AOI."""
    with rio.open(reference_image_path) as src:
        profile = src.profile.copy()
        transform = src.transform

    profile.update(
        dtype=np.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )
    if aoi_path:
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
        
        print(f"Classification map saved to: {save_path}")
    else:
        with rio.open(reference_image_path) as src:
            profile = src.profile.copy()
            profile.update(dtype=np.uint8, count=1, compress='lzw')
            with rio.open(save_path, 'w', **profile) as dst:
                dst.write(final_pred, 1)
        
        print(f"Classification map saved to: {save_path}")

class PatchGenerator:
    """Generates patches from multiple modalities for sliding window inference."""
    def __init__(self, image_dict, patch_height, patch_width, stride, batch_size):
        self.image_dict = image_dict
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride
        self.batch_size = batch_size
        self.modalities = list(image_dict.keys())
        self.coords = self._get_patch_coordinates()

    def _get_patch_coordinates(self):
        ref_image = list(self.image_dict.values())[0]
        ys = list(range(0, ref_image.shape[0] - self.patch_height + 1, self.stride))
        xs = list(range(0, ref_image.shape[1] - self.patch_width + 1, self.stride))
        return [(y, x) for y in ys for x in xs]

    def __len__(self):
        return int(np.ceil(len(self.coords) / self.batch_size))

    def __iter__(self):
        for i in range(0, len(self.coords), self.batch_size):
            batch_coords = self.coords[i : i + self.batch_size]
            batch_patches = {}
            for modality, image in self.image_dict.items():
                patches = np.array([
                    image[y:y+self.patch_height, x:x+self.patch_width] 
                    for y, x in batch_coords
                ])
                batch_patches[modality] = patches
            input_list = [batch_patches[modality] for modality in self.modalities]
            yield input_list, batch_coords


def hann_window(size):
    """Generates a 2D Hann window of given size."""
    hann_1d = np.hanning(size)  # 1D Hann window
    hann_2d = np.outer(hann_1d, hann_1d)  # Convert to 2D
    return hann_2d


def preprocess_modality(image, modality):
    """Preprocess input image based on modality."""
    if modality == 'S2':
        image = np.clip(image / 10000.0, 0.0, 1.0) if np.max(image) > 3.0 else image
    return image


#-------------------------------------------#
# inference code for Multi Modal MBCNN model
#-------------------------------------------#

def inference_mbcnn(model, image_sources, save_path, aoi_path=None, batch_size=8, stride_ratio=0.5):
    """inference using multi branch model."""
    # Load and preprocess input rasters
    rasters = [rio.open(src).read().transpose(1, 2, 0) for src in image_sources]
    rasters[0] = np.nan_to_num(rasters[0], nan=0.0)
    
    input1 = normalize_s2(rasters[0])  # Normalize Sentinel-2 input
    input2 = rasters[1]

    # print(f'Input 1 min: {np.min(input1)}, max: {np.max(input1)}')
    # print(f'Input 2 min: {np.min(input2)}, max: {np.max(input2)}')
    
    # print(f'S2 shape: {input1.shape}')
    # print(f'BD shape: {input2.shape}')

    patch_height, patch_width = model.inputs[0].shape[1:3]
    stride = int(patch_height * stride_ratio)
    image_height, image_width = input1.shape[:2]
    classes = model.outputs[0].shape[-1]  # assuming single task output

    y_pred = np.zeros((image_height, image_width, classes), dtype=np.float32)
    count_map = np.zeros((image_height, image_width, classes), dtype=np.float32)

    # create a dict with insertion order matching model input order
    image_dict = {"S2": input1, "BD": input2} 
    dataset = PatchGenerator(image_dict, patch_height, patch_width, stride, batch_size)
    window = hann_window(patch_height)[..., np.newaxis]  # Expand dims to match prediction shape

    pbar = tqdm(total=len(dataset), desc="Progress")

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
    print()

    # Compute final classification map
    averaged_predictions = np.divide(y_pred, count_map, out=np.zeros_like(y_pred), where=(count_map != 0))
    final_pred = np.argmax(averaged_predictions, axis=-1) + 1  # Ensure class indexing starts at 1

    # Save and clip the raster
    save_raster(final_pred, image_sources[0], save_path, aoi_path)