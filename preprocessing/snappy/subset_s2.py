import zipfile
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
import numpy as np
import argparse
import os
import geopandas as gpd
import rioxarray
from datetime import datetime
from tqdm import tqdm

def extract_date_from_filename(zip_path):
    """Extract year and month from Sentinel-2 zip filename."""
    filename = os.path.basename(zip_path)
    try:
        date_str = filename.split('_')[2][:8]  # Extract YYYYMMDD
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract date from filename: {filename}")

def find_band_in_zip(zip_ref, band_name, resolution):
    for file in zip_ref.namelist():
        if file.endswith(f"_{band_name}_{resolution}.jp2"):
            return file
    return None

def load_band_from_zip(zip_path, band_name, resolution):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        band_file = find_band_in_zip(zip_ref, band_name, resolution)
        if not band_file:
            raise FileNotFoundError(f"Band {band_name} at {resolution} not found in the zip file.")
        with zip_ref.open(band_file) as band_data:
            with rasterio.MemoryFile(band_data.read()) as memfile:
                with memfile.open() as src:
                    return src.read(1), src.profile

def resample_band(data, src_profile, target_profile):
    resampled_data = np.empty((target_profile['height'], target_profile['width']), dtype=data.dtype)
    reproject(
        source=data,
        destination=resampled_data,
        src_transform=src_profile['transform'],
        src_crs=src_profile['crs'],
        dst_transform=target_profile['transform'],
        dst_crs=target_profile['crs'],
        resampling=Resampling.bilinear
    )
    return resampled_data

def harmonize_data(raster_data):
    """Shift Sentinel-2 data values if it's from post-2022 scenes, avoiding uint16 underflow."""
    raster_data = raster_data.astype(np.int32)  # Convert to int32 to prevent underflow
    raster_data -= 1000
    raster_data = np.clip(raster_data, 0, 65535)  # Ensure valid range
    return raster_data.astype(np.uint16)  # Convert back to uint16


def clip_raster(aoi_path, mem_raster, output_raster, match_aoi_crs=True):
    city_bnd = gpd.read_file(aoi_path)
    raster = rioxarray.open_rasterio(mem_raster)

    # Decide which CRS to use
    if match_aoi_crs:
        raster = raster.rio.reproject(city_bnd.crs)
    else:
        city_bnd = city_bnd.to_crs(raster.rio.crs)

    aoi_bounds = city_bnd.total_bounds
    clipped_raster = raster.rio.clip_box(*aoi_bounds)
    clipped_raster.rio.to_raster(output_raster)
    # clipped_raster.rio.to_raster(output_raster, compress="lzw")



def process_sentinel2_zip(zip_path, output_path, aoi_path=None):
    bands = {
        "B02": "10m", "B03": "10m", "B04": "10m", "B05": "20m", "B06": "20m", 
        "B07": "20m", "B08": "10m", "B8A": "20m", "B11": "20m", "B12": "20m"
    }
    date_obj = extract_date_from_filename(zip_path)
    
    harmonize = date_obj > datetime(2022, 1, 25)
    if harmonize:
        print(f'Sentinel 2 acquisition date: {date_obj}')
        print('Bands will be harmonized to match with processing baseline 04.00')

    try:
        first_band_data, first_band_profile = load_band_from_zip(zip_path, "B02", "10m")
        target_profile = first_band_profile.copy()
        num_bands = len(bands)
        output_data = np.empty((num_bands, target_profile['height'], target_profile['width']), dtype=np.uint16)

        with tqdm(total=num_bands, desc="Processing bands", unit="band") as pbar:
            for i, (band_name, resolution) in enumerate(bands.items()):
                band_data, band_profile = load_band_from_zip(zip_path, band_name, resolution)
                resampled_data = resample_band(band_data, band_profile, target_profile) if resolution != "10m" else band_data

                if harmonize:
                    resampled_data = harmonize_data(resampled_data)
                
                output_data[i] = resampled_data.astype(np.uint16)
                pbar.update(1)

        target_profile.update(count=num_bands, dtype='uint16')

        # Store raster in memory instead of saving to disk
        with MemoryFile() as memfile:
            target_profile["driver"] = "GTiff"
            with memfile.open(**target_profile) as mem_raster:
                for i in range(num_bands):
                    mem_raster.write(output_data[i], i + 1)

                # Ensure mem_raster is closed before using in clip_raster
                mem_raster.close()

            # Now open it again to pass to clip_raster
            with memfile.open() as clipped_mem_raster:
                if aoi_path:
                    clip_raster(aoi_path, clipped_mem_raster, output_path)
                else:
                    with rasterio.open(output_path, 'w', **target_profile) as dst:
                        for i in range(num_bands):
                            dst.write(output_data[i], i + 1)

        print(f"Saved output to {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Sentinel-2 Level-2A zip files into GeoTIFFs with optional AOI clipping and harmonization.")
    parser.add_argument("--s2", type=str, required=True, help="Path to the Sentinel-2 zip file.")
    parser.add_argument("--save", type=str, required=True, help="Path to save the output GeoTIFF file.")
    parser.add_argument("--aoi", type=str, required=False, help="Path to AOI GeoJSON file for clipping (optional).")
    args = parser.parse_args()
    process_sentinel2_zip(args.s2, args.save, args.aoi)
