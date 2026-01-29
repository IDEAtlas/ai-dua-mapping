import argparse
import geopandas as gpd
import rioxarray
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import box
from dask.diagnostics import ProgressBar
import dask
import gc
from rasterio.warp import transform_bounds


def create_reference(sentinel_raster, ghsl, duas, output):
    """
    This script processes geospatial data to create a reference label raster with three classes:
    1. Built-up areas based on GHSL built-up fraction data.
    2. Locations of DUAs (Deprived Urban Areas), represented as polygons.
    3. Non Built-up areas where neither condition is met.

    The output raster will have the same extent, resolution, and CRS as the input Sentinel-2 raster.

    Arguments:
    - `--sentinel`: Path to the Sentinel-2 raster file (spatial reference).
    - `--ghsl`: Path to the GHSL raster file.
    - `--duas`: Path to the DUAs GeoJSON file.
    - `--output`: Path to save the output GeoTIFF raster.
    """

    print("Starting processing...")
    
    # Configure dask for memory efficiency
    dask.config.set({'array.chunk-size': '64MiB'})
    
    pbar = ProgressBar()
    pbar.register()

    print("Reading Sentinel-2 raster to get reference extent and transform...")
    # Load Sentinel raster to get the reference grid
    sentinel = rioxarray.open_rasterio(sentinel_raster, chunks={"x": 512, "y": 512})
    
    # Get the spatial reference from Sentinel raster
    target_crs = sentinel.rio.crs
    target_transform = sentinel.rio.transform()
    target_bounds = sentinel.rio.bounds()
    target_height = sentinel.rio.height
    target_width = sentinel.rio.width
    
    print(f"Target raster properties:")
    print(f"  Shape: {sentinel.shape}")
    print(f"  CRS: {target_crs}")
    
    # We only need the spatial info, so free the data
    del sentinel
    gc.collect()

    print("Reading DUAs vector data...")
    duas = gpd.read_file(duas)

    # Ensure DUAs are in the same CRS as the target raster
    if duas.crs != target_crs:
        print(f"Reprojecting DUAs from {duas.crs} to {target_crs}")
        duas = duas.to_crs(target_crs)

    print("Opening GHSL raster...")
    raster = rioxarray.open_rasterio(ghsl, chunks={"x": 512, "y": 512})
    
    print(f"GHSL raster shape: {raster.shape}")
    print(f"GHSL raster CRS: {raster.rio.crs}")
    
    # If GHSL raster is in different CRS, transform bounds first
    if str(raster.rio.crs) != str(target_crs):
        ghsl_bounds = transform_bounds(
            target_crs, 
            raster.rio.crs, 
            *target_bounds
        )
        
        print("Clipping GHSL raster to target bounds...")
        clipped_raster = raster.rio.clip_box(*ghsl_bounds)
        
        # Free memory
        del raster
        gc.collect()

        print("Reprojecting GHSL raster to match Sentinel grid...")
        aligned_raster = clipped_raster.rio.reproject(
            target_crs,
            shape=(target_height, target_width),
            transform=target_transform
        )
    else:
        print("Clipping and resampling GHSL raster to match Sentinel grid...")
        aligned_raster = raster.rio.clip_box(*target_bounds).rio.reproject(
            target_crs,
            shape=(target_height, target_width),
            transform=target_transform
        )
    
    # Free memory
    if 'clipped_raster' in locals():
        del clipped_raster
    if 'raster' in locals():
        del raster
    gc.collect()

    print("Binarizing the GHSL raster (values > 15 -> 1, values <= 15 -> 0)...")
    aligned_raster = aligned_raster.fillna(0)
    binary_raster = aligned_raster.where(aligned_raster > 15, 0)
    binary_raster = binary_raster.where(binary_raster == 0, 1)
    binary_raster = binary_raster.compute()
    
    # Free memory
    del aligned_raster
    gc.collect()

    print("Rasterizing DUAs polygons...")
    
    # Check if there are any DUAs to rasterize
    if len(duas) == 0:
        print("No DUAs found, skipping rasterization...")
        polygon_mask = np.zeros((target_height, target_width), dtype="uint8")
    else:
        print(f"Rasterizing {len(duas)} DUA polygons...")
        polygon_mask = rasterize(
            [(geom, 2) for geom in duas.geometry],  # Assign value 2 to DUAS polygons
            out_shape=(target_height, target_width),
            transform=target_transform,
            fill=0,
            dtype="uint8",
            all_touched=False
        )

    binary_raster.values[0] = np.where(polygon_mask > 0, polygon_mask, binary_raster.values[0])
    binary_raster.values[0] = np.clip(binary_raster.values[0], 0, 2)
    
    # Free memory
    del polygon_mask
    gc.collect()

    print(f"Saving reference mask...")
    binary_raster = binary_raster.fillna(0)
    binary_raster.rio.write_nodata(None, inplace=True)
    binary_raster.rio.to_raster(output, compress="LZW", dtype="uint8")
    print(f"Processing complete. Output saved to {output}")
    
    # Print some statistics
    unique_values, counts = np.unique(binary_raster.values[0], return_counts=True)
    print("Class distribution:")
    total_pixels = np.sum(counts)
    for val, count in zip(unique_values, counts):
        percentage = (count / total_pixels) * 100
        if val == 0:
            print(f"  Class 0 (Non built-up): {count:,} pixels ({percentage:.2f}%)")
        elif val == 1:
            print(f"  Class 1 (Built-up): {count:,} pixels ({percentage:.2f}%)")
        elif val == 2:
            print(f"  Class 2 (DUAs): {count:,} pixels ({percentage:.2f}%)")
    
    print(f"Total pixels: {total_pixels:,}")
    print(f"Output raster dimensions: {target_width} x {target_height}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a reference label raster aligned with Sentinel-2 data.")
    parser.add_argument("--sentinel", required=True, help="Path to the Sentinel-2 raster file (used as spatial reference)")
    parser.add_argument("--ghsl", required=True, help="Path to the GHSL raster file")
    parser.add_argument("--duas", required=True, help="Path to the DUAS GeoJSON file")
    parser.add_argument("--output", required=True, help="Path to save the output raster")

    args = parser.parse_args()

    create_reference(args.sentinel, args.ghsl, args.duas, args.output)
