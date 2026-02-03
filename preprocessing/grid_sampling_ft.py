import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import random
import argparse
import pandas as pd
import warnings
import os


def clip_raster_with_aoi(raster_path: str, aoi_path: str) -> str:
    """
    Clip input raster to AOI extent and return the path to the clipped raster.
    """
    print(f"Clipping raster to AOI: {aoi_path}")
    aoi = gpd.read_file(aoi_path)

    with rasterio.open(raster_path) as src:
        aoi = aoi.to_crs(src.crs)
        out_image, out_transform = mask(src, aoi.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        clipped_raster_path = raster_path.replace(".tif", "_clipped.tif")
        with rasterio.open(clipped_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"AOI clipping completed. Saved to {clipped_raster_path}")
    return clipped_raster_path


def create_grid_with_ids_ft(raster_path: str, patch_size: int = 128, slum_class: int = 2) -> gpd.GeoDataFrame:
    """
    Create grid for fine-tuning: only keep patches with slum pixels (class 2).
    All kept patches are assigned to 'train' set.
    """
    with rasterio.open(raster_path) as src:
        height, width = src.shape
        crs = src.crs
        transform = src.transform

        grid_cells = []
        slum_patch_ids = []
        
        patch_idx = 0
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                win_height = min(patch_size, height - y)
                win_width = min(patch_size, width - x)
                window = Window(x, y, win_width, win_height)
                bounds = rasterio.windows.bounds(window, transform)
                
                # Read data to check for slum pixels
                data = src.read(1, window=window)
                valid_data = data[data >= 0]
                
                # Keep only patches with slum pixels (class 2)
                if len(valid_data) > 0 and slum_class in valid_data:
                    grid_cells.append(box(*bounds))
                    slum_patch_ids.append(patch_idx)
                    
                    # Get class distribution
                    classes, counts = np.unique(valid_data, return_counts=True)
                    class_count_dict = dict(zip(classes, counts))
                    slum_count = class_count_dict.get(slum_class, 0)
                    total = counts.sum()
                    slum_percentage = (slum_count / total) * 100
                    
                    if patch_idx % 100 == 0:
                        print(f"  Patch {patch_idx}: {slum_percentage:.1f}% slum pixels")
                
                patch_idx += 1

        print(f"\nTotal grid cells created: {patch_idx}")
        print(f"Patches with slum pixels: {len(grid_cells)}")
        
        grid = gpd.GeoDataFrame({
            'geometry': grid_cells,
            'patch_id': list(range(1, len(grid_cells) + 1)),
            'set': 'train'  # All patches go to train set
        }, crs=crs)

        return grid


def assign_randomized_ids(grid: gpd.GeoDataFrame, random_seed: int = None) -> gpd.GeoDataFrame:
    """Assign randomized patch IDs for reproducibility."""
    if random_seed is not None:
        random.seed(random_seed)

    ids = list(range(1, len(grid) + 1))
    random.shuffle(ids)

    grid = grid.copy()
    grid['patch_id'] = ids
    return grid


def print_summary(grid: gpd.GeoDataFrame):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("FINE-TUNING GRID SUMMARY")
    print("="*50)
    print(f"Total patches (with slum pixels): {len(grid)}")
    print(f"All patches assigned to: train")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Create sampling grid for fine-tuning (slum-focused, train-only)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', required=True, help='Input reference raster path')
    parser.add_argument('--aoi', required=False, help='AOI shapefile or GeoJSON path to constrain the raster extent')
    parser.add_argument('--output', required=True, help='Output path to GeoJSON grid')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size in pixels')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--slum_class', type=int, default=2, help='Class ID for slum pixels')
    args = parser.parse_args()

    # Step 1: AOI clipping (optional)
    if args.aoi:
        raster_path_to_use = clip_raster_with_aoi(args.input, args.aoi)
    else:
        raster_path_to_use = args.input

    print(f"Creating slum-focused grid (class {args.slum_class})...")
    grid = create_grid_with_ids_ft(raster_path_to_use, args.patch_size, args.slum_class)

    print("Assigning randomized IDs...")
    grid = assign_randomized_ids(grid, args.seed)

    print_summary(grid)

    print(f"\nSaving grid to {args.output}")
    grid_to_save = grid[['patch_id', 'set', 'geometry']]
    grid_to_save.to_file(args.output, driver='GeoJSON')

    print(f"Done. Created {len(grid_to_save)} patches (all slum-focused, train-only).")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
