"""
grid_sampling.py

This module creates a stratified sampling grid from labelled reference raster data.
It generates grid patches, assigns class labels based on raster values, performs stratified train/val/test splits,
and saves the resulting grid as a GeoJSON file.

Functions:
    - create_grid_with_ids: Generates grid cells and assigns class labels.
    - assign_randomized_ids: Randomizes patch IDs for unbiased sampling.
    - stratified_split: Splits the grid into train/val/test sets based on stratification.
    - print_class_distribution: Prints class distribution statistics.
    - main: Command-line interface for running the grid sampling process.

Usage:
    python grid_sampling.py --input <input_raster.tif> --output <output_grid.geojson> [options]
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import random
import argparse
import pandas as pd
import warnings

def create_grid_with_ids(raster_path: str, patch_size: int = 128, class2_proportion: float = 15) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        height, width = src.shape
        crs = src.crs
        transform = src.transform

        grid_cells = []
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                win_height = min(patch_size, height - y)
                win_width = min(patch_size, width - x)
                window = Window(x, y, win_width, win_height)
                bounds = rasterio.windows.bounds(window, transform)
                grid_cells.append(box(*bounds))

        grid = gpd.GeoDataFrame({
            'geometry': grid_cells,
            'temp_id': range(1, len(grid_cells) + 1),
            'class': -1,
            'has_class_2': False,
            'class_counts': None,
            'strat_class': -1
        }, crs=crs)

        with rasterio.open(raster_path) as src:
            for idx, geom in enumerate(grid.geometry):
                window = src.window(*geom.bounds)
                data = src.read(1, window=window)
                valid_data = data[data >= 0]
                if len(valid_data) > 0:
                    classes, counts = np.unique(valid_data, return_counts=True)
                    total = counts.sum()
                    class_count_dict = dict(zip(classes, counts))
                    dominant_class = classes[np.argmax(counts)]
                    class_2_pct = class_count_dict.get(2, 0) / total

                    grid.at[idx, 'class'] = int(dominant_class)
                    grid.at[idx, 'has_class_2'] = 2 in class_count_dict
                    grid.at[idx, 'class_counts'] = {int(k): int(v) for k, v in class_count_dict.items()}
                    
                    # Assign stratification class: 2 if ≥class2_proportion of class 2, else dominant class
                    if class_2_pct >= class2_proportion / 100:
                        grid.at[idx, 'strat_class'] = 2
                    else:
                        grid.at[idx, 'strat_class'] = int(dominant_class)

        return grid[grid['class'] != -1]  # Remove empty patches

def assign_randomized_ids(grid: gpd.GeoDataFrame, random_seed: int = None) -> gpd.GeoDataFrame:
    if random_seed is not None:
        random.seed(random_seed)
    
    ids = list(range(1, len(grid) + 1))
    random.shuffle(ids)

    grid = grid.sort_values('temp_id').copy()
    grid['patch_id'] = ids
    return grid.drop(columns=['temp_id'])

def stratified_split(
    grid: gpd.GeoDataFrame,
    stratify_col: str = 'strat_class',
    splits: tuple = (0.7, 0.15, 0.15)
) -> gpd.GeoDataFrame:
    if not np.isclose(sum(splits), 1.0):
        raise ValueError("Split proportions must sum to 1.0")
    
    grid = grid.copy()
    grid['set'] = ''
    
    for value, group in grid.groupby(stratify_col):
        indices = group.index.tolist()
        random.shuffle(indices)

        n = len(indices)
        n_train = round(splits[0] * n)
        n_val = round(splits[1] * n)
        n_test = n - n_train - n_val

        grid.loc[indices[:n_train], 'set'] = 'train'
        grid.loc[indices[n_train:n_train + n_val], 'set'] = 'val'
        grid.loc[indices[n_train + n_val:], 'set'] = 'test'

    return grid

def print_class_distribution(grid: gpd.GeoDataFrame):
    print("\n📊 Stratified class distribution (dominant OR ≥X% class 2):")
    dist_counts = grid.groupby(['set', 'strat_class']).size().unstack(fill_value=0)
    dist_percent = dist_counts.div(dist_counts.sum(axis=1), axis=0).multiply(100).round(1)
    dist_combined = dist_counts.astype(str) + ' (' + dist_percent.astype(str) + '%)'
    print(dist_combined.fillna("-").to_string())

    print("\n🔍 Patches that CONTAIN any class 2 pixels (even if not dominant):")
    contains_class2 = grid[grid['has_class_2']]
    presence_counts = contains_class2.groupby('set').size()
    print(presence_counts.to_string())

def main():
    parser = argparse.ArgumentParser(
        description='Create stratified sampling grid with randomized IDs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', required=True, help='Input reference raster path')
    parser.add_argument('--output', required=True, help='Output path to GeoJSON grid')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size in pixels')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/val/test split proportions')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--class2_proportion', type=float, default=1, 
                        help='Proportion of class 2 pixels (in %) for inclusion in stratification')
    args = parser.parse_args()

    print("Creating initial grid...")
    grid = create_grid_with_ids(args.input, args.patch_size, args.class2_proportion)
    
    print("Assigning randomized IDs...")
    grid = assign_randomized_ids(grid, args.seed)

    print("Performing stratified split using enhanced class label (≥{:.1f}% class 2)...".format(args.class2_proportion))
    grid = stratified_split(grid, stratify_col='strat_class', splits=args.splits)
    
    print_class_distribution(grid)

    print(f"\nSaving to {args.output}")
    grid.to_file(args.output, driver='GeoJSON')
    print(f"✅ Done. Created {len(grid)} patches.")

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

