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


def create_grid_with_ids(raster_path: str, patch_size: int = 128, class2_proportion: float = 0) -> gpd.GeoDataFrame:
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

        for idx, geom in enumerate(grid.geometry):
            window = src.window(*geom.bounds)
            data = src.read(1, window=window)
            valid_data = data[data >= 0]
            if len(valid_data) > 0:
                classes, counts = np.unique(valid_data, return_counts=True)
                total = counts.sum()
                class_count_dict = dict(zip(classes, counts))
                dominant_class = classes[np.argmax(counts)]

                grid.at[idx, 'class'] = int(dominant_class)
                grid.at[idx, 'has_class_2'] = 2 in class_count_dict
                grid.at[idx, 'class_counts'] = {int(k): int(v) for k, v in class_count_dict.items()}

                if class2_proportion > 0:
                    class_2_pct = class_count_dict.get(2, 0) / total
                    if class_2_pct >= class2_proportion / 100:
                        grid.at[idx, 'strat_class'] = 2
                    else:
                        grid.at[idx, 'strat_class'] = int(dominant_class)
                else:
                    if 2 in class_count_dict:
                        grid.at[idx, 'strat_class'] = 2
                    else:
                        grid.at[idx, 'strat_class'] = int(dominant_class)

        return grid[grid['class'] != -1]


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


def filter_and_balance(
    grid: gpd.GeoDataFrame,
    minority_class: int = 2,
    single_class_threshold: float = 0.75,
    oversample_factor: int = 1,
) -> gpd.GeoDataFrame:
    filtered = []
    for idx, row in grid.iterrows():
        counts = row['class_counts']
        if counts is None:
            continue

        total = sum(counts.values())
        dominant_fraction = max(counts.values()) / total

        if row['strat_class'] == minority_class or row['has_class_2']:
            filtered.append(row)
        elif dominant_fraction <= single_class_threshold:
            filtered.append(row)

    filtered_grid = gpd.GeoDataFrame(filtered, crs=grid.crs)

    train_mask = (filtered_grid['set'] == 'train') & (filtered_grid['strat_class'] == minority_class)
    minority_train = filtered_grid[train_mask]
    
    if oversample_factor > 1 and not minority_train.empty:
        oversampled = pd.concat([minority_train] * (oversample_factor - 1), ignore_index=True)
        balanced = pd.concat([filtered_grid, oversampled], ignore_index=True)
    else:
        balanced = filtered_grid.copy()

    return balanced


def print_class_distribution(grid: gpd.GeoDataFrame):
    print("\n Stratified class distribution (dominant OR any class2):")
    dist_counts = grid.groupby(['set', 'strat_class']).size().unstack(fill_value=0)
    dist_percent = dist_counts.div(dist_counts.sum(axis=1), axis=0).multiply(100).round(1)
    dist_combined = dist_counts.astype(str) + ' (' + dist_percent.astype(str) + '%)'
    print(dist_combined.fillna("-").to_string())


def main():
    parser = argparse.ArgumentParser(
        description='Create stratified sampling grid with AOI clipping and randomized IDs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', required=True, help='Input reference raster path')
    parser.add_argument('--aoi', required=False, help='AOI shapefile or GeoJSON path to constrain the raster extent')
    parser.add_argument('--output', required=True, help='Output path to GeoJSON grid')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size in pixels')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/val/test split proportions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--class2_proportion', type=float, default=0.1,
                        help='Minimum percentage of pixels to consider for DUA existence in a patch')
    parser.add_argument('--oversample_factor', type=int, default=1, help='Oversampling factor for minority class')
    parser.add_argument('--filter_threshold', type=float, default=0.95,
                        help='Drop patches if dominant class covers more than this fraction')
    args = parser.parse_args()

    # Step 1: AOI clipping (optional)
    if args.aoi:
        raster_path_to_use = clip_raster_with_aoi(args.input, args.aoi)
    else:
        raster_path_to_use = args.input

    print("Creating initial grid...")
    grid = create_grid_with_ids(raster_path_to_use, args.patch_size, args.class2_proportion)

    print("Assigning randomized IDs...")
    grid = assign_randomized_ids(grid, args.seed)

    print(f"Performing stratified split using splits {args.splits}...")
    grid = stratified_split(grid, splits=tuple(args.splits))

    print("Filtering and balancing patches...")
    grid = filter_and_balance(grid, minority_class=2,
                              single_class_threshold=args.filter_threshold,
                              oversample_factor=args.oversample_factor)

    print_class_distribution(grid)

    print(f"\nSaving grid to {args.output}")
    grid_to_save = grid[['patch_id', 'set', 'geometry']]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    grid_to_save.to_file(args.output, driver='GeoJSON')

    print(f"Done. Created {len(grid_to_save)} patches (after filtering and balancing).")


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()