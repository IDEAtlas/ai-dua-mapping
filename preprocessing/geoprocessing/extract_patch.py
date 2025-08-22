#!/usr/bin/env python3
"""
extract_patches.py

Extracts fixed-size patches using patch_id as patch numbers, with:
- Randomized patch numbering
- Automatic train/val/test directory organization
- Support for any number of bands

Usage:
    python extract_patches.py --raster input.tif --grid grid.gpkg --output output_dir
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import os
from tqdm import tqdm
import sys

def validate_grid(grid: gpd.GeoDataFrame) -> bool:
    """Check if grid contains required columns."""
    # required_cols = {'geometry', 'set', 'class'}
    required_cols = {'set', 'patch_id'}
    if not required_cols.issubset(grid.columns):
        print(f"Error: Grid missing required columns {required_cols - set(grid.columns)}")
        return False
    if not all(grid['set'].isin(['train', 'val', 'test'])):
        print("Error: 'set' column must contain only 'train', 'val', or 'test'")
        return False
    return True

def extract_patches(
    raster_path: str,
    grid_path: str,
    output_root: str,
    prefix: str,
    patch_size: int = 128,
    nodata: int = None,
    verbose: bool = True
) -> None:
    """
    Extract patches using patch_id as patch numbers.
    
    Args:
        raster_path: Path to input raster
        grid_path: Path to grid file with FID and set assignments
        output_root: Root output directory
        prefix: Prefix for output files
        patch_size: Patch size in pixels (default: 128)
        nodata: Custom nodata value (default: read from raster)
        verbose: Show progress bars (default: True)
    """
    try:
        # Load and validate grid
        grid = gpd.read_file(grid_path)
        if not validate_grid(grid):
            sys.exit(1)

        # Create output directories
        for set_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_root, set_name), exist_ok=True)

        with rasterio.open(raster_path) as src:
            # Get raster metadata
            num_bands = src.count
            dtype = src.dtypes[0]
            raster_nodata = src.nodatavals[0]
            nodata = nodata if nodata is not None else raster_nodata

            if verbose:
                print(f"Raster Info: {num_bands} band(s), dtype: {dtype}, nodata: {nodata}")

            # Process each dataset split
            set_counts = {'train': 0, 'val': 0, 'test': 0}
            
            for set_name in ['train', 'val', 'test']:
                set_patches = grid[grid['set'] == set_name]
                if len(set_patches) == 0:
                    continue

                if verbose:
                    print(f"\nExtracting {len(set_patches)} {set_name} patches...")
                    pbar = tqdm(set_patches.iterrows(), total=len(set_patches))
                else:
                    pbar = set_patches.iterrows()

                for _, row in pbar:
                    # Use patch_id as patch number
                    patch_id = row['patch_id'] 
                    window = src.window(*row['geometry'].bounds)
                    
                    # Read data with padding if needed
                    data = src.read(
                        range(1, num_bands + 1),
                        window=window,
                        out_shape=(num_bands, patch_size, patch_size),
                        boundless=True,
                        fill_value=nodata
                    )

                    # Save patch using patch_id
                    output_path = os.path.join(output_root, set_name, f"{prefix}_{patch_id}.tif")# change the prfix depending on type of data used(RF_, S2_, S1_, VHR_)
                    with rasterio.open(
                        output_path,
                        'w',
                        driver='GTiff',
                        height=patch_size,
                        width=patch_size,
                        count=num_bands,
                        dtype=dtype,
                        crs=src.crs,
                        transform=rasterio.windows.transform(window, src.transform),
                        nodata=nodata
                    ) as dst:
                        dst.write(data)
                    
                    set_counts[set_name] += 1

        if verbose:
            print(f"\nTotal patches extracted: {sum(set_counts.values())}")

    except Exception as e:
        print(f"\nError during extraction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract patches using FID as patch numbers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--raster', required=True, help='Input raster path')
    parser.add_argument('--grid', required=True, help='Grid file with set assignments')
    parser.add_argument('--output', required=True, help='Root output directory')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for output files')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size in pixels')
    parser.add_argument('--nodata', type=float, help='Override nodata value')
    parser.add_argument('--quiet', action='store_true', help='Disable progress output')
    
    args = parser.parse_args()
    
    extract_patches(
        raster_path=args.raster,
        grid_path=args.grid,
        output_root=args.output,
        prefix=args.prefix,
        patch_size=args.patch_size,
        nodata=args.nodata,
        verbose=not args.quiet
    )