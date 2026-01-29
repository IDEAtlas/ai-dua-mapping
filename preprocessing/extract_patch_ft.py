#!/usr/bin/env python3
"""
extract_patch_ft.py

Extracts fixed-size patches for fine-tuning (train set only).
Uses patch_id as patch numbers. Focuses on slum-rich areas only.

Usage:
    python extract_patch_ft.py --raster input.tif --grid grid.geojson --output output_dir --prefix S2
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
    required_cols = {'patch_id', 'geometry'}
    if not required_cols.issubset(grid.columns):
        print(f"Error: Grid missing required columns {required_cols - set(grid.columns)}")
        return False
    # 'set' column is optional - if missing, we'll treat all as 'train'
    if 'set' in grid.columns and 'train' not in grid['set'].values and len(grid[grid['set'] != 'train']) > 0:
        # Only warn if 'set' exists and no 'train' patches found
        print("Warning: Fine-tuning grid has no 'train' patches, will use all patches")
    return True


def extract_patches_ft(
    raster_path: str,
    grid_path: str,
    output_root: str,
    prefix: str,
    patch_size: int = 128,
    nodata: int = None,
    verbose: bool = True
) -> None:
    """
    Extract patches for fine-tuning (train set only).
    
    Args:
        raster_path: Path to input raster
        grid_path: Path to grid file with patch_id and set assignments
        output_root: Root output directory
        prefix: Prefix for output files (e.g., 'S2', 'BD')
        patch_size: Patch size in pixels (default: 128)
        nodata: Custom nodata value (default: read from raster)
        verbose: Show progress bars (default: True)
    """
    try:
        # Load and validate grid
        grid = gpd.read_file(grid_path)
        if not validate_grid(grid):
            sys.exit(1)

        # Create output directory for train only
        train_dir = os.path.join(output_root, 'train')
        os.makedirs(train_dir, exist_ok=True)

        with rasterio.open(raster_path) as src:
            # Get raster metadata
            num_bands = src.count
            dtype = src.dtypes[0]
            raster_nodata = src.nodatavals[0]
            nodata = nodata if nodata is not None else raster_nodata

            if verbose:
                print(f"Raster Info: {num_bands} band(s), dtype: {dtype}, nodata: {nodata}")
                print(f"Output directory: {train_dir}")

            # Process train patches only, or all patches if 'set' column doesn't exist
            if 'set' in grid.columns:
                train_patches = grid[grid['set'] == 'train']
            else:
                # If no 'set' column, treat all patches as train
                train_patches = grid
            
            if len(train_patches) == 0:
                print("Warning: No patches found in grid!")
                return

            if verbose:
                print(f"\nExtracting {len(train_patches)} patches (slum-focused)...")
                pbar = tqdm(train_patches.iterrows(), total=len(train_patches))
            else:
                pbar = train_patches.iterrows()

            extracted_count = 0
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
                output_path = os.path.join(train_dir, f"{prefix}_{patch_id}.tif")
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
                
                extracted_count += 1

            if verbose:
                print(f"\nTotal patches extracted: {extracted_count}")
                print(f"Saved to: {train_dir}")

    except Exception as e:
        print(f"\nError during extraction: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract patches for fine-tuning (train set only, slum-focused)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--raster', required=True, help='Input raster path')
    parser.add_argument('--grid', required=True, help='Grid file with patch_id and set assignments')
    parser.add_argument('--output', required=True, help='Root output directory')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for output files (e.g., S2, BD)')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size in pixels')
    parser.add_argument('--nodata', type=float, help='Override nodata value')
    parser.add_argument('--quiet', action='store_true', help='Disable progress output')
    
    args = parser.parse_args()
    
    extract_patches_ft(
        raster_path=args.raster,
        grid_path=args.grid,
        output_root=args.output,
        prefix=args.prefix,
        patch_size=args.patch_size,
        nodata=args.nodata,
        verbose=not args.quiet
    )
