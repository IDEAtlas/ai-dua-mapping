#!/usr/bin/env python3
import argparse
import geopandas as gpd
import rioxarray
from rasterio.enums import Resampling
import numpy as np
import os
from pathlib import Path


def process_raster(input_path, output_dir, aoi=None):
    src = rioxarray.open_rasterio(input_path)
    print(f"\nProcessing: {input_path}")
    print(f"Source CRS: {src.rio.crs}")

    unique_vals = np.unique(src.data)
    print(f"Source unique values: {unique_vals}")

    needs_shift = set([0, 1, 2]).issubset(unique_vals) and 3 not in unique_vals
    print(f"Needs class shifting: {needs_shift}")

    if needs_shift:
        shifted = src.copy(data=((src.data + 1).astype("uint8")))
        print(f"After shift unique values: {np.unique(shifted.data)}")
    else:
        shifted = src.copy()
        print("No shift applied - using original values")

    shifted.rio.write_nodata(0, inplace=True)

    shifted = shifted.rio.reproject("EPSG:3857", resampling=Resampling.mode, nodata=0, resolution=10)
    print(f"After reprojection unique values: {np.unique(shifted.data)}")

    if aoi:
        gdf = gpd.read_file(aoi)
        gdf = gdf.to_crs("EPSG:3857")
        shifted = shifted.rio.clip(gdf.geometry.values, gdf.crs, all_touched=True, drop=False)
        shifted.rio.write_nodata(0, inplace=True)
        print(f"After clipping unique values: {np.unique(shifted.data)}")

    final_values = np.unique(shifted.data)
    print(f"Final unique values: {final_values}")

    # Extract city and year from filename
    base_name = Path(input_path).stem
    parts = base_name.split(".")
    if len(parts) >= 3:
        city = parts[0]
        year = parts[-1]
    else:
        city = base_name
        year = "unknown"

    output_10m_path = Path(output_dir) / f"{city}_{year}_10m.tif"
    output_100m_path = Path(output_dir) / f"{city}_{year}_100m.tif"

    shifted.rio.to_raster(
        output_10m_path,
        compress="lzw",
        tiled=True,
        nodata=0,
        dtype="uint8"
    )
    print(f"10m output saved to: {output_10m_path}")

    print("Creating 100m resampled version...")
    resampled_100m = shifted.rio.reproject(
        "EPSG:3857",
        resolution=100,
        resampling=Resampling.mode,
        nodata=0
    )
    resampled_100m.rio.to_raster(
        output_100m_path,
        compress="lzw",
        tiled=True,
        nodata=0,
        bigtiff="YES",
        dtype="uint8"
    )
    print(f"100m resampled version saved to: {output_100m_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch process rasters: shift classes, reproject to Web Mercator, and save 10m and 100m outputs.")
    parser.add_argument("input_dir", help="Path to folder containing .tif files")
    parser.add_argument("output_dir", help="Folder where outputs will be saved")
    parser.add_argument("--aoi", help="AOI shapefile/GeoJSON for clipping (optional)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tif_files = sorted(Path(args.input_dir).glob("*.tif"))

    if not tif_files:
        print("No .tif files found in the input directory.")
        return

    print(f"Found {len(tif_files)} files. Starting processing...")

    for tif_path in tif_files:
        try:
            process_raster(tif_path, args.output_dir, aoi=args.aoi)
        except Exception as e:
            print(f"Error processing {tif_path}: {e}")


if __name__ == "__main__":
    main()
