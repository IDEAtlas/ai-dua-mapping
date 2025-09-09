"""
prepare_raster_portal.py

Prepare reference and prediction landcover rasters for a user portal.

- Reference rasters: shift classes (0→1, 1→2, 2→3), optionally clip to a shapefile.
- Prediction rasters: optionally clip, save original, and save a 100 m resampled version.

Usage:
    python prepare_raster_portal.py input.tif output.tif --type reference --clip city.shp
    python prepare_raster_portal.py input.tif output.tif --type prediction
"""
import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import fiona
import rioxarray as rxr
import os

def shift_classes(array):
    """Shift classes: 0→1, 1→2, 2→3"""
    return np.where(array == 0, 1, np.where(array == 1, 2, np.where(array == 2, 3, array)))

def clip_raster(data, transform, shapefile):
    """Clip raster data to shapefile polygon, preserving nodata 0"""
    with fiona.open(shapefile, "r") as shp:
        shapes = [feature["geometry"] for feature in shp]
    mask_arr = geometry_mask(
        shapes,
        transform=transform,
        invert=True,
        out_shape=data.shape,
        all_touched=True,
    )
    clipped = np.zeros_like(data, dtype="uint8")
    clipped[mask_arr] = data[mask_arr]
    return clipped

def save_raster_rasterio(data, transform, crs, profile, output_path):
    """Save raster using rasterio"""
    profile = profile.copy()
    with rasterio.open(output_path, "w", **profile) as dst:
        reproject(
            source=data,
            destination=rasterio.band(dst, 1),
            src_transform=transform,
            src_crs=crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )

def resample_prediction_100m(input_path, output_path):
    """Resample prediction raster to 100 m in EPSG:3857 using rioxarray"""
    src = rxr.open_rasterio(input_path, masked=True)
    src_merc = src.rio.reproject("EPSG:3857")
    resampled = src_merc.rio.reproject(
        dst_crs=src_merc.rio.crs,
        resolution=(100, 100),
        resampling=Resampling.mode
    )
    resampled.rio.to_raster(output_path)
    print(f"Saved 100 m resampled raster to: {output_path}")

def process_raster(input_raster, output_raster, raster_type, clip_shapefile=None):
    with rasterio.open(input_raster) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

        if raster_type == "reference":
            data = shift_classes(data)

        if clip_shapefile:
            data = clip_raster(data, transform, clip_shapefile)

        profile.update(
            crs=crs,
            dtype="uint8",
            compress="lzw",
            count=1,
            nodata=0
        )

        save_raster_rasterio(data, transform, crs, profile, output_raster)
        print(f"Saved original raster to: {output_raster}")

        if raster_type == "prediction":
            base, ext = os.path.splitext(output_raster)
            resampled_path = f"{base}_100m{ext}"
            resample_prediction_100m(output_raster, resampled_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_raster")
    parser.add_argument("output_raster")
    parser.add_argument(
        "--type",
        choices=["reference", "prediction"],
        required=True,
        help="Type of raster: 'reference' (apply shift) or 'prediction' (already shifted)"
    )
    parser.add_argument("--clip", help="Shapefile to clip raster")
    args = parser.parse_args()

    process_raster(args.input_raster, args.output_raster, raster_type=args.type, clip_shapefile=args.clip)
