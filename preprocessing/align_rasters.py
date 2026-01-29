import argparse
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import numpy as np

def align_raster(input_raster, master_raster, output_raster):
    """
    Aligns a single raster to match the resolution and extent of a master raster,
    while preserving the metadata (data type, bands) of the input raster.

    Parameters:
    - input_raster (str): Path to the input raster.
    - master_raster (str): Path to the master raster.
    - output_raster (str): Path to save the aligned raster.
    """
    # Open the master raster (reference)
    with rasterio.open(master_raster) as ref_raster:
        ref_transform = ref_raster.transform
        ref_crs = ref_raster.crs
        ref_shape = (ref_raster.height, ref_raster.width)

        # Open the input raster
        with rasterio.open(input_raster) as src:
            # Initialize destination data array based on input raster's bands and data type
            destination = np.zeros((src.count, ref_shape[0], ref_shape[1]), dtype=src.dtypes[0])

            # Reproject and resample the input raster
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),  # Align all bands
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )

            # Save the aligned raster with input raster's metadata
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": ref_shape[0],
                "width": ref_shape[1],
                "transform": ref_transform,
                "crs": ref_crs,
            })

            with rasterio.open(output_raster, "w", **profile) as dst:
                dst.write(destination)

    print(f"Aligned raster saved to {output_raster}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align a raster to match a master raster's extent and resolution.")
    parser.add_argument("--input", required=True, help="Path to the input raster.")
    parser.add_argument("--master", required=True, help="Path to the master raster.")
    parser.add_argument("--output", required=True, help="Path to save the aligned raster.")

    args = parser.parse_args()

    align_raster(args.input, args.master, args.output)