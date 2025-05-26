import argparse
import rasterio
import rasterio.mask
import fiona
import numpy as np


def shift_classes(array):
    """
    Shift class values in the raster.
    - 0 → 1
    - 1 → 2
    - 2 → 3
    """
    print("Shifting class values: 0→1, 1→2, 2→3")
    return np.where(array == 0, 1, np.where(array == 1, 2, np.where(array == 2, 3, array)))


def clip(input_path, mask_path, output_path):
    # Read the vector mask
    with fiona.open(mask_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # Open raster and shift values before clipping
    with rasterio.open(input_path) as src:
        raster = src.read(1)  # Read the first (only) band
        raster = raster + 1
        shifted_raster = shift_classes(raster)

        # Clip the shifted raster using the vector mask
        clipped, trans = rasterio.mask.mask(src, shapes, crop=True)

        # Update the metadata for the output
        meta = src.meta
        meta.update(
            {
                "driver": "GTiff",
                "height": clipped.shape[0],
                "width": clipped.shape[1],
                "transform": trans,
                "dtype": "uint8",
                "count": 1,
            }
        )

    # Save the clipped and shifted raster to the output file
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(clipped, 1)

    print(f"Clipped and shifted raster saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .tif file to clip")
    parser.add_argument("mask_path", help="Path to .shp file to use as mask")
    parser.add_argument("output_path", help="Path to output .tif file")
    args = parser.parse_args()

    clip(args.input_path, args.mask_path, args.output_path)


if __name__ == "__main__":
    main()
