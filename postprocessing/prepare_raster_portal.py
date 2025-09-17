
import argparse
import rasterio
import rasterio.mask
import rasterio.warp
import rasterio.transform
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


def clip(input_path, aoi, output_path, do_shift=False, resample_100m=False):
    # Read the vector mask
    with fiona.open(aoi, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        aoi_crs = shapefile.crs
        print(f"AOI CRS: {aoi_crs}")

    # Open raster and shift values before clipping



    with rasterio.open(input_path) as src:
        print(f"Raster CRS: {src.crs}")
        print(f"Raster shape: {src.height} x {src.width}")
        # Clip the raster using the vector mask
        clipped, trans = rasterio.mask.mask(src, shapes, crop=True)
        print(f"Clipped shape: {clipped.shape}")
        # clipped shape: (bands, height, width)
        band = clipped[0]
        print(f"Unique values before shifting: {np.unique(band)}")
        if do_shift:
            # Only shift valid data (not nodata)
            nodata = src.nodata
            if nodata is not None:
                mask = band != nodata
            else:
                mask = np.ones_like(band, dtype=bool)
            shifted = band.copy()
            shifted[mask] = shift_classes(band[mask])
            print(f"Unique values after shifting: {np.unique(shifted)}")
            shifted = np.asarray(shifted, dtype=np.uint8)
        else:
            shifted = np.asarray(band, dtype=np.uint8)

    # Reproject to EPSG:3857 (Web Mercator)
    dst_crs = 'EPSG:3857'
    # Calculate the transform and shape for the new projection
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src.crs, dst_crs, shifted.shape[1], shifted.shape[0], *rasterio.transform.array_bounds(shifted.shape[0], shifted.shape[1], trans)
    )
    if dst_width is None or dst_height is None or dst_width == 0 or dst_height == 0:
        raise ValueError("Reprojection failed: AOI does not overlap raster or resulted in empty output.")
    dst_width = int(dst_width)
    dst_height = int(dst_height)
    out_transform = dst_transform
    out_width = dst_width
    out_height = dst_height
    # If resample_100m, adjust transform and shape for 100m resolution
    if resample_100m:
        # 100m per pixel in EPSG:3857
        pixel_size = 100.0
        minx, miny, maxx, maxy = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
        out_width = int(np.ceil((maxx - minx) / pixel_size))
        out_height = int(np.ceil((maxy - miny) / pixel_size))
        out_transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)
    # Prepare output array
    reprojected = np.empty((out_height, out_width), dtype=np.uint8)
    rasterio.warp.reproject(
        source=shifted,
        destination=reprojected,
        src_transform=trans,
        src_crs=src.crs,
        dst_transform=out_transform,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.mode
    )
    # Update the metadata for the output
    meta = src.meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": out_height,
        "width": out_width,
        "transform": out_transform,
        "dtype": "uint8",
        "count": 1,
        "compress": "DEFLATE",
        "crs": dst_crs,
        "nodata": 0,
    })

    # Save the reprojected, clipped, and shifted raster to the output file
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(reprojected, 1)

    print(f"Clipped and shifted raster saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .tif file to clip")
    parser.add_argument("aoi", help="Path to .shp file to use as mask")
    parser.add_argument("output_path", help="Path to output .tif file")
    parser.add_argument("--shift", action="store_true", help="Shift class values (0→1, 1→2, 2→3)")
    parser.add_argument("--resample", action="store_true", help="Resample output to 100m resolution in EPSG:3857")
    args = parser.parse_args()

    clip(args.input_path, args.aoi, args.output_path, do_shift=args.shift, resample_100m=args.resample)


if __name__ == "__main__":
    main()
