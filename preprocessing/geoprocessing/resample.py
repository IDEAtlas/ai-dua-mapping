import argparse
import numpy as np
import rioxarray as rxr
from rasterio.enums import Resampling

def meters_to_degrees(lat, meters):
    """Convert meters to degrees at given latitude."""
    deg_per_meter_lat = 1.0 / 111320.0
    deg_per_meter_lon = 1.0 / (111320.0 * np.cos(np.radians(lat)))
    return meters * deg_per_meter_lat, meters * deg_per_meter_lon

def resample_raster(input_path, output_path, target_res_meters, method):
    src = rxr.open_rasterio(input_path, masked=True)

    # CRS check
    if not src.rio.crs.is_geographic:
        print("??  Warning: input raster CRS is projected. Skipping meter-to-degree conversion.")
        target_res = (target_res_meters, target_res_meters)  # use meters directly
    else:
        bounds = src.rio.bounds()  # (minx, miny, maxx, maxy)
        center_lat = (bounds[1] + bounds[3]) / 2.0

        res_y_deg, res_x_deg = meters_to_degrees(center_lat, target_res_meters)
        print(f"Resampling to ~{target_res_meters}m")
        target_res = (res_x_deg, res_y_deg)

    resampling_method = getattr(Resampling, method)

    resampled = src.rio.reproject(
        dst_crs=src.rio.crs,
        resolution=target_res,
        resampling=resampling_method
    )

    print("Output resolution:", resampled.rio.resolution())

    resampled = resampled.astype(src.dtype)
    resampled.rio.to_raster(output_path)
    print(f"Saved resampled raster to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Resample raster to coarser resolution in meters (auto-converted to degrees for WGS84).")
    parser.add_argument("--input", required=True, help="Path to input raster (GeoTIFF).")
    parser.add_argument("--output", required=True, help="Path to output raster.")
    parser.add_argument("--resolution", type=float, required=True, help="Target resolution in meters.")
    parser.add_argument("--method", default="nearest", choices=[
        "mode", "nearest", "bilinear", "cubic", "average", "max", "min", "med", "q1", "q3"
    ], help="Resampling method to use (default: nearest)")

    args = parser.parse_args()
    resample_raster(args.input, args.output, args.resolution, args.method)

if __name__ == "__main__":
    main()
