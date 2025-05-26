import argparse
import rasterio
import numpy as np

def scale_raster(input_path, output_path):
    with rasterio.open(input_path) as src:
        raster = src.read(1)
        profile = src.profile

    # Scale the raster values between 0 and 1
    raster_min = np.min(raster)
    raster_max = np.max(raster)
    scaled_raster = (raster - raster_min) / (raster_max - raster_min)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(scaled_raster, 1)

    print(f"Raster scaled and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Scale raster values between 0 and 1.")
    parser.add_argument("input", type=str, help="Path to the input raster file.")
    parser.add_argument("output", type=str, help="Path to the output raster file.")
    
    args = parser.parse_args()
    scale_raster(args.input, args.output)

if __name__ == "__main__":
    main()