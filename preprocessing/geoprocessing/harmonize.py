import argparse
import rasterio
import numpy as np

def harmonize_s2(input_file, output_file, scale=False):
    """
    Harmonizes Sentinel-2 raster data by adjusting DN values and optionally scaling.
    
    Sentinel-2 scenes after 2022-01-25 (PROCESSING_BASELINE >= '04.00') have their 
    Digital Numbers (DN) shifted by an offset of 1000. This function corrects the 
    DN range to match older scenes for consistent multi-temporal mapping, and 
    optionally scales the values to a range of [0, 1] for machine learning purposes.
    
    Args:
        input_file (str): Path to the input raster file.
        output_file (str): Path to save the harmonized raster file.
        scale (bool): Whether to scale raster values (divide by 10000). Default is False.
    
    Returns:
        None: The harmonized raster is saved to the specified output file.
    """
    offset = 1000  # Fixed offset value

    with rasterio.open(input_file, 'r') as src:
        bands_data = src.read()
        bands_data -= offset
        
        # Apply scaling if requested
        if scale:
            bands_data = bands_data / 10000.0

        meta = src.meta

    meta['dtype'] = 'float32'
    meta['count'] = bands_data.shape[0]  # Update band count dynamically

    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(bands_data)

    print(f"Harmonized raster saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Harmonize Sentinel-2 raster by normalizing DN values.")
    parser.add_argument("input", type=str, help="Path to the input raster file.")
    parser.add_argument("output", type=str, help="Path to the output raster file.")
    parser.add_argument("--scale", action="store_true", help="Apply scaling (divide by 10000). Default is False.")
    args = parser.parse_args()

    harmonize_s2(args.input, args.output, args.scale)

if __name__ == "__main__":
    main()