# #import snapista
# #from snapista import Graph


# #a = Graph.list_operators()
# #print(a)
# #b= Graph.describe_operators() 
# #print(b)
# import sys
# sys.path.append('/home/eouser/anaconda3/envs/geoprocess/lib')  # Replace with the actual path to snappy

import os
from esa_snappy import ProductIO, HashMap, GPF
import geopandas as gpd
from pyproj import CRS
import time

# Preprocessing Functions
def apply_orbit_file(product):
    """Apply orbit file to a Sentinel-1 product."""
    params = HashMap()
    params.put('orbitType', 'Sentinel Precise (Auto Download)')
    params.put('continueOnFail', 'true')
    return GPF.createProduct('Apply-Orbit-File', params, product)

def terrain_correction(product):
    """Perform terrain correction."""
    params = HashMap()
    params.put('demName', 'SRTM 3Sec')
    params.put('pixelSpacingInMeter', 10.0)
    return GPF.createProduct('Terrain-Correction', params, product)

def calibrate(product):
    """Calibrate Sentinel-1 product."""
    params = HashMap()
    params.put('outputSigmaBand', True)
    return GPF.createProduct('Calibration', params, product)

def remove_speckle_noise(product):
    """Apply speckle filtering."""
    params = HashMap()
    params.put('filter', 'Refined Lee')
    return GPF.createProduct('Speckle-Filter', params, product)

def subset_aoi(product, shapefile_path):
    """Subset the Sentinel-1 product to the AOI, handling CRS mismatches."""
    # Read AOI shapefile
    print(f"AOI Shapefile Path: {shapefile_path}")
    region_shp = gpd.read_file(shapefile_path)
    
    if region_shp.crs != "EPSG:4326":
        region_shp = region_shp.to_crs("EPSG:4326")

    # Extract Sentinel-1 CRS from the product
    crs_s1 = CRS.from_string(product.getSceneGeoCoding().getMapCRS().toWKT()).to_epsg()
    if crs_s1 is None:
        crs_s1 = product.getSceneGeoCoding().getMapCRS().toWKT()  # Fallback to WKT

    # Reproject AOI to match Sentinel-1 CRS, if necessary
    if region_shp.crs.to_string() != crs_s1:
        print(f"Reprojecting AOI from {region_shp.crs} to {crs_s1}...")
        region_shp = region_shp.to_crs(crs_s1)

    # Convert AOI geometry to WKT
    region_shp = region_shp.explode(index_parts=True)  # Handles multi-part geometries
    wkt = region_shp.iloc[0].geometry.wkt  # Use WKT format for Subset operator

    # Apply the Subset operator
    params = HashMap()
    params.put('geoRegion', wkt)
    subsetted_product = GPF.createProduct('Subset', params, product)

    return subsetted_product

def save_as_geotiff(product, output_path):
    """Save processed product as GeoTIFF."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    # Ensure the output path has a .tif extension
    if not output_path.endswith('.tif'):
        output_path += '.tif'
    ProductIO.writeProduct(product, output_path, 'GeoTIFF-BigTIFF')
    print(f"GeoTIFF saved at: {output_path}")

# Main Workflow
def preprocess_s1(input_path, output_path, aoi):
    """
    Process Sentinel-1 product with optional AOI subsetting and orbit file application.

    Args:
    - input_path: Path to the Sentinel-1 product.
    - output_path: Directory to save the final output.
    - aoi: Path to shapefile for AOI clipping (optional).
    - apply_orbit: Boolean to apply orbit correction (default False).
    """
    print("Reading Sentinel-1 product...")
    product = ProductIO.readProduct(input_path)


    print("Applying orbit file...")
    product = apply_orbit_file(product)

    print("Calibrating product...")
    product = calibrate(product)
    
    print("Removing speckle noise...")
    product = remove_speckle_noise(product)

    print("Performing terrain correction...")
    product = terrain_correction(product)

    print("Subsetting to AOI...")
    product = subset_aoi(product, aoi)

    print(f"Saving output as GeoTIFF...")
    save_as_geotiff(product, output_path)
    print("Processing complete.")