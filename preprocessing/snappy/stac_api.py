import os
import numpy as np
import dask.array as da
import geopandas as gpd
import stackstac
import planetary_computer as pc
from pystac_client import Client
import rioxarray
from rasterio.enums import Resampling
import warnings
import logging
from datetime import datetime
import gc
import xarray as xr
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# ----------------- Configuration -----------------
CITY = "buenos_aires"
BASEDIR = "/data/raw/"
GEOJSON_FILE = os.path.join(BASEDIR, "aoi", f"{CITY}_aoi.geojson")
YEAR = 2020
START_DATE = f"{YEAR}-08-11"
END_DATE = f"{YEAR}-08-17"
MAX_CLOUD_COVER = 5
CHUNK_SIZE = 512
OUTPUT_DIR = os.path.join(BASEDIR, "sentinel", CITY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters
RESOLUTION = 10
RESAMPLING_METHOD = Resampling.bilinear
COMPRESSION = "LZW"

# Sentinel-2 bands
BAND_ASSETS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
BAND_NAMES = ["Blue", "Green", "Red", "Red_Edge_1", "Red_Edge_2", "Red_Edge_3",
              "NIR", "Narrow_NIR", "SWIR_1", "SWIR_2"]

# Constants for radiometric offset handling
BOA_ADD_OFFSET_PB04 = -1000  # For PB 04.00 and later
QUANTIFICATION_VALUE = 10000  # Standard quantification value for Sentinel-2

def get_processing_baseline_info(item) -> Dict:
    """Extract processing baseline information from STAC item"""
    props = item.properties
    processing_baseline = props.get('s2:processing_baseline', 'N/A')
    
    info = {
        'processing_baseline': processing_baseline,
        'cloud_cover': props.get('eo:cloud_cover', 'N/A'),
        'datetime': props.get('datetime', 'N/A'),
        'mgrs_tile': props.get('s2:mgrs_tile', 'N/A'),
        'requires_offset_correction': False
    }
    
    # Check if this is PB 04.00 or later that requires offset correction
    try:
        if processing_baseline != 'N/A':
            # Parse baseline version (e.g., "04.00" -> 4.0)
            baseline_version = float(processing_baseline.split('.')[0])
            info['requires_offset_correction'] = baseline_version >= 4.0
    except (ValueError, AttributeError):
        pass
    
    return info

def apply_radiometric_offset_correction(data_array, requires_correction: bool):
    """
    Apply radiometric offset correction for PB 04.00+ data
    Formula: Actual reflectance = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
    """
    if not requires_correction:
        return data_array
    
    logger.info("Applying radiometric offset correction for PB 04.00+ data")
    
    # Apply the correction: (DN + offset) / quantification
    corrected_data = (data_array.astype('float32') + BOA_ADD_OFFSET_PB04) / QUANTIFICATION_VALUE
    
    # Handle no-data values (DN = 0 should remain as no-data)
    corrected_data = da.where(data_array == 0, np.nan, corrected_data)
    
    # Clip negative values to 0 (or keep as is for analysis)
    corrected_data = da.where(corrected_data < 0, 0, corrected_data)
    
    return corrected_data

# ----------------- Load AOI -----------------
try:
    aoi = gpd.read_file(GEOJSON_FILE).to_crs(epsg=4326)
    bbox = aoi.total_bounds.tolist()
    logger.info(f"Querying Sentinel 2 L2A for {CITY} for year {YEAR}")
except Exception as e:
    logger.error(f"Failed to load AOI from {GEOJSON_FILE}: {e}")
    raise

# ----------------- STAC Search -----------------
try:
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        datetime=f"{START_DATE}/{END_DATE}",
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    )
    items = search.item_collection()
    if len(items) == 0:
        raise ValueError("No Sentinel-2 scenes found for the specified date range and AOI.")
    logger.info(f"Found {len(items)} scenes")
        
except Exception as e:
    logger.error(f"STAC search failed: {e}")
    raise

# ----------------- Group by MGRS tile -----------------
from shapely.geometry import shape

# Geometry of AOI (merged if multi-part)
aoi_geom = aoi.geometry.unary_union

items_by_tile = {}
for it in items:
    try:
        footprint = shape(it.geometry)
        if not footprint.intersects(aoi_geom):
            continue  # skip tiles not overlapping AOI

        tile_id = it.properties.get("s2:mgrs_tile") or it.id
        cloud = float(it.properties.get("eo:cloud_cover", 100.0))

        # Keep lowest-cloud item for each tile
        if tile_id not in items_by_tile or cloud < items_by_tile[tile_id]["cloud"]:
            items_by_tile[tile_id] = {"item": it, "cloud": cloud}

    except Exception as e:
        logger.warning(f"Error filtering item {it.id}: {e}")
        continue

# Final filtered items
filtered_items = [v["item"] for v in items_by_tile.values()]
signed_items = [pc.sign(it) for it in filtered_items]

logger.info(f"Selected {len(filtered_items)} unique tiles after filtering")

# Print processing baseline summary for selected items
logger.info("=== SELECTED ITEMS SUMMARY ===")
baseline_counts = {}
correction_needed_count = 0

for it in signed_items:
    info = get_processing_baseline_info(it)
    processing_baseline = info['processing_baseline']
    baseline_counts[processing_baseline] = baseline_counts.get(processing_baseline, 0) + 1
    
    if info['requires_offset_correction']:
        correction_needed_count += 1
    
    logger.info(f"  - {it.id} (cloud: {info['cloud_cover']}%, baseline: {processing_baseline}, correction: {info['requires_offset_correction']})")

logger.info(f"Processing Baseline Distribution: {baseline_counts}")
logger.info(f"Items requiring radiometric offset correction: {correction_needed_count}/{len(signed_items)}")

# Check if all items have the same processing baseline
if len(baseline_counts) > 1:
    logger.warning("WARNING: Items have different processing baselines! This may affect ML model consistency.")
    if correction_needed_count > 0 and correction_needed_count < len(signed_items):
        logger.warning("MIXED BASELINES: Some items require offset correction, others don't!")
        logger.warning("Consider filtering to a single processing baseline for consistent time series analysis.")

# ----------------- Process in native CRS -----------------
unique_items = {}
for item in signed_items:
    try:
        tile_id = item.properties.get("s2:mgrs_tile")
        cloud = item.properties.get("eo:cloud_cover", 100)
        if tile_id not in unique_items or cloud < unique_items[tile_id].properties.get("eo:cloud_cover", 100):
            unique_items[tile_id] = item
    except Exception as e:
        logger.warning(f"Error processing signed item: {e}")
        continue

selected_items = list(unique_items.values())

try:
    first_asset_href = signed_items[0].assets["B02"].href
    with rioxarray.open_rasterio(first_asset_href) as tmp:
        mosaic_crs = tmp.rio.crs
    logger.info(f"Native CRS: {mosaic_crs}")
except Exception as e:
    logger.error(f"Failed to determine CRS: {e}")
    raise

logger.info("Processing stack...")

try:
    # stackstac.stack returns a Dask Array, not xarray DataArray
    stack = stackstac.stack(
        signed_items,
        assets=BAND_ASSETS,
        resolution=RESOLUTION,
        bounds_latlon=None,
        chunksize=CHUNK_SIZE
    ).astype("float32")

    # ----------------- Apply Radiometric Offset Correction -----------------
    # Check if any items require correction
    requires_correction = any(get_processing_baseline_info(item)['requires_offset_correction'] 
                             for item in signed_items)
    
    if requires_correction:
        logger.info("Applying radiometric offset correction to the stack...")
        # Apply correction to the entire stack
        stack = apply_radiometric_offset_correction(stack, requires_correction)
    else:
        # Convert DN to reflectance for pre-PB04 data (divide by quantification value)
        logger.info("Converting DN to reflectance for pre-PB04 data...")
        stack = stack / QUANTIFICATION_VALUE
        # Handle no-data values
        stack = da.where(stack == 0, np.nan, stack)

    # ----------------- Median mosaic -----------------
    logger.info("Computing median mosaic...")
    mosaic = da.nanmedian(stack, axis=0) 

    # Get spatial coordinates from one of the original items
    with rioxarray.open_rasterio(signed_items[0].assets["B02"].href) as example_raster:
        transform = example_raster.rio.transform()
        height, width = mosaic.shape[1], mosaic.shape[2]  # bands, y, x
    
    y_coords = np.arange(height) * transform[4] + transform[5] + transform[4] / 2
    x_coords = np.arange(width) * transform[0] + transform[2] + transform[0] / 2
    
    # Convert to xarray DataArray
    mosaic_xr = xr.DataArray(
        mosaic,
        dims=["band", "y", "x"],
        coords={
            "band": BAND_NAMES,
            "y": y_coords,
            "x": x_coords
        }
    )
    
    # Set CRS
    mosaic_xr = mosaic_xr.rio.write_crs(mosaic_crs)
    mosaic_xr = mosaic_xr.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # Clip to AOI bounding box in UTM
    aoi_proj = aoi.to_crs(mosaic_crs)
    mosaic_xr = mosaic_xr.rio.clip_box(*aoi_proj.total_bounds)

    # Clear intermediate variables to free memory
    del stack, mosaic
    gc.collect()

    # ----------------- Save raster -----------------
    output_file = os.path.join(OUTPUT_DIR, f"S2_{YEAR}.tif")
    
    logger.info("Reprojecting to EPSG:4326...")
    mosaic_wgs = mosaic_xr.rio.reproject(
        "EPSG:4326",
        resampling=RESAMPLING_METHOD,
        nodata=np.nan
    )

    # Add metadata to the output file
    mosaic_wgs.attrs = {
        "bands": ", ".join(BAND_NAMES),
        "Harmonized": str(requires_correction),
        "data_range": "0-1 reflectance"
    }

    logger.info(f"Saving to {output_file}...")
    mosaic_wgs.rio.to_raster(output_file, compress=COMPRESSION, nodata=0)

    # Clean up
    del mosaic_xr, mosaic_wgs
    gc.collect()

    logger.info("Processing completed successfully!")
    logger.info(f"Output data range: 0-1 reflectance (radiometrically corrected)")

except Exception as e:
    logger.error(f"Processing failed: {e}")
    import traceback
    logger.error(traceback.format_exc())