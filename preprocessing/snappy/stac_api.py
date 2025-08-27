import os
import numpy as np
import dask.array as da
import geopandas as gpd
import stackstac
import planetary_computer as pc
from pystac_client import Client
import rioxarray
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from pyproj import Transformer
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
CITY = "lagos"
BASEDIR = "/data/raw/"
GEOJSON_FILE = os.path.join(BASEDIR, "aoi", f"{CITY}_aoi.geojson")
YEAR = 2024
START_DATE = f"{YEAR}-01-11"
END_DATE = f"{YEAR}-08-28"
MAX_CLOUD_COVER = 1
CHUNK_SIZE = 512
OUTPUT_DIR = os.path.join(BASEDIR, "sentinel", CITY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters
RESOLUTION = 10  # Will be converted to degrees based on native 10m band
RESAMPLING_METHOD = Resampling.bilinear
COMPRESSION = "LZW"
TO_SR = False  # Convert to surface reflectance

# Selected Sentinel-2 bands
BAND_ASSETS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

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
            baseline_version = float(processing_baseline.split('.')[0])
            info['requires_offset_correction'] = baseline_version >= 4.0
    except (ValueError, AttributeError):
        pass
    
    return info

# ----------------- Load AOI -----------------
try:
    aoi = gpd.read_file(GEOJSON_FILE).to_crs("EPSG:4326")
    bbox = aoi.total_bounds.tolist()
    logger.info(f"Searching Sentinel-2 L2A imagery for {CITY.replace('_', ' ').title()} ({YEAR})")
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
from shapely.ops import unary_union

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

# ----------------- Validate AOI Coverage -----------------
logger.info("Validating AOI coverage...")

# Create union of all selected item footprints
selected_footprints = []
for item in signed_items:
    try:
        footprint = shape(item.geometry)
        selected_footprints.append(footprint)
    except Exception as e:
        logger.warning(f"Error processing footprint for item {item.id}: {e}")
        continue

if not selected_footprints:
    logger.error("No valid footprints found from selected items!")
    raise ValueError("No valid footprints found from selected items!")

# Union of all selected footprints
combined_footprint = unary_union(selected_footprints)

# Check coverage
aoi_area = aoi_geom.area
covered_area = aoi_geom.intersection(combined_footprint).area
coverage_percentage = (covered_area / aoi_area) * 100

logger.info(f"AOI coverage: {coverage_percentage:.2f}%")

# Set minimum coverage threshold (e.g., 95%)
MIN_COVERAGE_THRESHOLD = 95.0

if coverage_percentage < MIN_COVERAGE_THRESHOLD:
    logger.error(f"Search results only cover {coverage_percentage:.1f}% of the AOI - incomplete coverage detected.")
    logger.error("Please adjust date range or increase cloud cover threshold and try again.")
    raise ValueError(f"Insufficient AOI coverage: {coverage_percentage:.1f}%")
else:
    logger.info("✓ AOI coverage validation passed - proceeding with processing")

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

logger.info("Processing stack...")

try:
    # Check for multiple CRS scenario
    crs_set = set()
    
    for item in signed_items:
        try:
            # Get CRS from the first band asset
            with rasterio.open(item.assets[BAND_ASSETS[0]].href) as src:
                epsg_code = src.crs.to_epsg()
                crs_set.add(epsg_code)
        except Exception as e:
            logger.warning(f"Could not determine CRS for item {item.id}: {e}")
    
    logger.info(f"Found CRS codes: {sorted(crs_set)}")
    
    # Choose target CRS
    if len(crs_set) > 1:
        # Multiple CRS detected - use the last one from the sorted list
        target_epsg = sorted(crs_set)[-1]
        target_crs = f"EPSG:{target_epsg}"
        logger.warning(f"Multiple CRS detected: {sorted(crs_set)}")
        logger.info(f"Using CRS: {target_crs}")
        
        # Use specified EPSG
        stack = stackstac.stack(
            signed_items,
            assets=BAND_ASSETS,
            resolution=RESOLUTION,  # Use 10m directly
            chunksize=CHUNK_SIZE,
            epsg=target_epsg
        ).astype("float32")
    else:
        # Single CRS - let stackstac handle automatically
        target_epsg = list(crs_set)[0] if crs_set else None
        target_crs = f"EPSG:{target_epsg}" if target_epsg else None
        logger.info(f"Using native CRS: {target_crs}")
        
        stack = stackstac.stack(
            signed_items,
            assets=BAND_ASSETS,
            resolution=RESOLUTION,  # Use 10m directly in native CRS
            chunksize=CHUNK_SIZE
        ).astype("float32")

    logger.info(f"Stack shape: {stack.shape}")
    logger.info(f"Stack dimensions: {stack.dims}")

    # ----------------- Apply Radiometric Offset Correction -----------------
    # Check if any items require correction (for metadata purposes)
    requires_correction = any(get_processing_baseline_info(item)['requires_offset_correction'] 
                             for item in signed_items)
    
    # Step 1: Apply BOA offset correction if required
    if requires_correction:
        logger.info("Applying BOA offset correction to the stack...")
        # Apply the BOA offset correction: DN_corrected = DN + BOA_ADD_OFFSET
        stack = stack + BOA_ADD_OFFSET_PB04
        # Clip negative values to 0 (same as original function behavior)
        stack = stack.where(stack >= 0, 0)
    
    # Step 2: Apply quantification (convert to surface reflectance) if TO_SR is True
    if TO_SR:
        logger.info("Converting Raw Digital Numbers to Surface Reflectance...")
        stack = stack / QUANTIFICATION_VALUE
        data_type_info = "SR (BOA offset corrected)" if requires_correction else "SR"
    else:
        logger.info("Keeping Data as Raw Digital Numbers")
        data_type_info = "RAW DN (BOA offset corrected)" if requires_correction else "RAW DN"
    
    stack = stack.where(stack != 0, np.nan)

    # ----------------- Median mosaic -----------------
    logger.info("Computing median mosaic...")
    
    # Use the target CRS we determined during stack creation
    mosaic_crs = target_crs
    if mosaic_crs is None:
        # Fallback to reading from stack if target_crs wasn't set
        mosaic_crs = stack.rio.crs
        if mosaic_crs is None:
            with rasterio.open(signed_items[0].assets["B02"].href) as src:
                mosaic_crs = src.crs
    
    logger.info(f"Mosaic CRS: {mosaic_crs}")
    
    mosaic_xr = stack.median(dim='time', skipna=True)
    
    if 'band' in mosaic_xr.dims and len(mosaic_xr.coords['band']) == len(BAND_ASSETS):
        mosaic_xr = mosaic_xr.assign_coords(band=BAND_ASSETS)

    # Ensure mosaic_crs is a proper CRS object for rioxarray
    if isinstance(mosaic_crs, str):
        from rasterio.crs import CRS
        mosaic_crs_obj = CRS.from_string(mosaic_crs)
    else:
        mosaic_crs_obj = mosaic_crs
    
    mosaic_xr = mosaic_xr.rio.write_crs(mosaic_crs_obj)
    mosaic_xr = mosaic_xr.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # Clip to AOI bounding box (reproject AOI to match mosaic CRS)
    aoi_proj = aoi.to_crs(mosaic_crs)
    clip_bounds = aoi_proj.total_bounds
    logger.info(f"Clipping bounds in {mosaic_crs}: {clip_bounds}")
    logger.info(f"Mosaic extent - X: {mosaic_xr.x.min().values} to {mosaic_xr.x.max().values}, Y: {mosaic_xr.y.min().values} to {mosaic_xr.y.max().values}")
    
    # Check if bounds are valid before clipping
    x_range = clip_bounds[2] - clip_bounds[0]  # max_x - min_x
    y_range = clip_bounds[3] - clip_bounds[1]  # max_y - min_y
    
    if x_range <= 0 or y_range <= 0:
        logger.warning("Invalid clipping bounds detected, skipping clipping step")
    else:
        try:
            mosaic_xr = mosaic_xr.rio.clip_box(*clip_bounds)
            logger.info("Successfully clipped to AOI bounds")
        except Exception as e:
            logger.warning(f"Clipping failed: {e}. Proceeding without clipping.")

    # Clear intermediate variables to free memory
    del stack
    gc.collect()

    # ----------------- Save raster -----------------
    output_file = os.path.join(OUTPUT_DIR, f"S2_{YEAR}.tif")
    
    # Reproject to EPSG:4326 for final output
    logger.info("Reprojecting to EPSG:4326...")
    mosaic_wgs = mosaic_xr.rio.reproject(
        "EPSG:4326",
        resampling=RESAMPLING_METHOD,
        nodata=np.nan
    )

    # update metadata
    mosaic_wgs.attrs = {
        "S2_BANDS": ", ".join(BAND_ASSETS),
        "BOA_OFFSET_APPLIED": str(requires_correction),
        "SURFACE_REFLECTANCE": str(TO_SR),
        "DATA_RANGE": data_type_info
    }

    logger.info(f"Saving to {output_file}...")
    mosaic_wgs.rio.to_raster(output_file, compress=COMPRESSION, nodata=0, tiled=True, BIGTIFF="YES")

    # Clean up
    del mosaic_xr, mosaic_wgs
    gc.collect()

    logger.info("Processing completed successfully!")
    logger.info(f"Output data range: {data_type_info}")

except Exception as e:
    logger.error(f"Processing failed: {e}")
    import traceback
    logger.error(traceback.format_exc())