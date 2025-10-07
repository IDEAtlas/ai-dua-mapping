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
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# ----------------- Constants -----------------
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
    try:
        if processing_baseline != 'N/A':
            baseline_version = float(processing_baseline.split('.')[0])
            info['requires_offset_correction'] = baseline_version >= 4.0
    except (ValueError, AttributeError):
        pass
    return info

def load_aoi(geojson_file: str) -> Tuple[gpd.GeoDataFrame, List[float]]:
    """Load AOI from geojson and get bounding box"""
    aoi = gpd.read_file(geojson_file).to_crs("EPSG:4326")
    bbox = aoi.total_bounds.tolist()
    return aoi, bbox

def stac_search(bbox: List[float], start_date: str, end_date: str, max_cloud_cover: int) -> List:
    """Search STAC for Sentinel-2 L2A items"""
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        datetime=f"{start_date}/{end_date}",
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )
    items = search.item_collection()
    if len(items) == 0:
        raise ValueError("No Sentinel-2 scenes found for the specified date range and AOI.")
    return items

def filter_items_by_tile(items, aoi_geom) -> List:
    """Filter items by MGRS tile, keeping lowest cloud cover per tile"""
    from shapely.geometry import shape
    items_by_tile = {}
    for it in items:
        try:
            footprint = shape(it.geometry)
            if not footprint.intersects(aoi_geom):
                continue
            tile_id = it.properties.get("s2:mgrs_tile") or it.id
            cloud = float(it.properties.get("eo:cloud_cover", 100.0))
            if tile_id not in items_by_tile or cloud < items_by_tile[tile_id]["cloud"]:
                items_by_tile[tile_id] = {"item": it, "cloud": cloud}
        except Exception as e:
            logger.warning(f"Error filtering item {it.id}: {e}")
            continue
    filtered_items = [v["item"] for v in items_by_tile.values()]
    return filtered_items

def validate_aoi_coverage(signed_items, aoi_geom, min_coverage_threshold: float = 95.0) -> None:
    """Validate AOI coverage by selected items"""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    selected_footprints = []
    for item in signed_items:
        try:
            footprint = shape(item.geometry)
            selected_footprints.append(footprint)
        except Exception as e:
            logger.warning(f"Error processing footprint for item {item.id}: {e}")
            continue
    if not selected_footprints:
        raise ValueError("No valid footprints found from selected items!")
    combined_footprint = unary_union(selected_footprints)
    aoi_area = aoi_geom.area
    covered_area = aoi_geom.intersection(combined_footprint).area
    coverage_percentage = (covered_area / aoi_area) * 100
    logger.info(f"AOI coverage: {coverage_percentage:.2f}%")
    if coverage_percentage < min_coverage_threshold:
        raise ValueError(f"Insufficient AOI coverage: {coverage_percentage:.1f}%")
    logger.info("✓ AOI coverage validation passed - proceeding with processing")

def summarize_items(signed_items, band_assets: List[str]) -> Tuple[Dict, int]:
    """Log summary of selected items and baselines"""
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
    if len(baseline_counts) > 1:
        logger.warning("WARNING: Items have different processing baselines! This may affect ML model consistency.")
        if correction_needed_count > 0 and correction_needed_count < len(signed_items):
            logger.warning("MIXED BASELINES: Some items require offset correction, others don't!")
            logger.warning("Consider filtering to a single processing baseline for consistent time series analysis.")
    return baseline_counts, correction_needed_count

def get_crs_set(signed_items, band_assets: List[str]) -> set:
    """Get set of EPSG codes from items"""
    crs_set = set()
    for item in signed_items:
        try:
            with rasterio.open(item.assets[band_assets[0]].href) as src:
                epsg_code = src.crs.to_epsg()
                crs_set.add(epsg_code)
        except Exception as e:
            logger.warning(f"Could not determine CRS for item {item.id}: {e}")
    return crs_set

def stack_items(signed_items, band_assets: List[str], resolution: int, chunksize: int, target_epsg=None) -> xr.DataArray:
    """Stack items using stackstac"""
    if target_epsg:
        stack = stackstac.stack(
            signed_items,
            assets=band_assets,
            resolution=resolution,
            chunksize=chunksize,
            epsg=target_epsg
        ).astype("float32")
    else:
        stack = stackstac.stack(
            signed_items,
            assets=band_assets,
            resolution=resolution,
            chunksize=chunksize
        ).astype("float32")
    return stack

def apply_offset_and_sr(stack: xr.DataArray, signed_items: List, band_assets: List[str]) -> Tuple[xr.DataArray, str]:
    """Apply radiometric offset correction and convert to surface reflectance"""
    requires_correction = any(get_processing_baseline_info(item)['requires_offset_correction'] for item in signed_items)
    if requires_correction:
        logger.info("Applying BOA offset correction and converting to Surface Reflectance...")
        stack = (stack + BOA_ADD_OFFSET_PB04) / QUANTIFICATION_VALUE
        data_type_info = "SR clipped [0,1] (BOA offset corrected)"
    else:
        logger.info("Converting Raw Digital Numbers to Surface Reflectance...")
        stack = stack / QUANTIFICATION_VALUE
        data_type_info = "SR clipped [0,1]"
    logger.info("Clipping surface reflectance values to [0,1] range...")
    stack = stack.clip(0, 1)
    stack = stack.where(stack >= 0, 0)
    stack = stack.where(stack != 0, np.nan)
    return stack, data_type_info

def median_mosaic(stack: xr.DataArray, band_assets: List[str], target_crs) -> xr.DataArray:
    """Compute median mosaic and assign CRS"""
    mosaic_xr = stack.median(dim='time', skipna=True)
    if 'band' in mosaic_xr.dims and len(mosaic_xr.coords['band']) == len(band_assets):
        mosaic_xr = mosaic_xr.assign_coords(band=band_assets)
    if isinstance(target_crs, str):
        from rasterio.crs import CRS
        mosaic_crs_obj = CRS.from_string(target_crs)
    else:
        mosaic_crs_obj = target_crs
    mosaic_xr = mosaic_xr.rio.write_crs(mosaic_crs_obj)
    mosaic_xr = mosaic_xr.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return mosaic_xr

def clip_to_aoi(mosaic_xr: xr.DataArray, aoi, target_crs) -> xr.DataArray:
    """Clip mosaic to AOI bounding box"""
    aoi_proj = aoi.to_crs(target_crs)
    clip_bounds = aoi_proj.total_bounds
    logger.info(f"Clipping bounds in {target_crs}: {clip_bounds}")
    x_range = clip_bounds[2] - clip_bounds[0]
    y_range = clip_bounds[3] - clip_bounds[1]
    if x_range <= 0 or y_range <= 0:
        logger.warning("Invalid clipping bounds detected, skipping clipping step")
        return mosaic_xr
    try:
        mosaic_xr = mosaic_xr.rio.clip_box(*clip_bounds)
        logger.info("Successfully clipped to AOI bounds")
    except Exception as e:
        logger.warning(f"Clipping failed: {e}. Proceeding without clipping.")
    return mosaic_xr

def reproject_and_save(mosaic_xr: xr.DataArray, output_file: str, resampling_method, compression: str, band_assets: List[str], data_type_info: str) -> None:
    """Reproject mosaic to EPSG:4326 and save as COG"""
    mosaic_wgs = mosaic_xr.rio.reproject(
        "EPSG:4326",
        resampling=resampling_method,
        nodata=0
    )
    mosaic_wgs.attrs = {
        "BANDS_USED": ", ".join(band_assets),
        "DATA_RANGE": data_type_info
    }
    mosaic_wgs = mosaic_wgs.where(~np.isnan(mosaic_wgs), 0)
    logger.info(f"Min and max values in the final output: {mosaic_wgs.min().values} to {mosaic_wgs.max().values}")
    logger.info(f"Saving to {output_file}...")
    mosaic_wgs.rio.to_raster(
        output_file,
        driver="COG",
        compress=compression,
        BIGTIFF="YES"
    )

def process_sentinel2_stack(
    city: str,
    year: int,
    basedir: str = "/data/raw/",
    start_date: str = None,
    end_date: str = None,
    max_cloud_cover: int = 15,
    chunk_size: int = 512,
    resolution: int = 10,
    resampling_method = Resampling.bilinear,
    compression: str = "LZW",
    band_assets: List[str] = None,
    min_coverage_threshold: float = 95.0
) -> str:
    """Main processing function for Sentinel-2 stack"""
    if band_assets is None:
        band_assets = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    geojson_file = os.path.join(basedir, "aoi", f"{city}_aoi.geojson")
    output_dir = os.path.join(basedir, "sentinel", city)
    os.makedirs(output_dir, exist_ok=True)
    if start_date is None:
        start_date = f"{year}-01-06"
    if end_date is None:
        end_date = f"{year}-12-18"
    output_file = os.path.join(output_dir, f"S2_{year}.tif")
    try:
        aoi, bbox = load_aoi(geojson_file)
        logger.info(f"Searching Sentinel-2 L2A imagery for {city.replace('_', ' ').title()} ({year}), date range: {start_date} to {end_date}, max cloud cover: {max_cloud_cover}%)")
        items = stac_search(bbox, start_date, end_date, max_cloud_cover)
        from shapely.ops import unary_union
        aoi_geom = aoi.geometry.unary_union
        filtered_items = filter_items_by_tile(items, aoi_geom)
        signed_items = [pc.sign(it) for it in filtered_items]
        logger.info(f"Selected {len(filtered_items)} unique tiles after filtering: {filtered_items}")
        validate_aoi_coverage(signed_items, aoi_geom, min_coverage_threshold)
        logger.info("=== SELECTED ITEMS SUMMARY ===")
        summarize_items(signed_items, band_assets)
        crs_set = get_crs_set(signed_items, band_assets)
        logger.info(f"Found CRS codes: {sorted(crs_set)}")
        if len(crs_set) > 1:
            target_epsg = sorted(crs_set)[-1]
            target_crs = f"EPSG:{target_epsg}"
            logger.warning(f"Multiple CRS detected: {sorted(crs_set)}")
            logger.info(f"Using CRS: {target_crs}")
            stack = stack_items(signed_items, band_assets, resolution, chunk_size, target_epsg)
        else:
            target_epsg = list(crs_set)[0] if crs_set else None
            target_crs = f"EPSG:{target_epsg}" if target_epsg else None
            logger.info(f"Using native CRS: {target_crs}")
            stack = stack_items(signed_items, band_assets, resolution, chunk_size)
        logger.info(f"Stack shape: {stack.shape}")
        logger.info(f"Stack dimensions: {stack.dims}")
        stack, data_type_info = apply_offset_and_sr(stack, signed_items, band_assets)
        mosaic_xr = median_mosaic(stack, band_assets, target_crs)
        mosaic_xr = clip_to_aoi(mosaic_xr, aoi, target_crs)
        del stack
        gc.collect()
        reproject_and_save(mosaic_xr, output_file, resampling_method, compression, band_assets, data_type_info)
        del mosaic_xr
        gc.collect()
        logger.info("Processing completed successfully!")
        logger.info(f"Output data range: {data_type_info}")
        logger.info(f"Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return output_file

# Example usage:
# process_sentinel2_stack("guatemala_city", 2024)
