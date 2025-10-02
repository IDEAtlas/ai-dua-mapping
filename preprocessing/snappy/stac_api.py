import os
import argparse
import logging
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from typing import Dict
from pystac_client import Client
import planetary_computer as pc
import stackstac
import rioxarray
import numpy as np

# Bands to download (10 m + resampled 20 m)
BAND_ASSETS = ["B02", "B03", "B04", "B05", "B06",
               "B07", "B08", "B8A", "B11", "B12"]

# Constants for baseline correction
BOA_ADD_OFFSET_PB04 = -1000
QUANTIFICATION_VALUE = 10000

# Configure logger
logger = logging.getLogger("s2_composite")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

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

def download_s2_composite(aoi, date_range, output_path,
                          max_cloud=20, reducer="median", min_coverage=95):
    """Download Sentinel-2 L2A composite with automatic baseline correction"""
    
    # Load AOI
    geom = gpd.read_file(aoi).to_crs("EPSG:4326")
    bbox = geom.total_bounds.tolist()
    start_date, end_date = date_range
    logger.info(f"Using AOI bbox {bbox} and date range {start_date} → {end_date}")

    # Search STAC items
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    logger.info(f"Found {len(items)} Sentinel-2 items.")
    if not items:
        raise FileNotFoundError("No Sentinel-2 items found. Try increasing max_cloud or expanding the date range.")

    # Check AOI coverage
    footprints = [shape(item.geometry) for item in items]
    footprints_union = unary_union(footprints)
    aoi_union = unary_union(geom.geometry)
    coverage_fraction = footprints_union.intersection(aoi_union).area / aoi_union.area * 100
    logger.info(f"Retrieved items cover {coverage_fraction:.1f}% of the AOI.")
    if coverage_fraction < min_coverage:
        raise RuntimeError(
            f"Coverage {coverage_fraction:.1f}% < min_coverage {min_coverage}%. "
            f"Consider increasing --max_cloud or expanding the date range."
        )

    # Sign items
    signed_items = [pc.sign(item) for item in items]
    logger.info("Signed STAC items with Planetary Computer credentials.")

    # Baseline summary & mark for correction
    logger.info("=== SELECTED ITEMS SUMMARY ===")
    baseline_counts = {}
    correction_needed_count = 0
    for it in signed_items:
        info = get_processing_baseline_info(it)
        processing_baseline = info['processing_baseline']
        baseline_counts[processing_baseline] = baseline_counts.get(processing_baseline, 0) + 1
        if info['requires_offset_correction']:
            correction_needed_count += 1
            logger.info(f"Info: Item {it.id} requires baseline correction (PB {processing_baseline}).")
            it.properties["_baseline_offset"] = BOA_ADD_OFFSET_PB04
        else:
            it.properties["_baseline_offset"] = 0
        logger.info(f"  - {it.id} (cloud: {info['cloud_cover']}%, baseline: {processing_baseline}, "
                    f"requires correction: {info['requires_offset_correction']})")
    logger.info(f"Processing Baseline Distribution: {baseline_counts}")
    logger.info(f"Items requiring radiometric offset correction: {correction_needed_count}/{len(signed_items)}")
    if len(baseline_counts) > 1:
        logger.warning("Items have different processing baselines! This may affect ML model consistency.")
        if 0 < correction_needed_count < len(signed_items):
            logger.warning("MIXED BASELINES: Some items require offset correction, others don't!")
            logger.warning("Consider filtering to a single processing baseline for consistent time series analysis.")

    # Stack signed items
    da = stackstac.stack(
        signed_items,
        assets=BAND_ASSETS,
        bounds_latlon=bbox,
        epsg=4326,
        resolution=None,
        chunksize=1024
    )
    logger.info(f"Stack shape: {da.shape} (time, band, y, x)")

    # Apply baseline correction and scaling per time slice
    baseline_offsets = np.array([
        it.properties.get("_baseline_offset", 0) for it in signed_items
    ], dtype=np.float32)

    da_corrected = da.copy()
    for t_idx, offset in enumerate(baseline_offsets):
        da_corrected[t_idx, :, :, :] = (da[t_idx, :, :, :] + offset) / QUANTIFICATION_VALUE

    da_corrected = da_corrected.astype('float32').compute()

    # Composite
    if reducer == "median":
        mosaic = da_corrected.median(dim="time", keep_attrs=True)
    elif reducer == "mean":
        mosaic = da_corrected.mean(dim="time", keep_attrs=True)
    else:
        raise ValueError("Reducer must be 'median' or 'mean'")

    mosaic = mosaic.sel(band=BAND_ASSETS)
    mosaic.rio.write_crs("EPSG:4326", inplace=True)

    # Save result as float32 with DEFLATE + predictor=2
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mosaic.rio.to_raster(
        output_path,
        compress='DEFLATE',
        predictor=2
    )
    logger.info(f"✅ Composite saved at {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel-2 L2A composite with automatic baseline correction")
    parser.add_argument("--aoi", required=True, help="Path to AOI file (shapefile or GeoJSON).")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--out", required=True, help="Output GeoTIFF file path.")
    parser.add_argument("--max_cloud", type=int, default=20, help="Max cloud cover (default: 20).")
    parser.add_argument("--reducer", choices=["median","mean"], default="median", help="Composite method (default: median).")
    parser.add_argument("--min_coverage", type=float, default=95, help="Minimum required AOI coverage in percent (default: 95).")
    args = parser.parse_args()

    try:
        download_s2_composite(
            aoi=args.aoi,
            date_range=(args.start, args.end),
            output_path=args.out,
            max_cloud=args.max_cloud,
            reducer=args.reducer,
            min_coverage=args.min_coverage
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Operation aborted: {e}")
