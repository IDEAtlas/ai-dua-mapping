import os
import logging
import geopandas as gpd
import rioxarray
import xarray as xr
import rasterio.enums
from pystac_client import Client
import planetary_computer as pc
from shapely.geometry import box, shape
from shapely.ops import unary_union
from typing import Dict, List
from itertools import groupby
import numpy as np

logger = logging.getLogger(__name__)

def get_processing_baseline_info(item) -> Dict:
    """Extracts metadata about the Sentinel-2 processing baseline from a STAC item."""
    props = item.properties
    processing_baseline = props.get('s2:processing_baseline', 'N/A')
    info = { 'processing_baseline': processing_baseline, 'cloud_cover': props.get('eo:cloud_cover', 'N/A'), 'datetime': props.get('datetime', 'N/A'), 'mgrs_tile': props.get('s2:mgrs_tile', 'N/A'), 'requires_offset_correction': False }
    try:
        if processing_baseline != 'N/A':
            # Sentinel-2 processing baselines 04.00 and newer require a -1000 DN offset
            baseline_version = float(processing_baseline)
            info['requires_offset_correction'] = baseline_version >= 4.0
    except (ValueError, AttributeError): pass
    return info

def main(geojson_path, start_date, end_date, max_cloud_cover, output_path, output_epsg=None, bands_to_process=None):
    if bands_to_process is None:
        bands_to_process = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    
    # Static list of 10m bands for resampling logic
    bands_10m = ["B02", "B03", "B04", "B08"]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read AOI and determine optimal CRS
    try:
        aoi_wgs84 = gpd.read_file(geojson_path).to_crs(epsg=4326)
        minx, miny, maxx, maxy = aoi_wgs84.total_bounds
        search_bbox = [minx, miny, maxx, maxy]
        utm_crs = aoi_wgs84.estimate_utm_crs()
        print(f"Loaded AOI: {geojson_path}")
        print(f"Determined optimal UTM CRS for processing: {utm_crs.to_string()}")
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        exit()
    
    # Search STAC and Filter for Best Scenes
    print("Searching STAC API for Sentinel-2 L2A data...")
    API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = Client.open(API_URL)
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=search_bbox, datetime=f"{start_date}/{end_date}", query={"eo:cloud_cover": {"lt": max_cloud_cover}})
    items = list(search.item_collection())
    if not items:
        print("No items found for the given criteria. Increase cloud cover threshold or change date range.")
        exit()
    print(f"Found {len(items)} initial STAC items.")
    print("Filtering for the best scene per tile location (lowest cloud cover)...")
    items.sort(key=lambda item: (item.properties['s2:mgrs_tile'], item.properties['eo:cloud_cover'], item.datetime), reverse=False)
    best_items = []
    for key, group in groupby(items, key=lambda item: item.properties['s2:mgrs_tile']):
        best_items.append(list(group)[0])
    print(f"Filtered down to {len(best_items)} best items to process.")
    items = best_items

    # Validate coverage of selected image to AOI
    print("Validating coverage of the AOI...")
    if items:
        footprints = [shape(item.geometry) for item in items if item.geometry is not None]
        if footprints:
            coverage_area = unary_union(footprints)
            aoi_geom = aoi_wgs84.unary_union
            
            # **IMPROVEMENT**: Quantify the lack of coverage if it exists.
            if not aoi_geom.within(coverage_area):
                # To get accurate area, reproject to a global equal-area projection
                aoi_equal_area = aoi_wgs84.to_crs("EPSG:54009")
                coverage_equal_area = gpd.GeoSeries([coverage_area], crs="EPSG:4326").to_crs("EPSG:54009").unary_union
                uncovered_area = aoi_equal_area.unary_union.difference(coverage_equal_area).area
                total_aoi_area = aoi_equal_area.unary_union.area
                percent_uncovered = (uncovered_area / total_aoi_area) * 100
                print(f"\n!! WARNING: The best available scenes do not fully cover your AOI. !!")
                print(f"  - Approximately {percent_uncovered:.2f}% of the AOI is not covered.\n")
            else:
                print("  - AOI is fully covered by the selected scenes.")
        else:
            print("  - WARNING: Could not validate coverage, no valid footprints found.")

    # Process tiles
    tiles_to_mosaic = []
    print("\nProcessing tiles ...")
    for item in items:
        print(f"  - Processing item: {item.id}")
        try:
            baseline_info = get_processing_baseline_info(item)
            print(f"    - Cloud Cover: {baseline_info['cloud_cover']}%")
            print(f"    - Baseline: {baseline_info['processing_baseline']}, Correction required: {baseline_info['requires_offset_correction']}")
            
            # Filter for bands available in the item
            available_bands = [band for band in bands_to_process if band in item.assets]
            if not available_bands:
                print(f"    - WARNING: No requested bands found in item {item.id}. Skipping.")
                continue

            hrefs = {band: pc.sign(item.assets[band].href) for band in available_bands}
            
            with rioxarray.open_rasterio(hrefs[available_bands[0]]) as template_da:
                native_crs = template_da.rio.crs
                aoi_native = aoi_wgs84.to_crs(native_crs)
                minx_native, miny_native, maxx_native, maxy_native = aoi_native.total_bounds
                bbox_geom_native = gpd.GeoSeries([box(minx_native, miny_native, maxx_native, maxy_native)], crs=native_crs)
            
            clipped_bands_data = []
            for band in available_bands:
                clipped_da = (
                    rioxarray.open_rasterio(hrefs[band])
                    .astype('float32')
                    .rio.clip(bbox_geom_native.geometry, from_disk=True)
                )
                clipped_da = clipped_da.where(clipped_da != 0, np.nan).rio.write_nodata(np.nan)
                clipped_bands_data.append(clipped_da)
            
            template_10m = next((da for i, da in enumerate(clipped_bands_data) if available_bands[i] in bands_10m), clipped_bands_data[0])

            # **IMPROVEMENT**: Use bilinear resampling for better quality.
            resampled_bands = [
                band_da.rio.reproject_match(template_10m, resampling=rasterio.enums.Resampling.bilinear)
                if available_bands[i] not in bands_10m else band_da
                for i, band_da in enumerate(clipped_bands_data)
            ]
            
            tile_da = xr.concat(resampled_bands, dim=xr.DataArray(available_bands, dims="band", name="band"))
            
            if baseline_info['requires_offset_correction']:
                print("    - Applying baseline offset correction (-1000).")
                tile_da = tile_da - 1000
                tile_da = tile_da.where(tile_da >= 0, 0)

            print("    - Scaling to surface reflectance by dividing by 10000.")
            tile_da = tile_da / 10000.0

            tiles_to_mosaic.append(tile_da)
            
        except Exception as e:
            print(f"    - Failed to process item {item.id}. Error: {e}")

    # Mosaic, Reproject, Clip, and Save
    if tiles_to_mosaic:
        from rioxarray.merge import merge_arrays
        print("\nAll tiles processed. Mosaicking into a single file...")
        
        if output_epsg:
            print("Merging tiles in their native projections...")
            merged_native = merge_arrays(tiles_to_mosaic, nodata=np.nan)

            print(f"Reprojecting merged mosaic to EPSG:{output_epsg}...")
            reprojected_da = merged_native.rio.reproject(
                f"EPSG:{output_epsg}",
                resampling=rasterio.enums.Resampling.bilinear, # **IMPROVEMENT**: Use bilinear
                nodata=np.nan
            )

            print("Clipping reprojected data to final AOI extent...")
            aoi_bbox = aoi_wgs84.total_bounds  # Get bounding box: [minx, miny, maxx, maxy]
            final_da = reprojected_da.rio.clip_box(aoi_bbox[0], aoi_bbox[1], aoi_bbox[2], aoi_bbox[3])
        else:
            # **PRIMARY IMPROVEMENT**: This block is now much more efficient.
            print(f"Reprojecting all tiles to target UTM CRS ({utm_crs.to_string()}) at 10m resolution...")
            reprojected_tiles = [
                da.rio.reproject(
                    utm_crs,
                    resolution=10, # Set target resolution once, here.
                    resampling=rasterio.enums.Resampling.bilinear,
                    nodata=np.nan,
                )
                for da in tiles_to_mosaic
            ]
            
            print("Merging reprojected tiles...")
            merged_da = merge_arrays(reprojected_tiles, nodata=np.nan)
            
            print("Clipping final mosaic to AOI extent...")
            aoi_final_crs = aoi_wgs84.to_crs(merged_da.rio.crs)
            aoi_bbox = aoi_final_crs.total_bounds  # Get bounding box: [minx, miny, maxx, maxy]
            final_da = merged_da.rio.clip_box(aoi_bbox[0], aoi_bbox[1], aoi_bbox[2], aoi_bbox[3])

        # Final cleanup of no-data values that may exist outside the clip geometry
        final_da = final_da.where(final_da.notnull())
        
        print(f"Saving final raster to: {output_path}")
        final_da.rio.to_raster(
            output_path, driver='GTiff', dtype='float32', compress='DEFLATE', windowed=True, predictor=3
        )
        print(f"\nSuccessfully created mosaic: {output_path}")
    else:
        print("\nNo data was processed to create a mosaic.")
    
    print("\nProcessing complete.")


def process_sentinel2_stack(city: str, year: int, basedir: str, max_cloud_cover: int = 20, 
                            output_epsg = 4326, bands_to_process = None) -> str:
    """
    Wrapper function to process Sentinel-2 data for a given city and year.
    
    Args:
        city: City name (lowercase, underscores, e.g., "dodoma_tanzania").
        year: Year for data collection.
        basedir: Base directory for output files.
        max_cloud_cover: Maximum cloud cover percentage.
        output_epsg: Optional EPSG code for output CRS.
        bands_to_process: List of bands to process (default: all standard S2 bands).
    
    Returns:
        Path to the output GeoTIFF file.
    """
    if bands_to_process is None:
        bands_to_process = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    
    # Construct paths
    aoi_path = os.path.join(basedir, "aoi", f"{city}_aoi.geojson")
    output_path = os.path.join(basedir, "sentinel", city, f"S2_{year}.tif")
    
    # Check if AOI exists
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI GeoJSON not found at: {aoi_path}")
    
    # Call main function with parameters
    try:
        main(
            geojson_path=aoi_path,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            max_cloud_cover=max_cloud_cover,
            output_path=output_path,
            output_epsg=output_epsg,
            bands_to_process=bands_to_process
        )
    except Exception as e:
        # Check if output file was created; if not, it's a data availability issue
        if not os.path.exists(output_path):
            if "No data found in bounds" in str(e) or "No data was processed" in str(e):
                raise RuntimeError(f"No valid Sentinel-2 data available for {city} in {year}")
            else:
                raise RuntimeError(f"Sentinel-2 processing failed: {str(e)[:100]}")
        raise
    
    # Verify output file exists
    if not os.path.exists(output_path):
        raise RuntimeError(f"No valid Sentinel-2 data available for {city} in {year}")
    
    return output_path
