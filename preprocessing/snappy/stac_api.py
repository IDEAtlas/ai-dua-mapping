import os
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

def get_processing_baseline_info(item) -> Dict:
    """Extracts metadata about the Sentinel-2 processing baseline from a STAC item."""
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
            baseline_version = float(processing_baseline)
            info['requires_offset_correction'] = baseline_version >= 4.0
    except (ValueError, AttributeError):
        pass
    return info

def download_s2(
    aoi_geojson: str,
    start_date: str,
    end_date: str,
    output_path: str,
    max_cloud_cover: float = 10,
    output_epsg: int = None,
    bands_to_process: List[str] = None
) -> str:
    """
    Programmatically download and mosaic Sentinel-2 L2A data from Planetary Computer.
    
    Args:
        aoi_geojson: Path to AOI GeoJSON file.
        start_date: Start date, e.g., '2023-01-01'.
        end_date: End date, e.g., '2023-01-31'.
        output_path: Full path to save the output GeoTIFF.
        max_cloud_cover: Maximum cloud cover percentage.
        output_epsg: EPSG code for output CRS. If None, optimal UTM zone is used.
        bands_to_process: List of Sentinel-2 bands to process. Defaults to standard 10/20m bands.
    
    Returns:
        str: Path to the saved GeoTIFF.
    """
    if bands_to_process is None:
        bands_to_process = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    bands_10m = ["B02", "B03", "B04", "B08"]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load AOI and determine CRS
    aoi_wgs84 = gpd.read_file(aoi_geojson).to_crs(epsg=4326)
    utm_crs = aoi_wgs84.estimate_utm_crs()
    minx, miny, maxx, maxy = aoi_wgs84.total_bounds
    search_bbox = [minx, miny, maxx, maxy]
    
    # Search STAC
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=search_bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )
    items = list(search.get_all_items())
    if not items:
        raise RuntimeError("No Sentinel-2 items found for the given criteria.")

    # Filter best scene per tile
    items.sort(key=lambda item: (item.properties['s2:mgrs_tile'], item.properties['eo:cloud_cover'], item.datetime))
    best_items = [list(group)[0] for key, group in groupby(items, key=lambda item: item.properties['s2:mgrs_tile'])]
    items = best_items

    # Validate AOI coverage
    footprints = [shape(item.geometry) for item in items if item.geometry is not None]
    coverage_area = unary_union(footprints)
    aoi_geom = aoi_wgs84.unary_union
    if not aoi_geom.within(coverage_area):
        aoi_equal_area = aoi_wgs84.to_crs("EPSG:54009")
        coverage_equal_area = gpd.GeoSeries([coverage_area], crs="EPSG:4326").to_crs("EPSG:54009").unary_union
        uncovered_area = aoi_equal_area.unary_union.difference(coverage_equal_area).area
        total_aoi_area = aoi_equal_area.unary_union.area
        percent_uncovered = (uncovered_area / total_aoi_area) * 100
        print(f"WARNING: ~{percent_uncovered:.2f}% of AOI not covered.")

    # Process tiles
    tiles_to_mosaic = []
    for item in items:
        baseline_info = get_processing_baseline_info(item)
        available_bands = [b for b in bands_to_process if b in item.assets]
        if not available_bands:
            continue

        hrefs = {band: pc.sign(item.assets[band].href) for band in available_bands}
        with rioxarray.open_rasterio(hrefs[available_bands[0]]) as template_da:
            native_crs = template_da.rio.crs
            aoi_native = aoi_wgs84.to_crs(native_crs)
            minx_native, miny_native, maxx_native, maxy_native = aoi_native.total_bounds
            bbox_geom_native = gpd.GeoSeries([box(minx_native, miny_native, maxx_native, maxy_native)], crs=native_crs)

        clipped_bands_data = []
        for i, band in enumerate(available_bands):
            clipped_da = (
                rioxarray.open_rasterio(hrefs[band])
                .astype('float32')
                .rio.clip(bbox_geom_native.geometry, from_disk=True)
            )
            clipped_da = clipped_da.where(clipped_da != 0, np.nan).rio.write_nodata(np.nan)
            clipped_bands_data.append(clipped_da)

        template_10m = next((da for i, da in enumerate(clipped_bands_data) if available_bands[i] in bands_10m), clipped_bands_data[0])
        resampled_bands = [
            band_da.rio.reproject_match(template_10m, resampling=rasterio.enums.Resampling.bilinear)
            if available_bands[i] not in bands_10m else band_da
            for i, band_da in enumerate(clipped_bands_data)
        ]
        tile_da = xr.concat(resampled_bands, dim=xr.DataArray(available_bands, dims="band", name="band"))

        if baseline_info['requires_offset_correction']:
            tile_da = tile_da - 1000
            tile_da = tile_da.where(tile_da >= 0, 0)
        tile_da = tile_da / 10000.0
        tiles_to_mosaic.append(tile_da)

    if tiles_to_mosaic:
        from rioxarray.merge import merge_arrays
        if output_epsg:
            merged_native = merge_arrays(tiles_to_mosaic, nodata=np.nan)
            final_da = merged_native.rio.reproject(
                f"EPSG:{output_epsg}",
                resampling=rasterio.enums.Resampling.bilinear,
                nodata=np.nan
            ).rio.clip(aoi_wgs84.geometry, drop=True)
        else:
            reprojected_tiles = [
                da.rio.reproject(
                    utm_crs,
                    resolution=10,
                    resampling=rasterio.enums.Resampling.bilinear,
                    nodata=np.nan
                )
                for da in tiles_to_mosaic
            ]
            merged_da = merge_arrays(reprojected_tiles, nodata=np.nan)
            final_da = merged_da.rio.clip(aoi_wgs84.to_crs(merged_da.rio.crs).geometry, drop=True, from_disk=True)

        final_da = final_da.where(final_da.notnull())
        final_da.rio.to_raster(output_path, driver='GTiff', dtype='float32', compress='DEFLATE', windowed=True, predictor=3)

    return output_path
