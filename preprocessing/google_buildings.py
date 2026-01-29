#!/usr/bin/env python3
"""Google Open Buildings v3 Downloader - Downloads building polygons for custom areas."""

import argparse
import functools
import json
import multiprocessing
import logging
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import shape
import s2geometry as s2
import requests

#logger time should be hour, minute, second 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_polygon_from_geojson(geojson_path: str) -> gpd.GeoDataFrame:
    """Load polygon from GeoJSON file."""
    logger.info(f"Loading AOI from: {geojson_path}")
    aoi = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    logger.info(f"AOI bounds: {aoi.total_bounds}")
    return aoi


def get_bounding_box_s2_covering_tokens(aoi: gpd.GeoDataFrame) -> List[str]:
    """Get S2 tokens covering the region bounding box with 1km buffer."""
    logger.info("Calculating S2 cell coverage for bounding box with 1km buffer")
    
    # Get unified geometry and bounds
    aoi_geom = aoi.geometry.unary_union
    region_bounds = aoi_geom.bounds
    
    # Add 1km buffer (approximately 0.009 degrees at equator)
    buffer_degrees = 0.009
    buffered_bounds = (
        region_bounds[0] - buffer_degrees,  # min longitude
        region_bounds[1] - buffer_degrees,  # min latitude
        region_bounds[2] + buffer_degrees,  # max longitude
        region_bounds[3] + buffer_degrees   # max latitude
    )
    
    s2_lat_lng_rect = s2.S2LatLngRect_FromPointPair(
        s2.S2LatLng_FromDegrees(buffered_bounds[1], buffered_bounds[0]),
        s2.S2LatLng_FromDegrees(buffered_bounds[3], buffered_bounds[2])
    )
    coverer = s2.S2RegionCoverer()
    coverer.set_fixed_level(6)
    coverer.set_max_cells(1000000)
    tokens = [cell.ToToken() for cell in coverer.GetCovering(s2_lat_lng_rect)]
    logger.info(f"Found {len(tokens)} S2 tokens to process: {tokens}")
    return tokens

def s2_token_to_shapely_polygon(s2_token: str):
    """Convert S2 token to shapely polygon."""
    s2_cell = s2.S2Cell(s2.S2CellId_FromToken(s2_token, len(s2_token)))
    coords = []
    for i in range(4):
        s2_lat_lng = s2.S2LatLng(s2_cell.GetVertex(i))
        coords.append((s2_lat_lng.lng().degrees(), s2_lat_lng.lat().degrees()))
    return shapely.geometry.Polygon(coords)


def download_s2_token_to_memory(s2_token: str, aoi: gpd.GeoDataFrame, original_bounds: tuple) -> Optional[pd.DataFrame]:
    """Download building data for S2 token."""
    s2_cell_geometry = s2_token_to_shapely_polygon(s2_token)
    
    # Use buffered bounds for S2 token intersection check
    aoi_geom = aoi.geometry.unary_union
    buffered_bounds = aoi_geom.bounds
    buffer_degrees = 0.009
    buffered_bounds = (
        buffered_bounds[0] - buffer_degrees,
        buffered_bounds[1] - buffer_degrees,
        buffered_bounds[2] + buffer_degrees,
        buffered_bounds[3] + buffer_degrees
    )
    
    s2_bounds = s2_cell_geometry.bounds
    
    if (s2_bounds[2] < buffered_bounds[0] or s2_bounds[0] > buffered_bounds[2] or
        s2_bounds[3] < buffered_bounds[1] or s2_bounds[1] > buffered_bounds[3]):
        return None
        
    try:
        import io
        # Convert GCS path to public HTTP URL
        url = f"https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_6_gzip_no_header/{s2_token}_buildings.csv.gz"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with io.BytesIO(response.content) as bio:
            df = pd.read_csv(bio, compression='gzip', header=None,
                           names=['latitude', 'longitude', 'area_in_meters', 'confidence', 'geometry', 'full_plus_code'])
            
            # Use original bounds for final filtering (no buffer on output)
            mask = ((df['latitude'] >= original_bounds[1]) & (df['latitude'] <= original_bounds[3]) &
                    (df['longitude'] >= original_bounds[0]) & (df['longitude'] <= original_bounds[2]))
            
            filtered_df = df[mask]
            if len(filtered_df) > 0:
                logger.info(f"Token {s2_token}: Found {len(filtered_df)} buildings")
                return filtered_df
            return None
            
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        logger.warning(f"Token {s2_token}: Failed to download - {e}")
        return None


def convert_dataframe_to_geospatial(df: pd.DataFrame, output_path: str, format_type: str):
    """Convert DataFrame to geospatial format."""
    logger.info(f"Saving dataframe as {format_type.upper()}")
    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    gdf = gdf[gdf['geometry'].notna()]
    
    if format_type == 'geojson':
        gdf.to_file(output_path, driver='GeoJSON')
    elif format_type == 'gpkg':
        gdf.to_file(output_path, driver='GPKG')
    elif format_type == 'shp':
        gdf.to_file(output_path, driver='ESRI Shapefile')
    elif format_type == 'parquet':
        gdf.to_parquet(output_path)
    
    logger.info(f"Saved {len(gdf)} buildings to: {output_path}")


def download_buildings_to_memory(aoi: gpd.GeoDataFrame, num_workers: int = 4):
    """Download building data to memory."""
    aoi_geom = aoi.geometry.unary_union
    original_bounds = aoi_geom.bounds
    s2_tokens = get_bounding_box_s2_covering_tokens(aoi)
    download_s2_token_fn = functools.partial(download_s2_token_to_memory, aoi=aoi, original_bounds=original_bounds)
    
    logger.info(f"Starting download with {num_workers} workers")
    all_dataframes = []
    
    with multiprocessing.Pool(num_workers) as pool:
        for df in pool.imap_unordered(download_s2_token_fn, s2_tokens):
            if df is not None:
                all_dataframes.append(df)
    
    if all_dataframes:
        total_buildings = sum(len(d) for d in all_dataframes)
        logger.info(f"Filtering completed: {len(all_dataframes)} tokens containing a total of {total_buildings} buildings")
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        logger.warning("No buildings found in any S2 tokens")
        return None


def download_google_open_buildings(aoi_path: str, output_path: str, format_type: str = 'gpkg', num_workers: int = 4):
    """
    Download Google Open Buildings v3 data for a given AOI and save to disk.

    Args:
        aoi_path (str): Path to AOI GeoJSON file.
        output_path (str): Output file path (with or without extension).
        format_type (str): Output format ('geojson', 'gpkg', 'shp', 'parquet').
        num_workers (int): Number of parallel workers.
    """
    logger.info("Starting Google Open Buildings v3 download (programmatic call)")
    logger.info(f"Configuration: format={format_type}, workers={num_workers}")

    aoi = load_polygon_from_geojson(aoi_path)
    combined_df = download_buildings_to_memory(aoi, num_workers)

    if combined_df is not None and len(combined_df) > 0:
        extensions = {'geojson': '.geojson', 'gpkg': '.gpkg', 'shp': '.shp', 'parquet': '.parquet'}
        # Only add extension if not already present
        ext = extensions[format_type]
        if not output_path.endswith(ext):
            final_output_path = output_path + ext
        else:
            final_output_path = output_path
        convert_dataframe_to_geospatial(combined_df, final_output_path, format_type)
        logger.info("Process completed successfully")
        return final_output_path
    else:
        logger.error("No buildings found - process completed with no output")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download Google Open Buildings v3 data')
    parser.add_argument('--aoi', required=True, help='AOI file path')
    parser.add_argument('--output', '-o', required=True, help='Output file path (without extension)')
    parser.add_argument('--format', '-f', choices=['geojson', 'gpkg', 'shp', 'parquet'], 
                       default='gpkg', help='Output format')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    logger.info("Starting Google Open Buildings v3 download")
    logger.info(f"Configuration: format={args.format}, workers={args.workers}")
    
    download_google_open_buildings(
        aoi_path=args.aoi,
        output_path=args.output,
        format_type=args.format,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()