import argparse
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import os

def get_aoi_bounds(aoi_path):
    aoi_path = os.path.join(aoi_path)
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI file not found at {aoi_path}")
    aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    if aoi.empty:
        raise ValueError(f"AOI file {aoi_path} contains no valid geometries.")
    aoi_geometry = aoi.iloc[0].geometry
    return aoi_geometry.bounds

def filter_geometries(csv_gz_path, aoi_bounds, output_path):
    aoi_bbox = box(*aoi_bounds)
    df = pd.read_csv(csv_gz_path, compression='gzip')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geometry']), crs='EPSG:4326')
    filtered_gdf = gdf[gdf.intersects(aoi_bbox)]
    filtered_gdf.to_file(output_path, driver='GeoJSON')
    print(f"Filtered GeoJSON saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Filter geometries from a compressed CSV file based on an AOI bounding box and save as GeoJSON.')
    parser.add_argument('input_path', type=str, help='Path to the compressed CSV.GZ file ')
    parser.add_argument('aoi_path', type=str, help='Path to where the AOI file is located.')
    parser.add_argument('output_path', type=str, help='Path to save the filtered GeoJSON file.')
    
    args = parser.parse_args()
    aoi_bounds = get_aoi_bounds(args.aoi_path)
    filter_geometries(args.input_path, aoi_bounds, args.output_path)

if __name__ == '__main__':
    main()