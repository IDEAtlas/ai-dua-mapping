import argparse
import logging
import geopandas as gpd
from rasterio import features
from rasterstats import zonal_stats
import rasterio
from shapely.geometry import shape
import os
import glob
from preprocessing import fetch_ghsl, adm_boundaries

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def polygonize_builtup_classes(classified_raster_path, class_values):
    with rasterio.open(classified_raster_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    polygons = []
    labels = []

    for val in class_values:
        mask = data == val
        shapes = features.shapes(data, mask=mask, transform=transform)

        for geom, raster_val in shapes:
            polygons.append(shape(geom))
            labels.append(val)

    gdf = gpd.GeoDataFrame({'class': labels, 'geometry': polygons}, crs=crs)
    return gdf

def compute_sdg111_stats(gdf, pop_raster_path, formal_val, informal_val, city, country, year, output_path=None):
    with rasterio.open(pop_raster_path) as src:
        crs = src.crs
        nodata = src.nodata
    if gdf.crs != crs:
        logger.info(f"Reprojecting GeoDataFrame for accurate area calculations.")
        gdf = gdf.to_crs(crs)
    else:
        logger.info("GeoDataFrame CRS matches population raster CRS.")
        logger.info(f'CRS: {crs}')
    # Area and population per polygon
    gdf['area_m2'] = gdf.geometry.area
    stats = zonal_stats(gdf, pop_raster_path, stats='sum', nodata=nodata)
    gdf['pop'] = [s['sum'] if s['sum'] is not None else 0 for s in stats]

    # Aggregate by class
    summary = gdf.groupby('class').agg({
        'area_m2': 'sum',
        'pop': 'sum'
    }).rename(index={formal_val: 'formal', informal_val: 'informal'})

    summary['area_ha'] = summary['area_m2'] / 10_000
    summary['area_sqkm'] = summary['area_m2'] / 1_000_000
    summary['area_pct'] = 100 * summary['area_m2'] / summary['area_m2'].sum()
    summary['pop_pct'] = 100 * summary['pop'] / summary['pop'].sum()

    # Report
    total_area_ha = summary['area_ha'].sum()
    total_population = summary['pop'].sum()

    print("\n" + "="*70)
    print(f"SDG 11.1.1 REPORT: {city}, {country}, {year}")
    print("="*70)
    
    print(f"\n BUILT-UP AREA")
    print(f"  Total:     {total_area_ha / 100:.2f} km²")
    print(f"  ├─ NDUA:   {summary.loc['formal', 'area_sqkm']:>8.2f} km²  {summary.loc['formal', 'area_pct']:>6.1f}%")
    print(f"  └─ DUA:    {summary.loc['informal', 'area_sqkm']:>8.2f} km²  {summary.loc['informal', 'area_pct']:>6.1f}%")
    
    print(f"\n POPULATION IN BUILT-UP AREAS")
    print(f"  Total:     {total_population:>16,.0f}")
    print(f"  ├─ NDUA:   {summary.loc['formal', 'pop']:>16,.0f}  {summary.loc['formal', 'pop_pct']:>6.1f}%")
    print(f"  └─ DUA:    {summary.loc['informal', 'pop']:>16,.0f}  {summary.loc['informal', 'pop_pct']:>6.1f}%")
    print("="*70 + "\n")

    if output_path:
        print(gdf.crs)
        gdf.to_file(output_path, driver="GPKG" if output_path.endswith(".gpkg") else "GeoJSON")
        logging.info(f"Saved full GeoDataFrame to: {output_path}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Compute SDG 11.1.1 statistics from classified building raster.")
    parser.add_argument("--city", type=str, required=True, help="City name")
    parser.add_argument("--country", type=str, required=True, help="Country name")
    parser.add_argument("--year", type=int, required=True, help="Year of classification")
    parser.add_argument("--output", type=str, help="Optional output file for polygons (GeoJSON or GPKG).")
    parser.add_argument("--idx", choices=['012', '123'], default='123',
                        help="Classification scheme: '012' (0=non,1=ndua,2=dua) or '123' (1=non,2=ndua,3=dua)")
    args = parser.parse_args()

    # Normalize city name
    city_normalized = f"{args.city}_{args.country}".lower().replace(" ", "_").replace("-", "_")
    
    # Find classified raster
    classified_raster_path = f"./output/{city_normalized}.s2.bd.mbcnn.{args.year}.tif"
    if not os.path.exists(classified_raster_path):
        logger.error(f"Classified raster not found: {classified_raster_path}")
        return
    
    # Find population raster (GHSL)
    pop_raster_pattern = f"./data/raw/ghsl/pop/{city_normalized[:3].upper()}_GHS_POP_*.tif"
    pop_raster_files = glob.glob(pop_raster_pattern)
    if not pop_raster_files:
        logger.info(f"Population raster not found. Downloading from GHSL...")
        # Download population data if not found
        try:
            aoi_geojson = f"./data/raw/aoi/{city_normalized}_aoi.geojson"
            if not os.path.exists(aoi_geojson):
                logger.error(f"AOI file not found: {aoi_geojson}")
                return
            downloader = fetch_ghsl.GHSLDownloader(temp_dir="./data/raw/ghsl/temp")
            downloaded_files = downloader.download_tiles(aoi_geojson=aoi_geojson,
                                                         output_dir="./data/raw/ghsl", 
                                                         data_type="pop")
            pop_raster_files = glob.glob(pop_raster_pattern)
            if not pop_raster_files:
                logger.error(f"Failed to download population raster")
                return
        except Exception as e:
            logger.error(f"Population raster download failed: {e}")
            return
    
    population_raster_path = pop_raster_files[0]
    
    logger.info(f"SDG 11.1.1 STATISTICS - {args.city}, {args.country} ({args.year})")
    logger.info(f"Classified raster: {classified_raster_path}")
    logger.info(f"Population raster: {population_raster_path}")
    
    # Set class labels based on scheme
    if args.idx == '012':
        formal_val, informal_val = 1, 2
    elif args.idx == '123':
        formal_val, informal_val = 2, 3
    else:
        raise ValueError("Invalid label scheme")

    gdf = polygonize_builtup_classes(classified_raster_path, class_values=(formal_val, informal_val))
    compute_sdg111_stats(gdf, population_raster_path, formal_val, informal_val, args.city, args.country, args.year, output_path=args.output)

if __name__ == "__main__":
    main()
