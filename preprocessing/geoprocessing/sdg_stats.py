import argparse
import logging
import geopandas as gpd
from rasterio import features
from rasterstats import zonal_stats
import rasterio
from shapely.geometry import shape

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

def compute_sdg111_stats(gdf, pop_raster_path, formal_val, informal_val, output_path=None):
    with rasterio.open(pop_raster_path) as src:
        crs = src.crs
        nodata = src.nodata

    gdf = gdf.to_crs(crs)

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

    print("\n SDG 11.1.1 Report")
    print(f"Total built-up area: {total_area_ha:.2f} Hectare ({total_area_ha / 100:.2f} sqkm)")
    print(f"  - NDUA:| {summary.loc['formal', 'area_ha']:.2f} Hectare " "OR "
        f"{summary.loc['formal', 'area_sqkm']:.2f} sqkm | ({summary.loc['formal', 'area_pct']:.1f}%)")
    print(f"  - DUA:| {summary.loc['informal', 'area_ha']:.2f} Hectare " "OR "
        f"{summary.loc['informal', 'area_sqkm']:.2f} sqkm |  ({summary.loc['informal', 'area_pct']:.1f}%)")
    print(f"Total population in built-up: {total_population:,.0f}")
    print(f"  - NDUA: {summary.loc['formal', 'pop']:,.0f} "
          f"({summary.loc['formal', 'pop_pct']:.1f}%)")
    print(f"  - DUA: {summary.loc['informal', 'pop']:,.0f} "
          f"({summary.loc['informal', 'pop_pct']:.1f}%)\n")

    if output_path:
        print(gdf.crs)
        gdf.to_file(output_path, driver="GPKG" if output_path.endswith(".gpkg") else "GeoJSON")
        logging.info(f"Saved full GeoDataFrame to: {output_path}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Compute SDG 11.1.1 stats from a classified raster and a population raster.")
    parser.add_argument("classified_raster", type=str, help="Path to classified raster with built-up classes.")
    parser.add_argument("population_raster", type=str, help="Path to population raster.")
    parser.add_argument("--output", type=str, help="Optional output file for polygons (GeoJSON or GPKG).")
    parser.add_argument("--idx", choices=['012','123'], default='123',
                        help="Classification scheme: '012' (0=non,1=ndua,2=dua) or '123' (1=non,2=ndua,3=dua)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Set class labels based on scheme
    if args.idx == '012':
        formal_val, informal_val = 1, 2
    elif args.idx == '123':
        formal_val, informal_val = 2, 3
    else:
        raise ValueError("Invalid label scheme")

    gdf = polygonize_builtup_classes(args.classified_raster, class_values=(formal_val, informal_val))
    compute_sdg111_stats(gdf, args.population_raster, formal_val, informal_val, output_path=args.output)

if __name__ == "__main__":
    main()
