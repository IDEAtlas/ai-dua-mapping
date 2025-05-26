import geopandas as gpd
import pandas as pd

# File paths for the shapefiles
file_paths = [
    "/data/raw_data/ancillary/mumbai/building_data/part/mum_morph_aoi1.shp",
    "/data/raw_data/ancillary/mumbai/building_data/part/mum_morph_aoi2.shp",
    "/data/raw_data/ancillary/mumbai/building_data/part/mum_morph_aoi3.shp",
    "/data/raw_data/ancillary/mumbai/building_data/part/mum_morph_aoi4.shp"
]

# Read shapefiles into a single GeoDataFrame
gdfs = [gpd.read_file(path) for path in file_paths]

# Combine all GeoDataFrames into one
combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Merge geometries into a single polygon using unary_union
merged_geometry = combined_gdf.unary_union

# Convert merged geometry back to a GeoDataFrame
merged_gdf = gpd.GeoDataFrame({'geometry': [merged_geometry]}, crs=combined_gdf.crs)

# Save merged polygon to a new shapefile
merged_gdf.to_file("/data/raw_data/ancillary/mumbai/building_data/mum_morph.shp")

# Print summary
print("Merged polygon saved to 'merged_polygon.shp'")
