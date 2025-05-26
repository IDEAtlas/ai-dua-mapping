import planetary_computer
import pystac_client
# import deltalake
import shapely.geometry
import geopandas as gpd
import pandas as pd
import os


catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
collection = catalog.get_collection("ms-buildings")
# print(collection.description)

asset = collection.assets["delta"]

storage_options = {
    "account_name": asset.extra_fields["table:storage_options"]["account_name"],
    "sas_token": asset.extra_fields["table:storage_options"]["credential"],
}

city = 'addis_ababa'
basedir = '/data/raw/'

aoi_path = os.path.join(basedir, city, f'{city}_bnd.geojson')
if not os.path.exists(aoi_path):
    raise FileNotFoundError(f"AOI file not found at {aoi_path}")

# aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")
aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
if aoi.empty:
    raise ValueError(f"AOI file {aoi_path} contains no valid geometries.")
# aoi_bounds = aoi.total_bounds
aoi = aoi.iloc[0].geometry
aoi_bounds = aoi.bounds

# aoi = shapely.geometry.box(-43.25, -23.01, -43.15, -22.95)

search = catalog.search(
    collections=["ms-buildings"],
    intersects=aoi,
    query={
        "msbuildings:region": {"eq": "Ethiopia"},
        "msbuildings:processing-date": {"eq": "2023-04-25"},
    },
)

ic = search.item_collection()
print(len(ic))

import adlfs

fs = adlfs.AzureBlobFileSystem(
    **ic[0].assets["data"].extra_fields["table:storage_options"]
)

prefixes = [item.assets["data"].href for item in ic]
parts = []
for item in ic:
    parts.extend(fs.ls(item.assets["data"].href))

    
df = pd.concat(
    [
        gpd.read_parquet(f"az://{part}", storage_options=storage_options)
        for part in parts
    ]
)
# print(df.head())

gdf = gpd.GeoDataFrame(df, geometry="geometry")

# gdf.crs = (gdf.estimate_utm_crs())
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)
# gdf.to_parquet(os.path.join(basedir, city, "buildings.parquet"), engine="pyarrow")
gdf.to_file(os.path.join(basedir, city, "buildings/buildings.geojson"), driver="GeoJSON")
print('done')



