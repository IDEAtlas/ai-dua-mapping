from pystac_client import Client
import os
import geopandas
from shapely.geometry import box
import planetary_computer
import stackstac
import rich.table
import rioxarray
import shapely
# import xarray as xr
import dask.array as da
import numpy as np

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

city = 'mexico_city'
basedir = '/data/raw/'

aoi = geopandas.read_file(os.path.join(basedir, city, f'{city}_bnd.geojson')).to_crs(epsg=4326)
bbox = aoi.total_bounds


year = '2020'
time_range = "2020-05-01/2020-05-07"

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 5}},
)
items = search.item_collection()
print(f'Found {len(items)} items')

#print item id and cloud cover as a table
table = rich.table.Table("Item ID", "Cloud Cover")
for item in items:
    print(f"{item.id:<25} - {item.properties['eo:cloud_cover']}")

df = geopandas.GeoDataFrame.from_features(items.to_dict(), crs="epsg:4326")
# print(df.head())

selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
# print selected item corrdinate system
print(selected_item.properties["proj:code"])

# Filter items that actually intersect with the AOI geometry
intersecting_items = [
    item for item in items
    if shapely.geometry.shape(item.geometry).intersects(aoi.union_all())
]

if not intersecting_items:
    raise ValueError("No Sentinel-2 scenes intersect the AOI.")

print(f"{len(intersecting_items)} intersecting items")

# Sign items with temporary credentials
signed_items = [planetary_computer.sign(item) for item in intersecting_items]

stack = (stackstac.stack(
    signed_items,
    assets=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    epsg=int(str(selected_item.properties["proj:code"]).replace("EPSG:", "")),
    resolution=10,
    bounds_latlon=bbox,
    chunksize=2048,)
)

# mosaic = stack.max("time")
mosaic = stack.reduce(da.nanmedian, dim="time")

aoi_proj = aoi.to_crs(stack.rio.crs)
geometry = [aoi_proj.union_all()]
mosaic = mosaic.rio.write_crs(stack.rio.crs)
# mosaic = mosaic.rio.clip(geometry, aoi_proj.crs)
mosaic = mosaic.rio.clip_box(*aoi_proj.total_bounds)# Clip using bounding box instead of exact geometry

out_path = os.path.join(basedir, city, "sentinel", year, f"S2_{year}_mosaic.tif")
mosaic.rio.to_raster(out_path)

print(f"Mosaic saved to {out_path}")

