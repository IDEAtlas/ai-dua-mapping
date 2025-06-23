from pystac_client import Client
import os
import geopandas
from shapely.geometry import box
import planetary_computer
import stackstac
import rich.table
import rioxarray
import shapely
from datetime import datetime
import dask.array as da
import numpy as np

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

city = 'buenos_aires'  # Change this to your desired city
basedir = '/data/raw/'

aoi = geopandas.read_file(os.path.join(basedir, 'aoi', f'{city}_aoi.geojson')).to_crs(epsg=4326)

print(f"AOI CRS: {aoi.crs}")
bbox = aoi.total_bounds

year = '2024'
time_range = f"{year}-03-02/{year}-03-28"  # Adjust the date range as needed

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 1}},
)


items = search.item_collection()
if len(items) == 0:
    raise ValueError("No Sentinel-2 scenes found for the specified date range and AOI. Adjust cloud cover or date range to find scenes")

print(f'Found {len(items)} items')

#print item id and cloud cover as a table
table = rich.table.Table("Item ID", "Cloud Cover")
for item in items:
    print(f"{item.id:<25} - {item.properties['eo:cloud_cover']}")

geoms = []
props = []

for item in items:
    geoms.append(shapely.geometry.shape(item.geometry))
    props.append(item.properties)

df = geopandas.GeoDataFrame(props, geometry=geoms, crs="epsg:4326")
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
mosaic = mosaic.compute()


baseline_change_date = datetime(2022, 1, 25)
all_after_baseline = all(
    item.datetime.date() > baseline_change_date.date()
    for item in intersecting_items
)

if all_after_baseline:
    print("Shifting DN values by -1000 due to baseline change after 2022-01-25")
    mosaic = (mosaic - 1000)
    mosaic = mosaic.rio.write_crs(stack.rio.crs)
    
# Harmonize bands to ensure they have the same CRS and resolution
aoi_proj = aoi.to_crs(stack.rio.crs)
geometry = [aoi_proj.union_all()]
mosaic = mosaic.rio.write_crs(stack.rio.crs)

# mosaic = mosaic.rio.clip(geometry, aoi_proj.crs)
mosaic = mosaic.rio.clip_box(*aoi_proj.total_bounds)# Clip using bounding box instead of exact geometry
mosaic.attrs["band_names"] = list(mosaic.band.values)
out_path = os.path.join(basedir, "sentinel", city, f"S2_{year}.tif")

#one last check of min and max values
print("Mosaic min:", mosaic.min().values)
print("Mosaic max:", mosaic.max().values)

# mosaic.rio.to_raster(out_path) #to save in UTM coordinates
mosaic_wgs = mosaic.rio.reproject("EPSG:4326")
# save the mosaic without nan values
mosaic_wgs = mosaic_wgs.where(~np.isnan(mosaic_wgs), 0)  # Replace NaN values with 0
mosaic_wgs.attrs["crs"] = "EPSG:4326"  # Set CRS to WGS84
mosaic_wgs.rio.to_raster(out_path)

print(f"Mosaic saved to {out_path}")