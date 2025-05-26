# -*- coding: utf-8 -*-

import os
os.environ["SNAPPY_MEMORY"] = "16G"  # Increase if needed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#import module that is one directory up
import time
import glob
from shapely.geometry import box
import geopandas
from preprocess_s1 import preprocess_s1
from preprocess_s2 import preprocess_s2
import infrastructure.creodias
from geoprocessing import align_rasters, mosaic, harmonize
import snappy_config

# Update Java max memory in snappy.ini
# snappy_config.update_java_max_mem()
start_time = time.time()
city = 'addis_ababa'
basedir = '/data/raw/'
year = '2025'

aoi_path = os.path.join(basedir, city, f'{city}_bnd.geojson')
if not os.path.exists(aoi_path):
    raise FileNotFoundError(f"AOI file not found at {aoi_path}")

aoi = geopandas.read_file(aoi_path).to_crs(epsg=4326)

if aoi.empty:
    raise ValueError(f"AOI file {aoi_path} contains no valid geometries.")

extent = aoi.total_bounds
geometry = box(extent[0], extent[1], extent[2], extent[3])

output_dir = os.path.dirname(basedir)
os.makedirs(output_dir, exist_ok=True)


# Search Sentinel-2 scenes
api = infrastructure.creodias.ODataAPI()
s2_scenes = api.search("S2", "L2A", start_date="2025-02-01", end_date="2025-02-07", max_cloud_cover=0.1, geometry=geometry, expand=True)
sentinel2_paths = [_["S3Path"] for _ in s2_scenes["value"]]
if len(sentinel2_paths) == 0:
    raise ValueError("No Sentinel-2 scenes found for the specified date range and AOI.")
else:
    print(f"Found {len(sentinel2_paths)} Sentinel-2 scene")
    print(f"Sentinel 2 path: {sentinel2_paths}")
for path in sentinel2_paths:
    filename = path.split(os.sep)[-1]
    filename = ''.join(filename.split('.')[:-1])
    output_path = os.path.join(basedir, city, "sentinel", year, f"{filename}.tif")
    preprocess_s2(path, aoi_path, output_path)

S2_list = glob.glob(os.path.join(basedir, city, "sentinel", year, "S2*.tif")) #
sentinel2_mosaic_path = os.path.join(basedir, city, "sentinel", year,  f"S2_{year}_mosaic.tif")
print(f"Sentinel2 glob: {S2_list}")
print(f"Sentinel2 paths: {sentinel2_paths}")

if len(S2_list) > 1:
    mosaic.mosaic(S2_list, sentinel2_mosaic_path, window_size=2048, chunk_size=512, num_workers=16, 
                  scheduler="threads", nodata=0)
align_rasters.align_raster(sentinel2_mosaic_path, basedir + city + '/Density_scaled.tif', 
                           os.path.join(basedir, city, "sentinel", year, f"S2_{year}_aligned.tif"))

#check if the acquisition date is after january 20 2022
if s2_scenes["value"][0]["OriginDate"] > "2022-01-20T00:00:00Z":
    print("S2 is acquired after January 20, 2022. The image will be harmonized to match the old processing baseline.")
    harmonize.harmonize_s2(os.path.join(basedir, city, "sentinel", year, f"S2_{year}_aligned.tif"), 
                              os.path.join(basedir, city, "sentinel", year, f"S2_{year}_harmonized.tif"), scale=False)

## process S1 data
# scenes = api.search("S1", "GRD", "IW", start_date="2024-03-01", end_date="2024-03-30", geometry=geometry, expand=True)
# sentinel1_asc_paths = [
#     item["S3Path"] for item in sentinel1_scenes["value"] 
#     if any([attr["Name"] == "orbitDirection" and attr["Value"] == "ASCENDING" for attr in item["Attributes"]])
# ]
# sentinel1_desc_paths = [
#     item["S3Path"] for item in sentinel1_scenes["value"] 
#     if any([attr["Name"] == "orbitDirection" and attr["Value"] == "DESCENDING" for attr in item["Attributes"]])
# ]
# sentinel1_asc_paths, sentinel1_desc_paths


# for path in sentinel1_asc_paths:
#     filename = path.split(os.sep)[-1]
#     filename = ''.join(filename.split('.')[:-1])
#     output_path = os.path.join(output, "raster", "NZ1", "sentinel1", "asc", f"{filename}.tif")
#     preprocess_s1(path, aoi_path, output_path)

# for path in sentinel1_desc_paths:
#     filename = path.split(os.sep)[-1]
#     filename = ''.join(filename.split('.')[:-1])
#     output_path = os.path.join(output, "raster", "NZ1", "sentinel1", "desc", f"{filename}.tif")
#     preprocess_s1(path, aoi_path, output_path)

# sentinel1_asc_glob = os.path.join(data_folder, "S1", "asc", "S1*.tif")
# sentinel1_asc_mosaic_path = os.path.join(output, "S1", "asc", "mosaic.tif")

# mosaic.py {sentinel1_asc_mosaic_path} {sentinel1_asc_glob} --nodata 0;
#s1_to_db.py {sentinel1_asc_mosaic_path}

# with rasterio.open(sentinel1_asc_mosaic_path, "r") as src:
#     array = src.read()[0, ...]
#     plt.imshow(array)
# plt.show()

# sentinel1_desc_glob = os.path.join(output, "S1", "desc", "S1*.tif")
# sentinel1_desc_mosaic_path = os.path.join(output, "S1", "desc", "mosaic.tif")

#mosaic.py {sentinel1_desc_mosaic_path} {sentinel1_desc_glob} --nodata 0;
#s1_to_db.py {sentinel1_desc_mosaic_path}
# input = scenes["value"][3]["S3Path"]
# preprocess_s1(input, aoi_path, output_path)

# stack_path = os.path.join(data_folder, "raster", "NZ1", "stack.tif")
# stack.py {stack_path} {sentinel2_mosaic_path} {sentinel1_asc_mosaic_path} {sentinel1_desc_mosaic_path} {dem_mosaic_path} {slope_path}

end_time = time.time()
total_time = (end_time - start_time) / 60
print(f"Total time taken for preprocessing: {round(total_time)} minutes")