import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import momepy as mm
import osmnx as ox
import numpy as np
from shapely.geometry import box
import utils

import time

start_time = time.time() #gets the current time


# # df = pd.read_csv('google_buildings.csv')
# # df.head()

# # # Project to UTM
# # from shapely import wkt

# # df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
# # buildings = gpd.GeoDataFrame(df, geometry='geometry', crs = 'EPSG:4326')
# buildings = gpd.read_file('google_buildings.geojson')
# # buildings = buildings.to_crs(32721)
# if buildings.crs.is_geographic:
# 	print('Reprojecting to UTM')
# 	buildings = buildings.to_crs(buildings.estimate_utm_crs())

# print(buildings.crs)
buildings = gpd.read_parquet('/data/training_data/ancillary/buenos_aires/building_data/buenos_buildings.pq')
# print(buildings.crs)
# # Clip building into AOI
clip_extent = gpd.read_file('/data/training_data/ancillary/buenos_aires/building_data/aoi55.shp').to_crs(buildings.crs)
print("clip extent coordinate:", clip_extent.crs)

buildings = gpd.clip(buildings, clip_extent)
print(f'Total number of building footprints{len(buildings)}')

# ### Check and correct geometries
# buildings.geometry = buildings.buffer(0)
# buildings = buildings[~buildings.geometry.isna()]
# buildings = buildings.reset_index(drop=True).explode().reset_index(drop=True)

# # ## Count the number of buildings again

# buildings.geom_type.value_counts()

# # ## Fill islands within polygons

# buildings = gpd.GeoDataFrame(geometry=utils.fill_insides(buildings))
# buildings["uID"] = range(len(buildings))
# print(buildings.crs)

# buildings.to_parquet('buenos_buildings.pq')
# buildings.to_file('buildings.shp')

# buildings = gpd.read_parquet('/data/training_data/ancillary/buenos_aires/building_data/buenos_aires_buildings.pq')
# buildings = gpd.read_file('buildings.shp')

# Preview the data attribute

# buildings.head()

# Check validity of input

check = mm.CheckTessellationInput(buildings)
print('Checked validity of input')

# Tessellation
limit = mm.buffered_limit(buildings, 100)
tess = mm.Tessellation(buildings, "uID", limit, segment=2).tessellation

# Save to *.pq

tess.to_parquet('/data/training_data/ancillary/buenos_aires/building_data/tess_aoi55.pq')
tess.to_file('/data/training_data/ancillary/buenos_aires/building_data/tess_aoi55.shp')

# # Count the number of tessellation

# print(tess.shape)


# Plot the geometry

# tess.plot(figsize=(20,20))

import geopandas as gpd
import momepy as mm
from tqdm import tqdm
from momepy import limit_range
import numpy as np
import pandas as pd
from inequality.theil import Theil
import libpysal
import scipy as sp
import mapclassify
import mapclassify.classifiers as classifiers

# Read preprocessed data: buildings and tessellation

blg = buildings #gpd.read_parquet('buildings.pq')
#streets = gpd.read_parquet('edges.pq')
tess = tess #gpd.read_parquet('tessellation.pq')
#blocks = gpd.read_parquet('blocks.pq')

### Few metrics computed over building footprint, please refer to the online documentations about the functionalities

blg['sdbAre'] = mm.Area(blg).series
blg['sdbPer'] = mm.Perimeter(blg).series
blg['ssbCCo'] = mm.CircularCompactness(blg, 'sdbAre').series
blg['ssbCor'] = mm.Corners(blg).series
blg['ssbSqu'] = mm.Squareness(blg).series
blg['ssbERI'] = mm.EquivalentRectangularIndex(blg, 'sdbAre', 'sdbPer').series
blg['ssbElo'] = mm.Elongation(blg).series

### Few more metrics computed over buildings and tessellations, again, please refer to the online documentations of the functions.

blg['stbOri'] = mm.Orientation(blg).series
tess['stcOri'] = mm.Orientation(tess).series
blg['stbCeA'] = mm.CellAlignment(blg, tess, 'stbOri', 'stcOri', 'uID', 'uID').series

tess['sdcLAL'] = mm.LongestAxisLength(tess).series
tess['sdcAre'] = mm.Area(tess).series
tess['sscCCo'] = mm.CircularCompactness(tess, 'sdcAre').series
tess['sscERI'] = mm.EquivalentRectangularIndex(tess, 'sdcAre').series
tess['sicCAR'] = mm.AreaRatio(tess, blg, 'sdcAre', 'sdbAre', 'uID').series


# print(blg.head())

# print(tess.head())

### Save the computed metrics seperately, NOT along with the geometries (polygons)

blg.drop(columns='geometry').to_parquet('/data/training_data/ancillary/buenos_aires/building_data/blg_data_aoi55.parquet')
tess.drop(columns='geometry').to_parquet('/data/training_data/ancillary/buenos_aires/building_data/tess_data_aoi55.parquet')

### We can also save the computed metrics of buildings and tessellations together

merged = tess.merge(blg.drop(columns=['geometry']), on='uID')

primary = merged.drop(columns=['geometry'])
primary.to_parquet('primary.parquet')

# Many different way to save the results, according to your own preference.
# For easy visualization, we can also attach all the metrics to the building attribute table, and save it as a *.shp file for external visualization in other GIS softwares.

merged2 = blg.merge(tess.drop(columns=['geometry']), on='uID')

merged2.to_file('/data/training_data/ancillary/buenos_aires/building_data/buenos_morph_aoi55.shp')
merged2.to_parquet('/data/training_data/ancillary/buenos_aires/building_data/buenos_morph_aoi55.pq')

# print(merged2.head())


# print(merged.columns)

end_time = time.time()

elapsed_time = end_time-start_time

# Calculate hours and minutes
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Elapsed time: {int(hours)} hours {int(minutes)} minutes")
