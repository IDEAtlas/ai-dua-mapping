import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import momepy as mm
import osmnx as ox
import numpy as np
from shapely.geometry import box
import utils
import logging

import geopandas as gpd
import momepy as mm
import numpy as np
import pandas as pd



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # df = pd.read_csv('google_buildings.csv')
# # df.head()
# # # Project to UTM
# # from shapely import wkt
# # df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
# # buildings = gpd.GeoDataFrame(df, geometry='geometry', crs = 'EPSG:4326')
logging.info("Reading building footprints...")

buildings = gpd.read_file('/data/raw/buildings/jakarta_bldg.gpkg')
logging.info(f'Total number of building footprints: {len(buildings)}')
logging.info(f"Original coordinate: {buildings.crs}")

if buildings.crs.is_geographic:
	logging.info('Reprojecting to UTM')
	buildings = buildings.to_crs(buildings.estimate_utm_crs())
logging.info(f"Reprojected coordinate: {buildings.crs}")

# logging.info(f"Clipping building footprints to AOI...")
# clip_extent = gpd.read_file('/data/raw/aoi/jakarta_aoi.geojson').to_crs(buildings.crs)
# buildings = gpd.clip(buildings, clip_extent)
# logging.info(f'Total number of building footprints after clipping: {len(buildings)}')

# ### Check and correct geometries
logging.info("Checking and correcting geometries...")
buildings.geometry = buildings.buffer(0)
buildings = buildings[~buildings.geometry.isna()]
# buildings = buildings.reset_index(drop=True).explode().reset_index(drop=True)
buildings = buildings.reset_index(drop=True).explode(index_parts=True).reset_index(drop=True)  # Add index_parts=True


# # ## Count the number of buildings again
logging.info(f'Total number of building footprints after cleaning: {len(buildings)}')
buildings.geom_type.value_counts()

# # ## Fill islands within polygons
logging.info("Filling islands within polygons...")
buildings = gpd.GeoDataFrame(geometry=utils.fill_insides(buildings))
# another option to fill islands
logging.info(f'Total number of building footprints after filling islands: {len(buildings)}')
buildings["uID"] = range(len(buildings))
# print(buildings.crs)

# buildings.to_parquet('buenos_buildings.pq')
# buildings.to_file('buildings.shp')

# buildings = gpd.read_parquet('/buenos_aires_buildings.pq')
# buildings = gpd.read_file('buildings.shp')

# Preview the data attribute

# buildings.head()

# Check validity of input
logging.info("Checking validity of building geometries...")
check = mm.CheckTessellationInput(buildings)
logging.info('Checked validity of input')

# Tessellation
logging.info("Creating tessellation...")
limit = mm.buffered_limit(buildings, 100)
tess = mm.Tessellation(buildings, "uID", limit, segment=2).tessellation

# Save to *.pq

# tess.to_parquet('/data/training_data/ancillary/buenos_aires/building_data/tess_aoi55.pq')
# tess.to_file('/data/training_data/ancillary/buenos_aires/building_data/tess_aoi55.shp')

# # Count the number of tessellation

# print(tess.shape)

# Read preprocessed data: buildings and tessellation

blg = buildings #gpd.read_parquet('buildings.pq')
#streets = gpd.read_parquet('edges.pq')
tess = tess #gpd.read_parquet('tessellation.pq')
#blocks = gpd.read_parquet('blocks.pq')

### Few metrics computed over building footprint, please refer to the online documentations about the functionalities
logging.info("Computing morphometric features for buildings and tessellations...")
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

logging.info("Morphometric features computed successfully.")
### Save the computed metrics seperately, NOT along with the geometries (polygons)

# blg.drop(columns='geometry').to_parquet('/data/training_data/ancillary/buenos_aires/building_data/blg_data_aoi55.parquet')
# tess.drop(columns='geometry').to_parquet('/data/training_data/ancillary/buenos_aires/building_data/tess_data_aoi55.parquet')

### We can also save the computed metrics of buildings and tessellations together
logging.info("Merging and saving results...")
merged = tess.merge(blg.drop(columns=['geometry']), on='uID')
primary = merged.drop(columns=['geometry'])
# primary.to_parquet('primary.parquet')

# Many different way to save the results, according to your own preference.
# For easy visualization, we can also attach all the metrics to the building attribute table, and save it as a *.shp file for external visualization in other GIS softwares.

merged2 = blg.merge(tess.drop(columns=['geometry']), on='uID')
merged2.to_file('/data/raw/buildings/umm/jakarta_morph.gpkg')
# merged2.to_parquet('/data/training_data/ancillary/buenos_aires/building_data/buenos_morph_aoi55.pq')

# print(merged2.head())
# print(merged.columns)
logging.info("Morphometric features saved to /data/raw/buildings/umm/jakarta_morph.gpkg.")
