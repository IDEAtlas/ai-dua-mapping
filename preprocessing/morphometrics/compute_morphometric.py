import geopandas as gpd
import pandas as pd
import momepy as mm
import numpy as np
import utils  # custom module
import logging
import os


def umm(city, raw_bldg, aoi, out_dir):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    os.makedirs(out_dir, exist_ok=True)

    logging.info(f"Loading raw building footprints for {city}...")
    try:
        buildings = gpd.read_parquet(raw_bldg)
    except ValueError:
        df = pd.read_parquet(raw_bldg)
        buildings = gpd.GeoDataFrame(
            df,
            geometry=gpd.GeoSeries.from_wkt(df['geometry']),
            crs='EPSG:4326'
        )

    if buildings.crs.is_geographic:
        buildings = buildings.to_crs(buildings.estimate_utm_crs())

    # logging.info("Clipping building footprints to AOI...")
    # clip_extent = gpd.read_file(aoi).to_crs(buildings.crs)
    # buildings = gpd.clip(buildings, clip_extent)
    # logging.info(f'Total number of building footprints after clipping: {len(buildings)}')

    logging.info("Checking and correcting geometries...")
    buildings.geometry = buildings.buffer(0)
    buildings = buildings[~buildings.geometry.isna()]
    buildings = buildings.reset_index(drop=True).explode(index_parts=True).reset_index(drop=True)

    logging.info(f'Total number of building footprints after cleaning: {len(buildings)}')

    logging.info("Filling islands within polygons...")
    buildings = gpd.GeoDataFrame(geometry=utils.fill_insides(buildings))
    buildings["uID"] = range(len(buildings))
    logging.info(f'Total number of building footprints after filling islands: {len(buildings)}')

    logging.info("Checking validity of building geometries...")
    check = mm.CheckTessellationInput(buildings)
    logging.info(f"Collapsed features  : {len(check.collapse)}")
    logging.info(f"Split features      : {len(check.split)}")
    logging.info(f"Overlapping features: {len(check.overlap)}")

    problematic = set(
        list(check.collapse.index) +
        list(check.split.index) +
        list(check.overlap.index)
    )

    if problematic:
        logging.info(f"Removing {len(problematic)} problematic geometries...")
        buildings = buildings.drop(index=problematic).reset_index(drop=True)
        buildings["uID"] = range(len(buildings))  # Reassign uID after cleaning

    buildings.to_parquet(os.path.join(out_dir, f'{city}_bldg_clean.pq'))

    logging.info("Creating tessellation using functional API...")
    limit = mm.buffered_limit(buildings, buffer=100)
    tess = mm.morphological_tessellation(
        geometry=buildings.geometry,
        clip=limit,
        # shrink=0.4,
        segment=5,
        # simplify=True
    )
    tess["uID"] = range(len(tess))
    tess.to_parquet(os.path.join(out_dir, f'{city}_tess.pq'))

    blg = buildings

    logging.info("Computing morphometric features for buildings and tessellations...")

    # Building metrics
    blg['sdbAre'] = blg.geometry.area
    blg['sdbPer'] = blg.geometry.length
    blg['ssbCCo'] = mm.circular_compactness(blg.geometry)
    blg['ssbCor'] = mm.corners(blg.geometry)
    blg['ssbSqu'] = mm.squareness(blg.geometry)
    blg['ssbERI'] = mm.equivalent_rectangular_index(blg.geometry)
    blg['ssbElo'] = mm.elongation(blg.geometry)

    # Orientation and alignment
    blg['stbOri'] = mm.orientation(blg.geometry)
    tess['stcOri'] = mm.orientation(tess.geometry)

    # Align by uID for cell alignment
    blg["stbCeA"] = mm.cell_alignment(
        blg.set_index("uID")["stbOri"],
        tess.set_index("uID")["stcOri"]
    )

    # Tessellation metrics
    tess['sdcLAL'] = mm.longest_axis_length(tess.geometry)
    tess['sdcAre'] = tess.geometry.area
    tess['sscCCo'] = mm.circular_compactness(tess.geometry)
    tess['sscERI'] = mm.equivalent_rectangular_index(tess.geometry)

    # Area ratio (manual calculation)
    tess["sicCAR"] = blg.set_index("uID")["sdbAre"] / tess.set_index("uID")["sdcAre"]

    logging.info("Morphometric features computed successfully.")

    logging.info("Merging and saving results...")
    merged = tess.merge(blg.drop(columns=['geometry']), on='uID')
    logging.info(f"Created {len(merged)} records")

    merged2 = blg.merge(tess.drop(columns=['geometry']), on='uID')
    merged2.to_parquet(os.path.join(out_dir, f'{city}_morph.pq'))
    merged2.to_file(os.path.join(out_dir, f'{city}_morph.gpkg'))

    logging.info(f"Morphometric features saved to {os.path.join(out_dir, f'{city}_morph.gpkg')}.")


if __name__ == "__main__":
    umm(
        city='buenos_aires',
        raw_bldg='buenos_aires_clipped.pq',
        aoi='../partidos_amba_IA_BID_2025+AMBA_IDE.geojson',
        out_dir='.'
    )