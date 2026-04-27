import os
import logging
import geopandas as gpd
import numpy as np
from pystac_client import Client
from odc.stac import load, configure_rio
from odc.geo.xr import write_cog
from rioxarray.merge import merge_arrays

logger = logging.getLogger(__name__)

def get_s2(city_normalized: str, year: int, basedir: str) -> str:
    """
    Downloads, masks, and mosaics Sentinel-2 imagery for a given city and year.
    Automatically adapts search criteria to find the clearest available data.
    """
    configure_rio(cloud_defaults=True)

    # 1. Setup Paths
    outdir = os.path.join(basedir, "sentinel", city_normalized)
    geojson_path = os.path.join(basedir, "aoi", f"{city_normalized}_aoi.geojson")
    output_path = os.path.join(outdir, f"S2_{year}.tif")
    os.makedirs(outdir, exist_ok=True)

    bands_of_interest = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]

    # 2. Load AOI
    gdf = gpd.read_file(geojson_path)
    bbox_wgs84 = gdf.to_crs(4326).total_bounds.tolist()
    utm_crs = gdf.estimate_utm_crs()

    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    
    # 3. Progressive Search Tiers (5% to 40% cloud cover)
    tiers = [5, 10, 15, 20, 30, 40]
    items = []
    
    for max_cloud in tiers:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox_wgs84,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": max_cloud}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
        )
        found_items = list(search.items())
        
        if len(found_items) >= 10:
            items = found_items[:35]
            logger.info(f"Found {len(items)} items at {max_cloud}% cloud cover threshold.")
            break
        items = found_items

    if not items:
        raise ValueError(f"No STAC items found even at 40% cloud cover for {city_normalized}. Please increase the cloud cover percentage in preprocessing/stac_api.py file")

    # 4. Process Tiles Individually
    tiles_to_mosaic = []
    for item in items:
        tile_ds = load(
            [item],
            bands=bands_of_interest + ["scl"],
            bbox=bbox_wgs84,
            crs=utm_crs,
            resolution=10,
            chunks={} 
        ).astype("float32").squeeze("time")

        # 6. Apply Harmonization (Scaling)
        tile_ds = tile_ds.where(tile_ds != 0, np.nan)
        spectral_vars = [v for v in tile_ds.data_vars if v != "scl"]
        tile_ds[spectral_vars] = tile_ds[spectral_vars] / 10000.0

        # 8. Create Median Composite
        mask = tile_ds.scl.isin([0, 1, 3, 8, 9, 10, 11])
        tile_processed = tile_ds[spectral_vars].where(~mask)
        
        tiles_to_mosaic.append(tile_processed.to_array(dim="band"))

    # Final Merge and Reprojection
    merged_native = merge_arrays(tiles_to_mosaic, nodata=np.nan)
    final_da = merged_native.rio.reproject("EPSG:4326", resampling=3, nodata=np.nan)

    # Save as COG
    # write_cog(final_da, output_path, overwrite=True, nodata=np.nan)
    final_da.rio.to_raster(
        output_path,
        driver="COG",
        compress="deflate",
        nodata=np.nan
    )
    return output_path