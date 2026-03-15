import os
import logging
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
import numpy as np
from pystac_client import Client
from odc.stac import load, configure_rio
from odc.geo.xr import write_cog
from rioxarray.merge import merge_arrays

# Configure Logger
logger = logging.getLogger(__name__)

def get_s2(city_normalized: str, year: int, basedir: str) -> str:
    """
    Downloads, masks, and mosaics Sentinel-2 imagery for a given city and year.
    Returns the path to the saved COG.
    """
    configure_rio(cloud_defaults=True)

    # 1. Setup Paths
    outdir = os.path.join(basedir, "sentinel", city_normalized)
    geojson_path = os.path.join(basedir, "aoi", f"{city_normalized}_aoi.geojson")
    output_path = os.path.join(outdir, f"S2_{year}.tif")
    os.makedirs(outdir, exist_ok=True)

    bands_of_interest = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]

    # 2. Load AOI and define Search
    gdf = gpd.read_file(geojson_path)
    bbox_wgs84 = gdf.to_crs(4326).total_bounds.tolist()
    utm_crs = gdf.estimate_utm_crs()

    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_wgs84,
        datetime=f"{year}-06-01/{year}-08-31",
        query={"eo:cloud_cover": {"lt": 50}}
    )
    items = list(search.items())
    
    if not items:
        raise ValueError(f"No STAC items found for {city_normalized} in {year}")

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

        # 6. Apply Scaling
        tile_ds = tile_ds.where(tile_ds != 0, np.nan)
        spectral_vars = [v for v in tile_ds.data_vars if v != "scl"]
        tile_ds[spectral_vars] = tile_ds[spectral_vars] / 10000.0

        # 8. Create Median Composite Masking
        # 0:NoData, 1:Defective, 3:Shadow, 8:MedProb, 9:HighProb, 10:Cirrus, 11:Snow
        mask_codes = [0, 1, 3, 8, 9, 10, 11]
        mask = tile_ds.scl.isin(mask_codes)
        tile_processed = tile_ds[spectral_vars].where(~mask)
        
        tile_da = tile_processed.to_array(dim="band")
        tiles_to_mosaic.append(tile_da)

    # Final Merge and Reprojection
    merged_native = merge_arrays(tiles_to_mosaic, nodata=np.nan)
    final_da = merged_native.rio.reproject("EPSG:4326", resampling=3, nodata=np.nan)

    # Save as COG
    write_cog(final_da, output_path, overwrite=True, nodata=np.nan)
    
    return output_path