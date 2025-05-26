import argparse
import geopandas as gpd
import rioxarray
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import box
from dask.diagnostics import ProgressBar


def create_reference(aoi, ghsl, duas, output):

    """
    This script processes geospatial data to create a reference label raster with three classes:

    1. Built-up areas based on GHSL built-up fraction data.
    2. Locations of DUAs (Deprived Urban Areas), represented as polygons.
    3. Non Built-up areas where neither condition is met.

    Steps:
    - Reads the AOI (area of interest) and DUAs GeoJSON files.
    - Reprojects the data to WGS84 (if necessary).
    - Loads and clips the GHSL raster to the bounding box of the AOI.
    - Binarizes the raster: values > 15 are classified as built-up, others as background.
    - Rasterizes the DUAs polygons as a separate class.
    - Combines the rasterized DUAs with the binary raster to create three classes.
    - Saves the resulting reference label raster to the specified output file.

    Inputs:
    - GHSL built-up fraction raster file.
    - AOI GeoJSON file (area of interest).
    - DUAs GeoJSON file (Defined Urban Areas).

    Output:
    - A GeoTIFF file containing the reference label raster.

    Arguments:
    - `--aoi`: Path to the AOI GeoJSON file.
    - `--ghsl`: Path to the GHSL raster file.
    - `--duas`: Path to the DUAs GeoJSON file.
    - `--output`: Path to save the output GeoTIFF raster.

    """

    pbar = ProgressBar()
    pbar.register()

    city_bnd = gpd.read_file(aoi)
    duas = gpd.read_file(duas)

    if city_bnd.crs != "EPSG:4326":
        city_bnd = city_bnd.to_crs("EPSG:4326")
    if duas.crs != "EPSG:4326":
        duas = duas.to_crs("EPSG:4326")

    raster = rioxarray.open_rasterio(ghsl, chunks={"x": 512, "y": 512})
    if raster.rio.crs != "EPSG:4326":
        raster = raster.rio.reproject("EPSG:4326")

    aoi_bounds = city_bnd.total_bounds  # [minx, miny, maxx, maxy]
    # aoi_extent = box(*aoi_bounds)
    clipped_raster = raster.rio.clip_box(*aoi_bounds)

    # Binarize the Raster (Values > 15 -> 1, Values <= 15 -> 0)
    binary_raster = clipped_raster.where(clipped_raster > 15, 0)
    binary_raster = binary_raster.where(binary_raster == 0, 1)

    # Rasterize DUAS (Class Three - Index 2)
    transform = binary_raster.rio.transform()
    polygon_mask = rasterize(
        [(geom, 2) for geom in duas.geometry],  # Assign value 2 to DUAS polygons
        out_shape=(binary_raster.rio.height, binary_raster.rio.width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    # Overlay the polygon mask onto the binary raster
    binary_raster.values[0] = np.where(polygon_mask > 0, polygon_mask, binary_raster.values[0])

    binary_raster.rio.to_raster(output, compress="lzw", dtype="uint8")
    print(f"Reference mask saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a reference label raster with three classes.")
    parser.add_argument("--aoi", required=True, help="Path to the AOI GeoJSON file")
    parser.add_argument("--ghsl", required=True, help="Path to the GHSL raster file")
    parser.add_argument("--duas", required=True, help="Path to the DUAS GeoJSON file")
    parser.add_argument("--output", required=True, help="Path to save the output raster")

    args = parser.parse_args()

    create_reference(args.aoi, args.ghsl, args.duas, args.output)
