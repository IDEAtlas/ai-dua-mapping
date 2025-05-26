from esa_snappy import ProductIO, HashMap, GPF
import geopandas as gpd
import os
import time
from pyproj import CRS
from shapely.geometry import box

def subset_aoi(product, shapefile_path):
    """Subset the Sentinel-2 product to the AOI, handling CRS mismatches and ensuring overlap."""
    print("Subsetting to AOI...")
    try:
        # Load AOI shapefile
        region_shp = gpd.read_file(shapefile_path)
        # if region_shp.crs != "EPSG:4326":
        #     region_shp = region_shp.to_crs("EPSG:4326")
        if region_shp.empty:
            raise ValueError(f"Shapefile {shapefile_path} contains no valid geometries.")
        
        # Extract CRS of Sentinel-2 product
        geo_coding = product.getSceneGeoCoding()
        product_crs = CRS.from_string(geo_coding.getMapCRS().toWKT()).to_epsg()
        print(f"Sentinel-2 Product CRS: EPSG:{product_crs}")
        
        # Reproject AOI to match Sentinel-2 CRS
        if region_shp.crs.to_epsg() != product_crs:
            print(f"Reprojecting AOI from {region_shp.crs} to EPSG:{product_crs}")
            region_shp = region_shp.to_crs(product_crs)

        # Get product bounds using GeoCoding
        scene_width = product.getSceneRasterWidth()
        scene_height = product.getSceneRasterHeight()
        upper_left = geo_coding.getGeoPos(0, 0)
        lower_right = geo_coding.getGeoPos(scene_width - 1, scene_height - 1)
        product_bbox = box(upper_left.lon, lower_right.lat, lower_right.lon, upper_left.lat)

        # Merge AOI geometries and expand slightly to avoid edge issues
        aoi_geom = region_shp.unary_union.buffer(0.0001)

        # Check if AOI intersects product bounds
        if not product_bbox.intersects(aoi_geom):
            raise ValueError("The AOI does not intersect with the Sentinel-2 product footprint. Please check your AOI geometry and product region.")

        # Convert AOI to WKT and subset
        wkt = aoi_geom.wkt
        params = HashMap()
        params.put('geoRegion', wkt)
        params.put('copyMetadata', True)
        subsetted_product = GPF.createProduct('Subset', params, product)

        # Validate subset dimensions
        if subsetted_product.getSceneRasterWidth() == 0 or subsetted_product.getSceneRasterHeight() == 0:
            raise ValueError("The AOI does not intersect the Sentinel-2 product. Please check your AOI and product bounds.")

        print("AOI Subsetting completed successfully.")
        return subsetted_product
    except Exception as e:
        raise RuntimeError(f"Error during AOI subsetting: {e}")

def preprocess_s2(input_path, output_path, aoi):
    """Preprocess Sentinel-2 product with AOI subsetting, resampling, and band selection."""
    start_time = time.time()

    # Validate input paths
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    print(f"S2 Scene Found: {input_path}")

    if not os.path.exists(aoi):
        raise FileNotFoundError(f"AOI file not found at {aoi}")
    print(f"AOI Path: {aoi}")

    # Read Sentinel-2 product
    print("Reading Sentinel-2 product...")
    product = ProductIO.readProduct(input_path)
    if product is None:
        raise RuntimeError(f"Failed to load product from {input_path}")

    # Subset to AOI
    product = subset_aoi(product, aoi)

    # Resample bands
    print("Resampling bands to 10m resolution...")
    try:
        params_resampling = HashMap()
        params_resampling.put("resolution", "10")
        params_resampling.put("upsampling", "Nearest")
        product = GPF.createProduct("S2Resampling", params_resampling, product)
        print("Resampling completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Error during resampling: {e}")

    # Subset to specified bands
    print("Subsetting to specified bands...")
    try:
        params_subset_bands = HashMap()
        params_subset_bands.put("sourceBands", "B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12")
        params_subset_bands.put("copyMetadata", True)
        product = GPF.createProduct("Subset", params_subset_bands, product)
        print("Band subsetting completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Error during band subsetting: {e}")

    # Save as GeoTIFF
    print("Saving output as GeoTIFF...")
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        ProductIO.writeProduct(product, output_path, "GeoTIFF-BigTIFF")
        print(f"GeoTIFF saved at: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error while saving GeoTIFF: {e}")

    # End time tracking
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for preprocessing: {total_time:.2f} seconds")