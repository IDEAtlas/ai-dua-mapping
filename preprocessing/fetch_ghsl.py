import os
import logging
import requests
import zipfile
import geopandas as gpd
import shutil
import weakref
import rioxarray
import xarray as xr
import subprocess
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)

# Source for the links: https://human-settlement.emergency.copernicus.eu/download.php

class GHSLDownloader:
    # Initialize with temp directory for downloads and unzipping
    def __init__(self, temp_dir: str = "/tmp/ghsl", delete_on_destruct: bool = True):
        self.temp_dir = temp_dir
        self.delete_on_destruct = delete_on_destruct
        os.makedirs(self.temp_dir, exist_ok=True)
        self.shp_url = "https://ghsl.jrc.ec.europa.eu/download/GHSL_data_54009_shapefile.zip"
        self.delete_on_destruct = delete_on_destruct
        self.zip_path = os.path.join(self.temp_dir, "GHSL_data_54009_shapefile.zip")
        response = requests.get(self.shp_url, stream=True, verify=False)
        response.raise_for_status()
        with open(self.zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        self.shapefile_dir = os.path.join(os.path.dirname(self.zip_path), "GHSL_shapefile")
        os.makedirs(self.shapefile_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.shapefile_dir)

        # Register safe cleanup if requested
        if delete_on_destruct:
            self._finalizer = weakref.finalize(
                self,
                self._cleanup_and_log,
                temp_dir  # capture as argument (not self!)
            )

    # Destructor to clean up temp files
    @staticmethod
    def _cleanup_and_log(temp_dir: str):
        """Safely remove the temporary directory."""
        import os, shutil  # import locally to avoid NoneType issues at shutdown
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # Get intersecting tile codes
    def get_intersecting_tile_codes(self, aoi_geojson: str) -> list:
        if not self.shapefile_dir:
            raise RuntimeError("Shapefile directory not set. Unzip shapefile first.")
        aoi = gpd.read_file(aoi_geojson)

        shp_files = [f for f in os.listdir(self.shapefile_dir) if f.lower().endswith('.shp')]
        if not shp_files:
            raise FileNotFoundError("No .shp files found in the unzipped directory.")
        elif len(shp_files) > 1:
            raise RuntimeError("Multiple .shp files found; expected only one.")
        shapefile_path = os.path.join(self.shapefile_dir, shp_files[0])

        shapefile = gpd.read_file(shapefile_path)

        if aoi.crs != shapefile.crs:
            aoi = aoi.to_crs(shapefile.crs)

        intersecting_tiles = gpd.overlay(aoi, shapefile, how='intersection')
        if intersecting_tiles.empty:
            raise ValueError("No intersecting tiles found between AOI and GHSL shapefile.")
        else:
            print(f"Found {len(intersecting_tiles)} intersecting tiles.")
            tile_codes = intersecting_tiles['tile_id'].unique().tolist()
            print(f"Intersecting tile codes: {tile_codes}")
            return tile_codes

    def download_tiles(self, aoi_geojson: str, output_dir: str = None, data_type: str = "built") -> list:
        """
        Download GHSL tiles for the given AOI.
        
        Args:
            aoi_geojson: Path to the AOI GeoJSON file
            output_dir: Output directory for downloaded files
            data_type: Type of data to download - "built", "pop", or "pop, built"
        
        Returns:
            List of downloaded file paths
        """
        # Validate data_type parameter
        if data_type not in ["built", "pop", "pop, built"]:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'built', 'pop', or 'pop, built'.")
        
        # Determine which products to download
        if data_type == "pop, built":
            products = ["pop", "built"]
        else:
            products = [data_type]
        
        all_downloaded_files = []
        
        # Download each product
        for product in products:
            all_downloaded_files.extend(self._download_product(aoi_geojson, output_dir, product))
        
        return all_downloaded_files
    
    def _download_product(self, aoi_geojson: str, output_dir: str, product: str) -> list:
        """
        Internal method to download a single GHSL product.
        
        Args:
            aoi_geojson: Path to the AOI GeoJSON file
            output_dir: Output directory for downloaded files
            product: Product type - "built" or "pop"
        
        Returns:
            List of downloaded file paths for this product
        """
        # Get intersecting tile codes
        tile_codes = self.get_intersecting_tile_codes(aoi_geojson)

        # Generate a prefix with the first 3 letters of the AOI file name in capital letters
        aoi_prefix = os.path.basename(aoi_geojson)[:3].upper()

        # Select URL and filename pattern based on product
        if product == "built":
            base_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10/V1-0/tiles"
            file_pattern = "GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_{tile_code}.zip"
            product_subdir = "built"
        else:  # pop
            base_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
            file_pattern = "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_{tile_code}.zip"
            product_subdir = "pop"
        
        if output_dir is None:
            output_dir = self.temp_dir
        
        # Create product-specific subdirectory
        product_output_dir = os.path.join(output_dir, product_subdir)
        os.makedirs(product_output_dir, exist_ok=True)
        downloaded_files = []

        print(f"Downloading {product.upper()} data...")
        for tile_code in tile_codes:
            file_name = file_pattern.format(tile_code=tile_code)
            url = f"{base_url}/{file_name}"
            local_zip_path = os.path.join(self.temp_dir, file_name)
            print(f"Downloading tile {tile_code} from {url}")
            response = requests.get(url, stream=True, verify=False)
            if response.status_code == 200:
                with open(local_zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded to {local_zip_path}")

                # unzip the downloaded file and delete the zip
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
                os.remove(local_zip_path)

                # add the extracted tif path to the list downloaded_files
                tif_name = file_name.replace('.zip', '.tif')

                #rename the tif to include the AOI prefix
                new_tif_name = f"{aoi_prefix}_{tif_name}"
                new_tif_path = os.path.join(self.temp_dir, new_tif_name)
                os.rename(os.path.join(self.temp_dir, tif_name), new_tif_path)

                tif_path = os.path.join(self.temp_dir, new_tif_name)

                # Move to product_output_dir if different from temp_dir
                if product_output_dir != self.temp_dir:
                    shutil.move(tif_path, os.path.join(product_output_dir, new_tif_name))
                    tif_path = os.path.join(product_output_dir, new_tif_name)

                if os.path.exists(tif_path):
                    downloaded_files.append(tif_path)
            else:
                print(f"Failed to download tile {tile_code}: HTTP {response.status_code}")

        # Post-process: clip to AOI bounds and mosaic if multiple tiles
        downloaded_files = self._post_process_tiles(aoi_geojson, downloaded_files, product_output_dir)
        
        return downloaded_files

    def _post_process_tiles(self, aoi_geojson: str, tile_files: list, output_dir: str) -> list:
        """
        Post-process downloaded GHSL tiles:
        1. Clip each tile to AOI bounds
        2. Delete source unclipped tiles
        3. If multiple clipped tiles, merge using gdal_merge
        
        Args:
            aoi_geojson: Path to the AOI GeoJSON file
            tile_files: List of downloaded tile file paths
            output_dir: Output directory where tiles are stored
        
        Returns:
            List with either single clipped tile or merged file
        """
        if not tile_files:
            return tile_files
        
        # Load AOI
        aoi = gpd.read_file(aoi_geojson)
        
        # Clip each tile to AOI bounds
        clipped_files = []
        for tile_file in tile_files:
            try:
                # Open raster to get CRS
                raster = rioxarray.open_rasterio(tile_file)
                raster_crs = raster.rio.crs
                
                # Transform AOI to raster CRS if needed
                if aoi.crs != raster_crs:
                    aoi_reprojected = aoi.to_crs(raster_crs)
                    clip_bounds = aoi_reprojected.total_bounds
                else:
                    clip_bounds = aoi.total_bounds
                
                # Clip to bounds
                clipped = raster.rio.clip_box(*clip_bounds)
                
                # Create clipped filename
                base_name = os.path.basename(tile_file).replace('.tif', '_clipped.tif')
                clipped_path = os.path.join(output_dir, base_name)
                
                # Save clipped tile
                clipped.rio.to_raster(clipped_path)
                clipped_files.append(clipped_path)
                print(f"Clipped {os.path.basename(tile_file)} -> {base_name}")
                
                # Delete original unclipped file
                try:
                    os.remove(tile_file)
                    print(f"Deleted unclipped source: {os.path.basename(tile_file)}")
                except Exception as e:
                    logger.warning(f"Failed to delete {tile_file}: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to clip {tile_file}: {e}. Keeping original.")
                clipped_files.append(tile_file)
        
        # If multiple clipped tiles, merge using gdal_merge
        if len(clipped_files) > 1:
            print(f"Merging {len(clipped_files)} clipped tiles using gdal_merge...")
            try:
                # Create merged filename
                base_name = os.path.basename(clipped_files[0]).replace('_clipped.tif', '_merged.tif')
                merged_path = os.path.join(output_dir, base_name)
                
                # Run gdal_merge
                cmd = ['gdal_merge.py', '-o', merged_path] + clipped_files
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Successfully merged to: {base_name}")
                    
                    # Delete clipped source files
                    for clipped_file in clipped_files:
                        try:
                            os.remove(clipped_file)
                            print(f"Deleted clipped source: {os.path.basename(clipped_file)}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {clipped_file}: {e}")
                    
                    return [merged_path]
                else:
                    logger.warning(f"gdal_merge failed: {result.stderr}. Using individual clipped tiles.")
                    return clipped_files
            except Exception as e:
                logger.warning(f"Merging failed: {e}. Using individual clipped tiles.")
                return clipped_files
        
        return clipped_files

