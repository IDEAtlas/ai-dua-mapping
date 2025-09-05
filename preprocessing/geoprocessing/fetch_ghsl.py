import os
import requests
import zipfile
import geopandas as gpd
import shutil

# Source for the links: https://human-settlement.emergency.copernicus.eu/download.php

class GHSLDownloader:
    def __init__(self, temp_dir: str = "/tmp/ghsl", delete_on_destruct: bool = True):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.shp_url = "https://ghsl.jrc.ec.europa.eu/download/GHSL_data_54009_shapefile.zip"
        self.delete_on_destruct = delete_on_destruct

        print("---")
        print(f"Downloading GHSL shapefile from {self.shp_url} to {self.temp_dir}")
        self.zip_path = os.path.join(self.temp_dir, "GHSL_data_54009_shapefile.zip")
        response = requests.get(self.shp_url, stream=True)
        response.raise_for_status()
        with open(self.zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded to {self.zip_path}")

        self.shapefile_dir = os.path.join(os.path.dirname(self.zip_path), "GHSL_shapefile")
        os.makedirs(self.shapefile_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.shapefile_dir)
        print("---")
        print(f"Unzipped GHSL shapefile to {self.shapefile_dir}")

    def __del__(self):
        try:
            if os.path.exists(self.temp_dir) and self.delete_on_destruct:
                shutil.rmtree(self.temp_dir)
                print(f"Temporary directory {self.temp_dir} deleted.")
        except Exception as e:
            print(f"Error deleting temporary directory {self.temp_dir}: {e}")

    def get_intersecting_tile_codes(self, aoi_geojson: str) -> list:
        if not self.shapefile_dir:
            raise RuntimeError("Shapefile directory not set. Unzip shapefile first.")
        print("---")
        print(f"Finding intersecting tiles between AOI {aoi_geojson} and GHSL shapefile in {self.shapefile_dir}")
        aoi = gpd.read_file(aoi_geojson)
        print(f"AOI loaded with {len(aoi)} features, CRS: {aoi.crs}")

        shp_files = [f for f in os.listdir(self.shapefile_dir) if f.lower().endswith('.shp')]
        if not shp_files:
            raise FileNotFoundError("No .shp files found in the unzipped directory.")
        elif len(shp_files) > 1:
            raise RuntimeError("Multiple .shp files found; expected only one.")
        shapefile_path = os.path.join(self.shapefile_dir, shp_files[0])

        shapefile = gpd.read_file(shapefile_path)
        print(f"GHSL shapefile loaded with {len(shapefile)} features, CRS: {shapefile.crs}")

        if aoi.crs != shapefile.crs:
            print(f"Reprojecting AOI from {aoi.crs} to {shapefile.crs}")
            aoi = aoi.to_crs(shapefile.crs)

        intersecting_tiles = gpd.overlay(aoi, shapefile, how='intersection')
        if intersecting_tiles.empty:
            raise ValueError("No intersecting tiles found between AOI and GHSL shapefile.")
        else:
            print(f"Found {len(intersecting_tiles)} intersecting tiles.")
            tile_codes = intersecting_tiles['tile_id'].unique().tolist()
            print(f"Intersecting tile codes: {tile_codes}")
            return tile_codes

    def download_tiles(self, aoi_geojson: str, output_dir: str = None) -> list:
        # Get intersecting tile codes
        tile_codes = self.get_intersecting_tile_codes(aoi_geojson)

        # Generate a prefix with the first 3 letters of the AOI file name in capital letters
        aoi_prefix = os.path.basename(aoi_geojson)[:3].upper()

        base_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10/V1-0/tiles"
        if output_dir is None:
            output_dir = self.temp_dir
        os.makedirs(output_dir, exist_ok=True)
        downloaded_files = []

        for tile_code in tile_codes:
            file_name = f"GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_{tile_code}.zip"
            url = f"{base_url}/{file_name}"
            local_zip_path = os.path.join(self.temp_dir, file_name)
            print(f"Downloading tile {tile_code} from {url}")
            response = requests.get(url, stream=True)
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

                # Move to output_dir if different from temp_dir
                if output_dir != self.temp_dir:
                    shutil.move(tif_path, os.path.join(output_dir, new_tif_name))
                    tif_path = os.path.join(output_dir, new_tif_name)

                if os.path.exists(tif_path):
                    downloaded_files.append(tif_path)
            else:
                print(f"Failed to download tile {tile_code}: HTTP {response.status_code}")

        return downloaded_files

# Example usage
# downloader = GHSLDownloader(temp_dir="/data/raw/ghsl/temp")
# downloaded_files = downloader.download_tiles(aoi_geojson="/data/raw/aoi/guatemala_city_aoi.geojson",
#                                             output_dir="/data/raw/ghsl")