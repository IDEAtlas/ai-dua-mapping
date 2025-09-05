import os
from geoprocessing import adm_boundaries
from snappy import stac_api_new
from geoprocessing import fetch_ghsl_gee, fetch_ghsl, create_ref, grid_sampling, extract_patch
from morphometrics import create_density
from morphometrics import google_buildings

city = "Guatemala City, Guatemala"
output_dir = "/data/raw"
path_duas = os.path.join(output_dir, "reference_data/v1/guatemala_city_reference_2024_v1.geojson")
output_patches_root = "/data/ideabench/extra"

# Fetch the adm boundaries
aoi_geojson = adm_boundaries.fetch_and_save_adm_borders_geojson(city, os.path.join(output_dir, "aoi"))

city = city.lower().replace(" ", "_").replace(",", "").replace("-", "_")
output_reference = os.path.join(output_dir, f"reference_data/v1/{city}_reference_2024_v1.tif")

# Download and pre-process the S2 image
s2_img = stac_api_new.process_sentinel2_stack(city, year=2024, basedir=output_dir)

# Download GHSL data
downloader = fetch_ghsl.GHSLDownloader(temp_dir=os.path.join(output_dir, "ghsl", "temp"))
downloaded_files = downloader.download_tiles(aoi_geojson=aoi_geojson, 
                                             output_dir=os.path.join(output_dir, "ghsl"))

# Create the reference raster
create_ref.create_reference(s2_img, downloaded_files[0], path_duas, output_reference)

# Download building footprints
aoi_df = google_buildings.download_google_open_buildings(
    aoi_path=aoi_geojson,
    output_path=os.path.join(output_dir, f'buildings/{city}_bldg'),
    format_type='gpkg')

# Create building density raster
bd_path = create_density.CreateDensity(s2_img, 
                                       os.path.join(output_dir, f'buildings/{city}_bldg.gpkg'), 
                                       os.path.join(output_dir, f'buildings/density/{city}_bd.tif'))

# Create Grid file for reference dataset
grid_path = grid_sampling.generate_grid_sampling(input_raster=s2_img,
                                     output_geojson=os.path.join(output_dir, "sampling_grid", f"{city}_grid.geojson"))

dict_data = {"S2": s2_img, 
             "BD": bd_path,
             "RF": output_reference}


for i, (data_name, data_path) in enumerate(dict_data.items()):
    print(f"\nExtracting patches for {data_name} ({i+1}/{len(dict_data)})")

    extract_patch.extract_patches(raster_path=data_path,
                                    grid_path=grid_path,
                                    output_root=os.path.join(output_patches_root, "sentinel", city),
                                    prefix=data_name)


