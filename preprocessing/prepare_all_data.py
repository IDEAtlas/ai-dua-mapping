import os
import glob
import subprocess
from geoprocessing import adm_boundaries
from snappy import stac_api_new
from geoprocessing import fetch_ghsl, create_ref, extract_patch
from morphometrics import create_density, google_buildings

city = "Dodoma, Tanzania"
output_dir = "/data/raw"
output_patches_root = "/data/ideabench/extra"
S2_year = 2025
cloud_cover = 20
ref_year = 2016
city = city.lower().replace(" ", "_").replace(",", "").replace("-", "_")

print(f"Processing data for {city}")

# Fetch the adm boundaries
if os.path.exists(os.path.join(output_dir, "aoi", f"{city}_aoi.geojson")):
    print("AOI GeoJSON already exists. Skipping download.")
    aoi_geojson = os.path.join(output_dir, "aoi", f"{city}_aoi.geojson")
else:
    aoi_geojson = adm_boundaries.fetch_and_save_adm_borders_geojson(city, os.path.join(output_dir, "aoi"))

# Download and pre-process the S2 images
if os.path.exists(os.path.join(output_dir, "sentinel", city, f"S2_{S2_year}.tif")):
    print(f"S2 data for {city} already exists. Skipping download.")
    s2_img = os.path.join(output_dir, "sentinel", city, f"S2_{S2_year}.tif")
else:
    s2_img = stac_api_new.process_sentinel2_stack(city, year=S2_year, basedir=output_dir, max_cloud_cover=cloud_cover)

# Download GHSL-BUILD data
ghs_build_file = glob.glob(os.path.join(output_dir, "ghsl", "built", f"{city[:3].upper()}_*.tif"))
if ghs_build_file:
    print(f"GHSL data for {city} already exists. Skipping download.")
    downloaded_files = ghs_build_file
else:
    downloader = fetch_ghsl.GHSLDownloader(temp_dir=os.path.join(output_dir, "ghsl", "temp"))
    downloaded_files = downloader.download_tiles(aoi_geojson=aoi_geojson,
                                                output_dir=os.path.join(output_dir, "ghsl/built/"))

# Create reference dataset
if os.path.exists(os.path.join(output_dir, f"reference_data/v1/{city}_reference_{ref_year}_v1.tif")):
    print(f"Reference data for {city} already exists. Skipping creation.")
    output_reference = os.path.join(output_dir, f"reference_data/v1/{city}_reference_{ref_year}_v1.tif")
else:
    output_reference = os.path.join(output_dir, f"reference_data/v1/{city}_reference_{ref_year}_v1.tif")
    path_duas = os.path.join(output_dir, f"reference_data/v1/{city}_reference_{ref_year}_v1.geojson")
    create_ref.create_reference(s2_img, downloaded_files[0], path_duas, output_reference)

# Download building footprints
if os.path.exists(os.path.join(output_dir, f'buildings/{city}_bldg.gpkg')):
    print(f"Building footprints for {city} already exists. Skipping download.")
    aoi_df = os.path.join(output_dir, f'buildings/{city}_bldg.gpkg')
else:
    aoi_df = google_buildings.download_google_open_buildings(
        aoi_path=aoi_geojson,
        output_path=os.path.join(output_dir, f'buildings/{city}_bldg'),
        format_type='gpkg')

# Create building density raster
if os.path.exists(os.path.join(output_dir, f'buildings/density/{city}_bd.tif')):
    print(f"Building density raster for {city} already exists. Skipping creation.")
    bd_path = os.path.join(output_dir, f'buildings/density/{city}_bd.tif')
else:
    print("\nCreating building density raster...")
    bd_path = create_density.CreateDensity(s2_img, 
                                           os.path.join(output_dir, f'buildings/{city}_bldg.gpkg'), 
                                           os.path.join(output_dir, f'buildings/density/{city}_bd.tif'))

# Create sampling grid
if os.path.exists(os.path.join(output_dir, "sampling_grid", f"{city}_grid.geojson")):
    print(f"Sampling grid for {city} already exists. Skipping creation.")
    grid_path = os.path.join(output_dir, "sampling_grid", f"{city}_grid.geojson")
else:
    grid_path = os.path.join(output_dir, "sampling_grid", f"{city}_grid.geojson")
    cmd = [
        "python",
        "geoprocessing/grid_sampling.py",
        "--input", output_reference,
        "--output", grid_path,
        "--aoi", aoi_geojson,
    ]

    grid = subprocess.run(cmd, capture_output=True, text=True)
    
    print(grid.stdout)
    print(grid.stderr)

# Extract patches using the sampling grid
if os.path.exists(os.path.join(output_patches_root, "sentinel", city)):
    print(f"Patches for {city} already exist. Skipping extraction.")
else:
    dict_data = {"S2": s2_img, 
                "BD": bd_path,
                "RF": output_reference}


    for i, (data_name, data_path) in enumerate(dict_data.items()):
        print(f"\nExtracting patches for {data_name} ({i+1}/{len(dict_data)})")

        extract_patch.extract_patches(raster_path=data_path,
                                    grid_path=grid_path,
                                    output_root=os.path.join(output_patches_root, "sentinel", city),
                                    prefix=data_name)