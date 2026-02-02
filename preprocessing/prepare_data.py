import os
import sys
import glob
import subprocess
import argparse
import logging
import warnings
from tqdm import tqdm
from . import adm_boundaries, stac_api, fetch_ghsl, create_ref, extract_patch, create_density, google_buildings

# Setup logging with consistent format - force=True ensures it applies even if logging is configured elsewhere
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

# Suppress verbose loggers from imported modules
logging.getLogger('rasterio').setLevel(logging.WARNING)
logging.getLogger('fiona').setLevel(logging.WARNING)
logging.getLogger('shapely').setLevel(logging.WARNING)
logging.getLogger('osgeo').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('google_buildings').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare data for city and year by downloading and processing satellite imagery, building footprints, and creating reference datasets."
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="City name (e.g., 'Salvador', 'Dodoma')"
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country name (e.g., 'Brazil', 'Tanzania')"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year for data collection (e.g., 2025)"
    )
    
    parser.add_argument(
        "--cloud-cover",
        type=int,
        default=20,
        help="Maximum cloud cover percentage for Sentinel-2 data (default: 20)"
    )
    parser.add_argument(
        "--caller",
        type=str,
        choices=["train", "finetune", "classify"],
        default="train",
        help="Task type: 'train' for full dataset (train/val/test), 'finetune' for limited data (train only), or 'classify' for inference only"
    )
    return parser.parse_args()


# Parse CLI arguments
args = parse_arguments()
city = args.city.lower()
country = args.country.lower()
year = args.year
cloud_cover = args.cloud_cover
caller = args.caller  # 'train', 'finetune', 'classify'
output_dir = 'data/raw'
output_patches_root = 'data/processed'

# Log task type silently (no need to print decorative headers)

# Suppress subprocess output function
def suppress_output(func, *args, **kwargs):
    """Execute function with suppressed stdout/stderr and logging"""
    import sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_level = logging.getLogger().level
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        # Suppress all logging while executing the function
        logging.getLogger().setLevel(logging.CRITICAL)
        return func(*args, **kwargs)
    except Exception as e:
        # Always print errors 
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.getLogger().setLevel(old_level)
        logger.error(f"Error in {func.__name__}: {str(e)}")
        raise
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.getLogger().setLevel(old_level)

# Construct city string for normalization
city_str = f"{city}, {country}"
city_normalized = city_str.lower().replace(" ", "_").replace(",", "").replace("-", "_")

# Ensure all required directories exist
os.makedirs(os.path.join(output_dir, "aoi"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "sentinel", city_normalized), exist_ok=True)
os.makedirs(os.path.join(output_dir, "reference_data"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "buildings", "density"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "sampling_grid"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "ghsl", "built"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "ghsl", "pop"), exist_ok=True)
os.makedirs(output_patches_root, exist_ok=True)

# List of preparation steps (total 8, but classify only needs 6)
total_steps = 6 if caller == "classify" else 8
progress = tqdm(total=total_steps, desc="Data preparation", unit="step", position=0, leave=True)

# Step 1: Fetch the adm boundaries
progress.set_description("Getting AOI")
if os.path.exists(os.path.join(output_dir, "aoi", f"{city_normalized}_aoi.geojson")):
    aoi_geojson = os.path.join(output_dir, "aoi", f"{city_normalized}_aoi.geojson")
else:
    try:
        aoi_geojson = adm_boundaries.fetch_and_save_adm_borders_geojson(city_str, os.path.join(output_dir, "aoi"), output_file=f"{city_normalized}_aoi.geojson")
    except Exception as e:
        logger.error(f"Unable to fetch city boundary. Provide your own AOI in data/raw/aoi/{city_normalized}_aoi.geojson with wgs84 projection")
        progress.close()
        sys.exit(1)
progress.update(1)

# Step 2: Download and pre-process the S2 images
progress.set_description("Downloading Sentinel-2 image")
if os.path.exists(os.path.join(output_dir, "sentinel", city_normalized, f"S2_{year}.tif")):
    s2_img = os.path.join(output_dir, "sentinel", city_normalized, f"S2_{year}.tif")
else:
    try:
        s2_img = suppress_output(stac_api.process_sentinel2_stack, 
                                 city_normalized, year=year, basedir=output_dir, max_cloud_cover=cloud_cover)
    except Exception as e:
        logger.error("Sentinel-2 data not found")
        progress.close()
        sys.exit(1)
progress.update(1)

# Step 3: Fetch GHSL data
progress.set_description("Fetching GHSL")
ghs_build_files = glob.glob(os.path.join(output_dir, "ghsl", "built", f"{city_normalized[:3].upper()}_*.tif"))
if ghs_build_files:
    downloaded_files = ghs_build_files
else:
    try:
        downloader = fetch_ghsl.GHSLDownloader(temp_dir=os.path.join(output_dir, "ghsl", "temp"))
        downloaded_files = suppress_output(downloader.download_tiles, aoi_geojson=aoi_geojson,
                                                    output_dir=os.path.join(output_dir, "ghsl"), data_type="built")
    except Exception as e:
        logger.error("GHSL data download failed")
        progress.close()
        sys.exit(1)

# Use the GHSL built file - clipping and merging happens automatically
ghsl_built_file = downloaded_files[0] if downloaded_files else None

progress.update(1)

# Step 4: Create reference dataset (skip for classify - not needed for inference)
if caller == "classify":
    output_reference = None
    progress.set_description("Creating labels (skipped)")
else:
    progress.set_description("Creating labels")
    # Check if reference data already exists (any version)
    # Format: {city_normalized}_reference_{year}_v*.tif
    existing_ref_tif = glob.glob(os.path.join(output_dir, f"reference_data/{city_normalized}_reference_{year}_v*.tif"))
    
    if existing_ref_tif:
        output_reference = existing_ref_tif[0]
    else:
        # Look for reference GeoJSON input (any version)
        # Format: {city_normalized}_reference_{year}_v*.geojson
        ref_geojson_files = glob.glob(os.path.join(output_dir, "reference_data/", f"{city_normalized}_reference_{year}_v*.geojson"))
        
        if ref_geojson_files:
            path_duas = ref_geojson_files[0]
        else:
            # Try without version suffix
            ref_geojson_files = glob.glob(os.path.join(output_dir, "reference_data/", f"{city_normalized}_reference_{year}.geojson"))
            if ref_geojson_files:
                path_duas = ref_geojson_files[0]
            else:
                logger.error(f"No reference data file found. Expected: {city_normalized}_reference_{year}_v*.geojson")
                progress.close()
                exit(1)
        
        # Extract version from geojson filename and use for tif output
        version_str = path_duas.split("_v")[-1].split(".")[0]  # Extract version number
        output_reference = os.path.join(output_dir, f"reference_data/{city_normalized}_reference_{year}_v{version_str}.tif")
        
        # Call create_reference with output suppressed
        try:
            suppress_output(create_ref.create_reference, s2_img, ghsl_built_file, path_duas, output_reference)
        except Exception as e:
            logger.error("Reference data creation failed")
            progress.close()
            sys.exit(1)
progress.update(1)

# Step 5: Download building footprints
progress.set_description("Downloading building footprints")
if os.path.exists(os.path.join(output_dir, f'buildings/{city_normalized}_bldg.gpkg')):
    aoi_df = os.path.join(output_dir, f'buildings/{city_normalized}_bldg.gpkg')
else:
    try:
        aoi_df = suppress_output(google_buildings.download_google_open_buildings,
            aoi_path=aoi_geojson,
            output_path=os.path.join(output_dir, f'buildings/{city_normalized}_bldg'),
            format_type='gpkg')
    except Exception as e:
        logger.error("Building footprints download failed")
        progress.close()
        sys.exit(1)
progress.update(1)

# Step 6: Create building density raster
progress.set_description("Computing built-up density")
if os.path.exists(os.path.join(output_dir, f'buildings/density/{city_normalized}_bd.tif')):
    bd_path = os.path.join(output_dir, f'buildings/density/{city_normalized}_bd.tif')
else:
    try:
        bd_path = suppress_output(create_density.CreateDensity, s2_img, 
                                               os.path.join(output_dir, f'buildings/{city_normalized}_bldg.gpkg'), 
                                               os.path.join(output_dir, f'buildings/density/{city_normalized}_bd.tif'))
    except Exception as e:
        logger.error("built-up density computation failed")
        progress.close()
        sys.exit(1)
progress.update(1)

# Step 7: Create sampling grid (skip for classify - not needed for inference)
if caller != "classify":
    progress.set_description("Generating sampling grid")
    if os.path.exists(os.path.join(output_dir, "sampling_grid", f"{city_normalized}_grid.geojson")):
        grid_path = os.path.join(output_dir, "sampling_grid", f"{city_normalized}_grid.geojson")
    else:
        grid_path = os.path.join(output_dir, "sampling_grid", f"{city_normalized}_grid.geojson")
        
        # Choose grid sampling script based on task
        if caller == "train":
            grid_script = "preprocessing/grid_sampling.py"
        else:  # finetune
            grid_script = "preprocessing/grid_sampling_ft.py"
        
        cmd = [
            "python",
            grid_script,
            "--input", output_reference,
            "--output", grid_path,
            "--aoi", aoi_geojson,
        ]

        # Suppress subprocess output
        grid = subprocess.run(cmd, capture_output=True, text=True)
    progress.update(1)

# Step 8: Extract patches using the sampling grid (skip for classify - not needed for inference)
if caller != "classify":
    progress.set_description("Extracting patches")
    if os.path.exists(os.path.join(output_patches_root, city_normalized)):
        pass
    else:
        dict_data = {"S2": s2_img, 
                    "BD": bd_path,
                    "RF": output_reference}

        for i, (data_name, data_path) in enumerate(dict_data.items()):
            if caller == "train":
                suppress_output(extract_patch.extract_patches, raster_path=data_path,
                                            grid_path=grid_path,
                                            output_root=os.path.join(output_patches_root, city_normalized),
                                            prefix=data_name)
            else:  # finetune
                from preprocessing import extract_patch_ft
                suppress_output(extract_patch_ft.extract_patches_ft, raster_path=data_path,
                                                  grid_path=grid_path,
                                                  output_root=os.path.join(output_patches_root, city_normalized),
                                                  prefix=data_name)
    progress.update(1)

progress.close()
logger.info("Data preparation complete")