#!/usr/bin/env python3
"""
Classification wrapper - prepares data (if needed) and generates segmentation maps using trained/fine-tuned model.

Usage:
    If weights are not provided, it uses the global model weights located at ./checkpoint/global.s2.bd.mbcnn.weights.h5 as initial weights.
    python classify.py --city Salvador --country Brazil --year 2025

    To specify custom weights file, use the --weights flag
    python classify.py --city Salvador --country Brazil --year 2025 --weights checkpoint/custom.weights.h5

"""

import subprocess
import sys
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Classify/generate segmentation maps for a city.")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--country", type=str, required=True, help="Country name")
parser.add_argument("--year", type=int, required=True, help="Year for data collection")
parser.add_argument("--weights", type=str, default=None, help="Path to model weights (optional, defaults to global model)")
args = parser.parse_args()

# normalize city name
city_normalized = f"{args.city}_{args.country}".lower().replace(" ", "_").replace("-", "_")

logger.info("CLASSIFICATION WORKFLOW START")

# Step 1: Prepare data
logger.info("[1/2] Preparing Data...")
result = subprocess.run([
    sys.executable, "prepare_data.py",
    "--city", args.city,
    "--country", args.country,
    "--year", str(args.year),
    "--caller", "classify"
], check=False)

if result.returncode != 0:
    logger.error("[1/2] Preparing Data - FAILED!")
    sys.exit(1)

# Step 2: Run classification
logger.info("[2/2] Running Classification...")
s2_img = f"./data/raw/sentinel/{city_normalized}/S2_{args.year}.tif"
bd_path = f"./data/raw/buildings/density/{city_normalized}_bd.tif"

cmd = [sys.executable, "_orchestrate.py", "--stage", "infer", "--city", city_normalized, "--s2", s2_img, "--bd", bd_path]
if args.weights:
    cmd.extend(["--weight", args.weights])

result = subprocess.run(cmd, check=False)

if result.returncode != 0:
    logger.error("[2/2] Running Classification - FAILED!")
    sys.exit(1)

logger.info("CLASSIFICATION WORKFLOW COMPLETED SUCCESSFULLY")

sys.exit(result.returncode)
