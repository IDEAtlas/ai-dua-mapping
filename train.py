#!/usr/bin/env python3
"""
Training wrapper - prepares data, trains model from scratch, and automatically runs test.

Usage:
    python train.py --city Salvador --country Brazil --year 2025
"""

import subprocess
import sys
import argparse
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Prepare data, train a model, and automatically test it.")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--country", type=str, required=True, help="Country name")
parser.add_argument("--year", type=int, required=True, help="Year for data collection")
args = parser.parse_args()

# City normalization to match main.py format
city = f"{args.city}_{args.country}".lower().replace(" ", "_").replace("-", "_")

logger.info("TRAINING WORKFLOW START")
logger.info(f"City: {args.city}, Country: {args.country}, Year: {args.year}")

# Step 1: Prepare data (skip if already processed)
processed_dir = os.path.join("./data/processed/", city)
if os.path.exists(processed_dir) and os.path.exists(os.path.join(processed_dir, "train")):
    logger.info("[1/3] Data prepatation skipped. Processed data already exists. Delete the processed folder to start from scratch.")
else:
    logger.info("[1/3] Preparing Data...")
    result = subprocess.run([
        sys.executable, "prepare_data.py",
        "--city", args.city,
        "--country", args.country,
        "--year", str(args.year),
        "--caller", "train"
    ], check=False)

    if result.returncode != 0:
        logger.error("[1/3] Preparing Data - FAILED!")
        sys.exit(1)

# Step 2: Train
logger.info("[2/3] Training Model...")
result = subprocess.run([
    sys.executable, "_orchestrate.py",
    "--stage", "train",
    "--city", city
], check=False)

if result.returncode != 0:
    logger.error("[2/3] Training Model - FAILED!")
    sys.exit(1)

# Step 3: Test
logger.info("[3/3] Evaluating Model...")
result = subprocess.run([
    sys.executable, "_orchestrate.py",
    "--stage", "test",
    "--city", city
], check=False)

if result.returncode != 0:
    logger.error("[3/3] Evaluating Model - FAILED!")
    logger.error("Evaluation encountered an error. Check the error messages above.")
    sys.exit(1)

logger.info("TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
