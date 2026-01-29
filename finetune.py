#!/usr/bin/env python3
"""
Fine-tuning wrapper - prepares limited data, fine-tunes pre-trained model on city-specific data.
Freezes encoder layers and trains only decoder + output layers on slum-focused patches.

Usage:
    python finetune.py --city Salvador --country Brazil --year 2025
    if weights are not provided, it uses the global model weights located at ./checkpoint/global.s2.bd.mbcnn.weights.h5 as initial weights.

    To specify custom initial weights file, use the --weights flag    
    python finetune.py --city Salvador --country Brazil --year 2025 --weights checkpoint/custom.s2.bd.mbcnn.weights.h5
"""

import subprocess
import sys
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Fine-tune a pre-trained model on city-specific data.")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--country", type=str, required=True, help="Country name")
parser.add_argument("--year", type=int, required=True, help="Year for data collection")
parser.add_argument("--weights", type=str, default=None, help="Initial weights (optional, defaults to global model)")
args = parser.parse_args()

# normalize city name
city = f"{args.city}_{args.country}".lower().replace(" ", "_").replace("-", "_")

logger.info("FINE-TUNING WORKFLOW START")

# Step 1: Prepare data
logger.info("[1/2] Preparing Data...")
result = subprocess.run([
    sys.executable, "prepare_data.py",
    "--city", args.city,
    "--country", args.country,
    "--year", str(args.year),
    "--caller", "finetune"
], check=False)

if result.returncode != 0:
    logger.error("[1/2] Preparing Data - FAILED!")
    sys.exit(1)

# Step 2: Fine-tune model
logger.info("[2/2] Fine-tuning Model...")

cmd = [sys.executable, "_orchestrate.py", "--stage", "finetune", "--city", city]
if args.weights:
    cmd.extend(["--weight", args.weights])

result = subprocess.run(cmd, check=False)
if result.returncode != 0:
    logger.error("[2/2] Fine-tuning Model - FAILED!")
    sys.exit(1)

logger.info("FINE-TUNING WORKFLOW COMPLETED SUCCESSFULLY")
