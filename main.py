#!/usr/bin/env python3
"""
Unified entry point for training, fine-tuning, and classification workflows.

Usage:
    # Train from scratch
    python main.py --task train --city Salvador --country Brazil --year 2025
    
    # Fine-tune with pre-trained weights
    python main.py --task finetune --city Salvador --country Brazil --year 2025 (default global model weights)
    python main.py --task finetune --city Salvador --country Brazil --year 2025 --weights checkpoint/custom.h5
    
    # Generate segmentation maps (classification/inference)
    python main.py --task classify --city Salvador --country Brazil --year 2025 (applies the specified city's weights)
    python main.py --task classify --city Salvador --country Brazil --year 2025 --weights checkpoint/custom.h5
"""

import subprocess
import sys
import argparse
import logging
import os
from datetime import datetime

# Import configs early to set environment variables
from utils.configs import load_configs
load_configs('config.yaml')

# Import orchestrator
from utils.pipelines import Pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments for all tasks."""
    parser = argparse.ArgumentParser(
        description="Unified workflow for training, fine-tuning, and classification of building detection models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "finetune", "classify", "sdg_stats"],
        help="Task type: 'train' (full dataset), 'finetune' (limited data with pre-trained weights), 'classify' (inference only), 'sdg_stats' (compute SDG statistics)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mbcnn",
        choices=["mbcnn", "unet", "lightunet", "fcndk6", "fpn", "deeplab", "glavitu"],
        help="Model architecture (default: mbcnn)"
    )
    parser.add_argument("--city", type=str, required=True, help="City name (e.g., Salvador, Cape-Town)")
    parser.add_argument("--country", type=str, required=True, help="Country name (e.g., Brazil, South-Africa)")
    parser.add_argument("--year", type=int, required=True, help="Year for data collection (e.g., 2025)")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to initial weights file (optional, for finetune/classify tasks)"
    )
    parser.add_argument(
        "--sdg_stats",
        action="store_true",
        help="Compute SDG 11.1.1 statistics after classification (optional)"
    )
    
    return parser.parse_args()


def normalize_city_name(city: str, country: str) -> str:
    """Normalize city and country names for file paths."""
    return f"{city}_{country}".lower().replace(" ", "_").replace("-", "_")


def run_workflow_train(city: str, country: str, year: int, city_normalized: str, model: str = "mbcnn") -> int:
    """Run training workflow: prepare → train → test"""
    logger.info("TRAINING WORKFLOW START")
    logger.info(f"City: {city.capitalize()}, Country: {country.capitalize()}, Year: {year}, Model: {model.upper()}")
    
    # Step 1: Prepare data
    processed_dir = os.path.join("./data/processed/", city_normalized)
    if os.path.exists(processed_dir) and os.path.exists(os.path.join(processed_dir, "train")):
        logger.info("[1/3] Data preparation skipped. Processed data already exists. Delete the processed folder to start from scratch.")
    else:
        logger.info("[1/3] Preparing Data...")
        result = subprocess.run([
            sys.executable, "-m", "preprocessing.prepare_data",
            "--city", city,
            "--country", country,
            "--year", str(year),
            "--caller", "train"
        ], check=False)
        
        if result.returncode != 0:
            logger.error("[1/3] Preparing Data - FAILED!")
            return 1
    
    # Step 2: Train
    logger.info("[2/3] Training Model...")
    try:
        pipeline = Pipeline(stage="train", city=city_normalized, model=model)
        pipeline.run()
    except Exception as e:
        logger.error(f"[2/3] Training Model - FAILED! {e}")
        return 1
    
    # Step 3: Test
    logger.info("[3/3] Evaluating Model...")
    try:
        pipeline = Pipeline(stage="test", city=city_normalized, model=model)
        pipeline.run()
    except Exception as e:
        logger.error(f"[3/3] Evaluating Model - FAILED! {e}")
        return 1
    
    logger.info("TRAINING WORKFLOW COMPLETED")
    return 0


def run_workflow_finetune(city: str, country: str, year: int, city_normalized: str, weights: str = None, model: str = "mbcnn") -> int:
    """Run fine-tuning workflow: prepare → finetune"""
    logger.info("FINE-TUNING WORKFLOW START")
    logger.info(f"City: {city.capitalize()}, Country: {country.capitalize()}, Year: {year}, Model: {model.upper()}")
    if weights:
        logger.info(f"Using initial weights: {weights}")
    
    # Step 1: Prepare data
    logger.info("[1/2] Preparing Data...")
    result = subprocess.run([
        sys.executable, "-m", "preprocessing.prepare_data",
        "--city", city,
        "--country", country,
        "--year", str(year),
        "--caller", "finetune"
    ], check=False)
    
    if result.returncode != 0:
        logger.error("[1/2] Preparing Data - FAILED!")
        return 1
    
    # Step 2: Fine-tune model
    logger.info("[2/2] Fine-tuning Model...")
    try:
        pipeline = Pipeline(stage="finetune", city=city_normalized, weight=weights, model=model)
        pipeline.run()
    except Exception as e:
        logger.error(f"[2/2] Fine-tuning Model - FAILED! {e}")
        return 1
    
    logger.info("FINE-TUNING WORKFLOW COMPLETED")
    return 0


def run_workflow_classify(city: str, country: str, year: int, city_normalized: str, weights: str = None, model: str = "mbcnn") -> int:
    """Run classification workflow: prepare → infer"""
    logger.info("CLASSIFICATION WORKFLOW START")
    logger.info(f"City: {city.capitalize()}, Country: {country.capitalize()}, Year: {year}, Model: {model.upper()}")
    if weights:
        logger.info(f"Using model weights: {weights}")
    
    # Step 1: Prepare data
    logger.info("[1/2] Preparing Data...")
    result = subprocess.run([
        sys.executable, "-m", "preprocessing.prepare_data",
        "--city", city,
        "--country", country,
        "--year", str(year),
        "--caller", "classify"
    ], check=False)
    
    if result.returncode != 0:
        logger.error("[1/2] Preparing Data - FAILED!")
        return 1
    
    # Step 2: Run classification
    logger.info("[2/2] Running Classification...")
    s2_img = f"./data/raw/sentinel/{city_normalized}/S2_{year}.tif"
    bd_path = f"./data/raw/buildings/density/{city_normalized}_bd.tif"
    
    try:
        pipeline = Pipeline(stage="infer", city=city_normalized, s2=s2_img, bd=bd_path, weight=weights, model=model)
        pipeline.run()
    except Exception as e:
        logger.error(f"[2/2] Running Classification - FAILED! {e}")
        return 1


def run_workflow_sdg_stats(city: str, country: str, year: int, city_normalized: str, model: str = "mbcnn") -> int:
    """Run SDG statistics computation workflow."""
    logger.info("SDG STATISTICS WORKFLOW START")
    logger.info(f"City: {city.capitalize()}, Country: {country.capitalize()}, Year: {year}")
    
    # Step 1: Compute SDG statistics
    logger.info("[1/1] Computing SDG Statistics...")
    try:
        pipeline = Pipeline(stage="sdg_stats", city=city_normalized, model=model, year=year)
        pipeline.run()
    except Exception as e:
        logger.error(f"[1/1] Computing SDG Statistics - FAILED! {e}")
        return 1
    
    logger.info("SDG STATISTICS WORKFLOW COMPLETED")
    return 0


def main():
    """Main entry point."""
    args = parse_arguments()
    
    city_normalized = normalize_city_name(args.city, args.country)
    
    # Route to appropriate workflow
    if args.task == "train":
        return_code = run_workflow_train(args.city, args.country, args.year, city_normalized, args.model)
    elif args.task == "finetune":
        return_code = run_workflow_finetune(args.city, args.country, args.year, city_normalized, args.weights, args.model)
    elif args.task == "classify":
        return_code = run_workflow_classify(args.city, args.country, args.year, city_normalized, args.weights, args.model)
        
        # Compute SDG stats if requested
        if return_code == 0 and args.sdg_stats:
            logger.info("[3/3] Computing SDG Statistics...")
            try:
                pipeline = Pipeline(stage="sdg_stats", city=city_normalized)
                pipeline.run()
                logger.info("SDG Statistics computation completed")
            except Exception as e:
                logger.error(f"[3/3] Computing SDG Statistics - FAILED! {e}")
                return_code = 1
    elif args.task == "sdg_stats":
        return_code = run_workflow_sdg_stats(args.city, args.country, args.year, city_normalized, args.model)
    else:
        logger.error(f"Unknown task: {args.task}")
        return_code = 1
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()
