# Training, fine-tuning and classification/inference orchestrator pipeline

import os
import sys
import argparse
import logging
# Import config first to set environment variables
from utils.configs import load_configs
import tensorflow as tf
from models.utils import select_model
from utils.callbacks import build_callbacks
from metrics.metrics import F1Score
from losses import losses
from utils import dataloader as dl
from utils.inference import inference_mbcnn
from metrics import metrics
import segmentation_models as sm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Unified pipeline for training, fine-tuning, testing, and inference."""
    
    def __init__(self, stage: str, city: str, config=None, s2: str = None, bd: str = None, weight: str = None, model: str = "mbcnn", year: int = None):
        """
        Initialize orchestrator.
        
        Args:
            stage: One of 'train', 'finetune', 'test', 'infer', 'sdg_stats'
            city: City name (normalized)
            config: Configuration object (loads from yaml if None)
            s2: Sentinel-2 image path (for inference)
            bd: Building density raster path (for inference)
            weight: Path to model weights (optional)
            model: Model architecture name (default: 'mbcnn')
            year: Year of data (for sdg_stats)
        """
        self.stage = stage
        self.city = city
        self.config = config or load_configs('config.yaml')
        self.s2 = s2
        self.bd = bd
        self.weight = weight
        self.model = model
        self.year = year
        
        # Setup paths
        self.data_dir = os.path.join(self.config.DATA_PATH, city)
        self.inputs = ".".join(self.config.DATASET)
        self.input_shapes = {inp: self.config.IN_SHAPE[inp] for inp in self.config.DATASET}
        
        # Create necessary directories
        os.makedirs(self.config.CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(self.config.LOG_PATH, exist_ok=True)
        os.makedirs(self.config.PREDICTION_PATH, exist_ok=True)
    
    def _get_loss_function(self, class_weights: list):
        """Get loss function with class weights."""
        return sm.losses.CategoricalFocalLoss()
        # Alternative: Dice loss
        # return sm.losses.DiceLoss(class_weights=class_weights)
    
    def _build_model(self):
        """Build model using select_model from models/utils.py."""
        return select_model(self.model, self.config)
    
    def _prepare_train_data(self):
        """Prepare training and validation dataloaders."""
        train_dataset = dl.MBCNNDataset(
            os.path.join(self.data_dir, "train"),
            target_size=self.config.PATCH_SIZE
        )
        train_loader = dl.mbcnn_loader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        val_dataset = dl.MBCNNDataset(
            os.path.join(self.data_dir, "val"),
            target_size=self.config.PATCH_SIZE
        )
        val_loader = dl.mbcnn_loader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        return train_loader, val_loader, train_dataset
    
    def _prepare_finetune_data(self):
        """Prepare training dataloader for fine-tuning (no validation split)."""
        train_dataset = dl.MBCNNDataset(
            os.path.join(self.data_dir, "train"),
            target_size=self.config.PATCH_SIZE
        )
        train_loader = dl.mbcnn_loader(
            train_dataset,
            batch_size=self.config.FINETUNE.get('BATCH_SIZE', self.config.BATCH_SIZE),
            shuffle=True
        )
        
        return train_loader, train_dataset
    
    def run_train(self):
        """Execute training stage."""
        
        train_loader, val_loader, train_dataset = self._prepare_train_data()
        
        # Get class weights
        _, class_weight_list = train_dataset.compute_class_weights()
        
        # Build and compile model
        model = self._build_model()
        loss_fn = self._get_loss_function(class_weight_list)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LR),
            loss=loss_fn,
            metrics=[F1Score(name='f1')]
        )
        
        # Build callbacks
        callbacks = build_callbacks(self.city, self.inputs, model, train_loader, self.config)
        
        # Train
        model.fit(
            train_loader,
            epochs=self.config.N_EPOCHS,
            callbacks=callbacks,
            validation_data=val_loader,
            verbose=0
        )
        
        logger.info("Training completed")
    
    def run_finetune(self):
        """Execute fine-tuning stage with configurable freeze strategy."""
        train_loader, train_dataset = self._prepare_finetune_data()
        
        # Get class weights
        _, class_weight_list = train_dataset.compute_class_weights()
        
        # Build model
        model = self._build_model()
        
        # Load pre-trained weights
        weight_path = self.weight or os.path.join(self.config.CHECKPOINT_PATH, 'global.s2.bd.mbcnn.weights.h5')
        if not os.path.exists(weight_path):
            logger.error(f"Weight file not found: {weight_path}")
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        model.load_weights(weight_path)
        logger.info(f"Loaded pre-trained weights from: {weight_path}")
        
        # Get fine-tuning configuration
        finetune_config = self.config.FINETUNE
        finetune_type = finetune_config.get('TYPE', 'full')
        
        # Apply layer freezing strategy
        if finetune_type == 'head':
            # Freeze encoder (layers 0-28), train decoder (layers 29-42)
            num_freeze = 29
            for i, layer in enumerate(model.layers):
                layer.trainable = i >= num_freeze
            logger.info(f"Head-only fine-tuning")
            learning_rate = finetune_config.get('LR_HEAD', 1.0e-4)
        else:  # 'full' or default
            # Train all layers with low learning rate
            for layer in model.layers:
                layer.trainable = True
            logger.info(f"Full model fine-tuning")
            learning_rate = finetune_config.get('LR_FULL', 5.0e-5)
        
        # Compile with appropriate learning rate
        loss_fn = self._get_loss_function(class_weight_list)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=[F1Score(name='f1')]
        )
        
        # Build callbacks
        callbacks = build_callbacks(self.city, self.inputs, model, train_loader, self.config, is_finetune=True)
        
        # Train with epochs from FINETUNE config
        epochs = finetune_config.get('N_EPOCHS', 10)
        model.fit(
            train_loader,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )
    
    def run_test(self):
        """Execute test stage."""
        logger.info("Evaluating model...")
        
        model = self._build_model()
        
        # Load weights
        weight_path = self.weight or os.path.join(
            self.config.CHECKPOINT_PATH,
            f"{self.city}.{self.inputs.lower()}.{model.name}.weights.h5"
        )
        
        if not os.path.exists(weight_path):
            logger.error(f"Weight file not found: {weight_path}")
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        model.load_weights(weight_path)
        logger.info(f"Loaded weights from: {weight_path}")
        
        # Prepare test data
        test_dataset = dl.MBCNNDataset(
            os.path.join(self.data_dir, 'test'),
            target_size=self.config.PATCH_SIZE
        )
        test_loader = dl.mbcnn_loader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        # Evaluate
        class_report, _, iou_values, mean_iou, fwiou = metrics.report(test_loader, model)
        
        logger.info("Classification Report:\n" + str(class_report))
        logger.info(f"IoU per Class:\n{iou_values}")
        logger.info(f"Mean IoU: {mean_iou:.4f}")
        logger.info(f"Frequency Weighted IoU: {fwiou:.4f}")
    
    def run_infer(self):
        """Execute inference stage."""
        
        # Validate inputs
        if not self.s2 or not self.bd:
            logger.error("Both --s2 and --bd arguments are required for inference")
            raise ValueError("Missing required inputs: s2 and/or bd")
        
        # Prepare image sources
        image_sources = [self.s2, self.bd]
        
        # Extract year from s2 path
        year = self.s2.split('/')[-1].split('_')[1].split('.')[0]
        
        # Build model
        model = self._build_model()
        
        # Load weights
        if not self.weight:
            self.weight = os.path.join(
                self.config.CHECKPOINT_PATH,
                f"{self.city}.{self.inputs.lower()}.{model.name}.weights.h5"
            )
        
        if not os.path.exists(self.weight):
            logger.error(f"Model weight file not found: {self.weight}")
            raise FileNotFoundError(f"Model weight file not found: {self.weight}")
        
        model.load_weights(self.weight)
        logger.info(f"Loaded weights from: {self.weight}")
        
        # Run inference
        output_path = os.path.join(
            self.config.PREDICTION_PATH,
            f"{self.city}.{self.inputs.lower()}.{model.name}.{year}.tif"
        )
        
        inference_mbcnn(
            model=model,
            image_sources=image_sources,
            save_path=output_path,
            aoi_path=os.path.join(self.config.AOI_PATH, f"{self.city}_aoi.geojson"),
            batch_size=self.config.BATCH_SIZE,
            stride_ratio=self.config.STRIDE_RATIO
        )
            
    def run_sdg_stats(self):
        """Compute SDG 11.1.1 statistics from classified raster."""
        import subprocess
        
        # Determine label scheme from config
        idx = getattr(self.config, 'LABEL_INDICES', '123')
        
        # Get year - from parameter or config
        year = self.year or getattr(self.config, 'YEAR', 2025)
        
        result = subprocess.run([
            sys.executable, "-m", "utils.sdg_stats",
            '--city', self.city.split('_')[0],
            '--country', self.city.split('_')[1],
            '--year', str(year),
            '--model', self.model,
            '--idx', idx
        ], check=False)
        
        if result.returncode != 0:
            logger.error("SDG stats computation failed")
            raise RuntimeError("SDG stats computation failed")
            
    def run(self):
        """Execute the requested stage."""
        try:
            if self.stage == "train":
                self.run_train()
            elif self.stage == "finetune":
                self.run_finetune()
            elif self.stage == "test":
                self.run_test()
            elif self.stage == "infer":
                self.run_infer()
            elif self.stage == "sdg_stats":
                self.run_sdg_stats()
            else:
                raise ValueError(f"Unknown stage: {self.stage}")
        except Exception as e:
            logger.error(f"Stage '{self.stage}' failed with error: {e}")
            raise
