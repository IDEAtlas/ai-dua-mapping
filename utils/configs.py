import os
import yaml
import random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf


class Config:
    # Declare attributes so the IDE knows them
    BATCH_SIZE: int = None
    N_EPOCHS: int = None
    LR: float = None
    N_CLASSES: int = None
    IN_SHAPE: dict = None
    DATASET: list = None
    DATA_PATH: str = None
    CHECKPOINT_PATH: str = None
    LOG_PATH: str = None
    PREDICTION_PATH: str = None
    AOI_PATH: str = None
    STRIDE: float = None
    MONITOR: str = None
    MODE: str = None
    PATIENCE: int = None
    LOSS: str = None
    OPTIMIZER: str = None
    METRICS: list = None
    SEED: int = 42

    def __init__(self, config_file="config.yaml"):
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)

        # Update attributes
        for k, v in cfg.items():
            setattr(self, k, v)

        # Derived example
        self.PATCH_SIZE = self.IN_SHAPE[self.DATASET[0]][0]

        self._configure_tf()
        self._set_seed()

    def _configure_tf(self):
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.get_logger().setLevel("ERROR")

    def _set_seed(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        tf.random.set_seed(self.SEED)


config = Config()
