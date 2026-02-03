import os
import yaml
import random
import numpy as np
from types import SimpleNamespace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['TF_DISABLE_GRAPPLER_OPTIMIZER'] = '1'

import tensorflow as tf


def load_configs(path: str):
    """Load configuration from YAML file with TensorFlow setup."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = SimpleNamespace(**cfg)

    # Derived configuration
    cfg.PATCH_SIZE = cfg.IN_SHAPE[cfg.DATASET[0]][0]

    # Seed
    seed = getattr(cfg, "SEED", 42)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Configure TensorFlow
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel("ERROR")

    return cfg
