import json
import random
import numpy as np
import tensorflow as tf

class Config:
    def __init__(self, config_file='config.json'):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Assign values to class variables from JSON
        self.BATCH_SIZE = config_data["BATCH_SIZE"]
        self.EPOCHS = config_data["EPOCHS"]
        self.LR = config_data["LR"]
        self.N_CLASSES = config_data["N_CLASSES"]
        self.IN_SHAPE = config_data["IN_SHAPE"]
        self.DATASET = config_data["DATASET"]
        self.PATCH_SIZE = self.IN_SHAPE[self.DATASET[0]][0]
        self.DATA_PATH = config_data["DATA_PATH"]
        self.CHECKPOINT_PATH = config_data["CHECKPOINT_PATH"]
        self.LOG_PATH = config_data["LOG_PATH"]
        self.PREDICTION_PATH = config_data["PREDICTION_PATH"]
        self.AOI = config_data["AOI"]
    
    # Set seed for reproducibility
    def set_seed(self):
        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)

    #function to use gpu
    def use_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found, using CPU.")

# Initialize config
config = Config('config.json')
