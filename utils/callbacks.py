import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tqdm.keras import TqdmCallback

def build_callbacks(city, inputs, model, train_dataset, config=None, monitor="val_seg_loss", patience=15):
    """
    Build and return a list of Keras callbacks.

    Args:
        city (str): city name used in filenames.
        inputs (str): input descriptor used in filenames.
        model: Keras model (used for model.name).
        train_dataset: dataset or iterable (len() should be supported).
        config: optional config object; if None, use module-level `config`.
        monitor (str): metric to monitor for checkpointing/early stopping.
        patience (int): patience for EarlyStopping.

    Returns:
        list: list of Keras callback instances.
    """
    config = config or config

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(f'./{config.CHECKPOINT_PATH}/{city}.{inputs.lower()}.{model.name}.weights.h5'),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=True
        ),
        CSVLogger(
            os.path.join("log", f"{city}.{inputs.lower()}.{model.name}.log.csv")
        ),
        TqdmCallback(
            epochs=config.EPOCHS,
            # data_size=len(train_dataset),
            batch_size=config.BATCH_SIZE,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor,
            mode='min',
            patience=patience,
            restore_best_weights=True
        )
    ]

    return callbacks