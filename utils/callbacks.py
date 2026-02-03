import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tqdm.keras import TqdmCallback

def build_callbacks(city, inputs, model, train_dataset, config=None, is_finetune=False):
    """
    Build and return a list of Keras callbacks.

    Args:
        city (str): city name used in filenames.
        inputs (str): input descriptor used in filenames.
        model: Keras model (used for model.name).
        train_dataset: tf.data.Dataset or Sequence with len() support.
        config: optional config object; if None, use module-level `config`.
        is_finetune (bool): If True, monitor training loss instead of validation loss.

    Returns:
        list: list of Keras callback instances.
    """
    if config is None:
        from utils.configs import config as default_config
        config = default_config
    
    # Try to get dataset length - tf.data.Dataset doesn't have len(), so estimate if needed
    try:
        dataset_len = len(train_dataset)
    except TypeError:
        dataset_len = None

    # For finetune (no validation), monitor training loss instead
    monitor = config.MONITOR
    if is_finetune and monitor.startswith('val_'):
        monitor = monitor.replace('val_', '')

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(f'./{config.CHECKPOINT_PATH}/{city}.{inputs.lower()}.{model.name}.weights.h5'),
            monitor=monitor,
            mode=config.MODE if hasattr(config, 'MODE') and config.MODE else 'auto',
            save_best_only=True,
            save_weights_only=True
        ),
        CSVLogger(
            os.path.join(config.LOG_PATH, f"{city}.{inputs.lower()}.{model.name}.log.csv")
        ),
        EarlyStopping(
            monitor=monitor,
            mode=config.MODE if hasattr(config, 'MODE') and config.MODE else 'auto',
            patience=config.PATIENCE,
            restore_best_weights=True
        )
    ]
    
    # TqdmCallback works only if we have dataset length
    if dataset_len is not None:
        callbacks.insert(2, TqdmCallback(
            epochs=config.N_EPOCHS,
            data_size=dataset_len,
            batch_size=config.BATCH_SIZE,
            verbose=1
        ))

    return callbacks