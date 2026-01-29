# Training, fine-tuning and classification/inference orchestrator pipeline 

import os
import argparse
import numpy as np
import logging
from utils.configs import load_configs
from models.mbcnn import mbcnn
from utils.callbacks import build_callbacks
from metrics.metrics import F1Score
import tensorflow as tf
from models.mbcnn import mbcnn
from losses import losses
from utils import dataloader as dl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train, evaluate or save a deep learning model.")
parser.add_argument("--stage", choices=["train", "finetune", "test", "infer"], required=True, help="Select mode: train, finetune, test or infer")
parser.add_argument("--city", type=str, required=True, help="Specify the city name")
parser.add_argument("--s2", type=str, required=False, help="Sentinel-2 image path (only for inference)")
parser.add_argument("--bd", type=str, required=False, help="Building density raster path (only for inference)")
parser.add_argument("--weight", type=str, required=False, help="Specify path to the model weight")
args = parser.parse_args()

config = load_configs('config.yaml')

city = args.city
dir = os.path.join(config.DATA_PATH, city)
inputs = ".".join(config.DATASET)
input_shapes = {input: config.IN_SHAPE[input] for input in config.DATASET}
h, w, _ = config.IN_SHAPE[config.DATASET[0]]

if args.stage == "train":
    
    # Create dataset using reference codebase approach
    train_dataset = dl.MBCNNDataset(
        os.path.join(dir, "train"),
        target_size=config.PATCH_SIZE
    )
    train_loader = dl.mbcnn_loader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    val_dataset = dl.MBCNNDataset(
        os.path.join(dir, "val"),
        target_size=config.PATCH_SIZE
    )
    val_loader = dl.mbcnn_loader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Calculate class weights from training dataset
    class_weight_dict, class_weight_list = train_dataset.compute_class_weights()

    import segmentation_models as sm
    dice_loss = sm.losses.DiceLoss(class_weights=class_weight_list) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    t_loss =  focal_loss
    
    # # Combined Dice + Focal loss with balanced class weighting
    # t_loss = losses.DiceFocalLoss(
    #     dice_weight=0.5,
    #     focal_weight=2.0,
    #     dice_smooth=1e-5,
    #     focal_gamma=2.0,
    #     focal_alpha=0.25,
    #     class_weights=class_weight_list
    # )

    model = mbcnn(config.N_CLASSES, input_shapes, dropout_rate=0.0, batch_norm=True, drop_train=False)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
        loss=t_loss,
        metrics=[F1Score(name='f1')]
    )

    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)

    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)

    callbacks = build_callbacks(city, inputs, model, train_loader, config)

    model.fit(train_loader,
            epochs=config.N_EPOCHS,
            callbacks=callbacks,
            validation_data=val_loader,
            verbose=0)
    
                    
elif args.stage == "finetune":

    # Create dataset using reference codebase approach
    train_dataset = dl.MBCNNDataset(
        os.path.join(dir, "train"),
        target_size=config.PATCH_SIZE
    )
    train_loader = dl.mbcnn_loader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Calculate class weights from training dataset
    class_weight_dict, class_weight_list = train_dataset.compute_class_weights()

    import segmentation_models as sm
    dice_loss = sm.losses.DiceLoss(class_weights=class_weight_list) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    t_loss =  focal_loss

    # t_loss = losses.DiceFocalLoss(
    #     dice_weight=0.5,
    #     focal_weight=2.0,  # Same as training for consistency
    #     dice_smooth=1e-5,
    #     focal_gamma=2.0,
    #     focal_alpha=0.5,
    #     class_weights=class_weight_list
    # )

    model = mbcnn(config.N_CLASSES, input_shapes, dropout_rate=0.0, batch_norm=True, drop_train=False)
    
    # Load pre-trained weights
    weight_path = args.weight if args.weight else f'./{config.CHECKPOINT_PATH}/global.s2.bd.mbcnn.weights.h5'
    if not os.path.exists(weight_path):
        logger.error(f"Weight file not found at {weight_path}")
        exit(1)
    
    model.load_weights(weight_path)
    
    # Freeze all layers except the last block (decoder) + output layer
    # This is conservative fine-tuning that prevents overfitting
    # while allowing the model to adapt to the new region
    num_layers_to_unfreeze = 2  # Last block + output layer
    
    for layer in model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FT_LR),  # Fine-tuning learning rate
        loss=t_loss,
        metrics=[F1Score(name='f1')]
    )

    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)

    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)

    callbacks = build_callbacks(city, inputs, model, train_loader, config, is_finetune=True)

    model.fit(train_loader,
            epochs=config.FT_EPOCHS,
            callbacks=callbacks,
            verbose=0)
    
                
elif args.stage == "test":
    from metrics import metrics
    
    model = mbcnn(config.N_CLASSES, input_shapes)
    weight = args.weight if args.weight else (f'./{config.CHECKPOINT_PATH}/{city}.{inputs.lower()}.{model.name}.weights.h5')

    if not os.path.exists(weight):
        logger.error(f"Weight file not found: {weight}")
        raise FileNotFoundError(f"Weight file not found: {weight}")
    
    model.load_weights(weight)

    # Create test dataset using reference codebase approach
    test_dataset = dl.MBCNNDataset(
        os.path.join(dir, 'test'),
        target_size=config.PATCH_SIZE
    )
    test_loader = dl.mbcnn_loader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    class_report, cm, iou_values, mean_iou_value, fwiou_value = metrics.report(test_loader, model)
    
    print("Classification Report:\n", class_report)
    print("IoU Scores per Class:\n", iou_values)
    print(f"Mean IoU: {mean_iou_value:.4f}")
    print(f"Frequency Weighted IoU: {fwiou_value:.4f}")


elif args.stage == "infer":
    from utils.inference import inference_mbcnn

    image_sources = []
    for modality in config.DATASET:
        if modality == 'S2' and args.s2:
            image_sources.append(args.s2)
        elif modality == 'BD' and args.bd:
            image_sources.append(args.bd)

    missing_modalities = set(config.DATASET) - set([ 'S2' if args.s2 else '', 'BD' if args.bd else ''])
    if missing_modalities:
        print(f"Missing required inputs: {missing_modalities}")
        exit(1)
    else:
        pass

    year = args.s2.split('/')[-1].split('_')[1].split('.')[0]

    model = mbcnn(config.N_CLASSES, input_shapes)

    if not args.weight:
        # check default weight path if it exists
        args.weight = f'./{config.CHECKPOINT_PATH}/{city}.{inputs.lower()}.{model.name}.weights.h5'
        if not os.path.exists(args.weight):
            logger.error("Model weight file not provided. Use --weight to specify the path.")
            exit(1)
             
    if os.path.exists(args.weight):
        model.load_weights(args.weight)
    else:
        logger.error(f"Model weight file not found: {args.weight}")
        exit(1)

    inference = inference_mbcnn(
        model=model,
        image_sources=image_sources,
        save_path=f'{config.PREDICTION_PATH}/{city}.{inputs.lower()}.{model.name}.{year}.tif',
        aoi_path=os.path.join(config.AOI_PATH, f"{city}_aoi.geojson"),
        batch_size=config.BATCH_SIZE,
        stride_ratio=config.STRIDE_RATIO
    )
