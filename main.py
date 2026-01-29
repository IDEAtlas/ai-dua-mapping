# Training, testing and inference pipeline 

import os
import argparse
from utils.configs import Config
from utils.callbacks import build_callbacks
import tensorflow as tf
from models.utils import select_model
import losses
import segmentation_models as sm
from utils import dataloader as dl

import math

parser = argparse.ArgumentParser(description="Train, evaluate or save a deep learning model.")
parser.add_argument("--stage", choices=["train", "test", "infer"], required=True, help="Select mode: train, test or save")
parser.add_argument("--city", type=str, required=True, help="Specify the city name")
parser.add_argument("--model", type=str, required=True, choices=['unet', 'fcndk6', 'deeplab', 'glavitu', 'mbcnn', 'lightunet', 'fpn'], 
                    help="Choose one of the model to use")
parser.add_argument("--s2", type=str, required=False, help="Sentinel-2 image path (only for inference)")
parser.add_argument("--bd", type=str, required=False, help="Building density raster path (only for inference)")
parser.add_argument("--weight", type=str, required=False, help="Specify path to the model weight")

args = parser.parse_args()


def print_info(name, data):
    if isinstance(data, list):  # Multiple inputs (list of arrays)
        print(f"{name} Shape: {[d.shape for d in data]}")
    else:  # Single input case
        print(f"{name} Shape: {data.shape}")

config = Config('config.yaml')

city = args.city
dir = os.path.join(config.DATA_PATH, city)
inputs = ".".join(config.DATASET)

input_shapes = {input: config.IN_SHAPE[input] for input in config.DATASET}
h, w, _ = config.IN_SHAPE[config.DATASET[0]]


if args.stage == "train":

    print('Training model on', city.capitalize(), 'dataset')
    print('Input modality: ', config.DATASET)

    train_images = [dl.load_data(os.path.join(dir, 'train'), config.PATCH_SIZE, input) for input in config.DATASET]
    val_images = [dl.load_data(os.path.join(dir, 'val'), config.PATCH_SIZE, input) for input in config.DATASET]

    if len(config.DATASET) == 1:
        train_images = train_images[0]
        val_images = val_images[0]

    train_label = dl.load_data(os.path.join(dir, 'train'), config.PATCH_SIZE, 'RF', config.N_CLASSES)
    val_label = dl.load_data(os.path.join(dir, 'val'), config.PATCH_SIZE, 'RF', config.N_CLASSES)

    print_info("Train Images", train_images)
    print_info("Validation Images", val_images)
    print_info("Train Label", train_label)
    print_info("Validation Label", val_label)

    cl_weights = dl.calculate_class_weights(train_label)
    print(f'Class weight: {cl_weights}')
    
    dice_loss = sm.losses.DiceLoss(class_weights=cl_weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    t_loss =  dice_loss + (2 * focal_loss)
    j_loss = sm.losses.JaccardLoss(class_weights=cl_weights, class_indexes=None, per_image=False, smooth=1e-05)
    
    # dice_focal = losses.CombinedDiceFocalLoss(class_idx = 2, gamma=2.0, alpha=0.25, dice_weight=0.25, focal_weight=0.75, class_weights=cl_weights)
    # focal = losses.FocalLoss(gamma=2.0, alphas=0.25)

    model = select_model(args.model, config)
    print(f'Model -> {model.name.upper()} | Parameters -> {model.count_params():,}')
    

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
    loss=t_loss,
    metrics=[sm.metrics.FScore(name='f1')]         
    )

    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)

    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)

    callbacks = build_callbacks(city, inputs, model, train_images[0], config)

    model.fit(train_images, train_label,
            batch_size=config.BATCH_SIZE,
            steps_per_epoch = math.ceil(len(train_label) / config.BATCH_SIZE),
            epochs=config.N_EPOCHS,
            callbacks=callbacks,
            validation_data=(val_images, val_label),
            validation_steps = len(val_label) // config.BATCH_SIZE,
            verbose=0)
                
elif args.stage == "test":
    import metrics
    print('Evaluating model on', city.capitalize(), 'dataset \n')
    print('Input modality: ', config.DATASET)

    model = select_model(args.model, config)
    weight = args.weight if args.weight else (f'./{config.CHECKPOINT_PATH}/{city}.{inputs.lower()}.{model.name}.weights.h5')
    print(f'Model weight: {weight}')

    if not os.path.exists(weight):
        raise FileNotFoundError(f"Weight file not found: {weight}")
    model.load_weights(weight)

    test_images = [dl.load_data(os.path.join(dir, 'test'), config.PATCH_SIZE, input) for input in config.DATASET]
    if len(config.DATASET) == 1:
        test_images = test_images[0]

    test_label = dl.load_data(os.path.join(dir, 'test'), config.PATCH_SIZE, 'RF', config.N_CLASSES)

    print_info("Test Images", test_images)
    print_info("Test Label", test_label)

    report, conf_matrix, iou_scores, mean_iou, fwiou = metrics.class_report(test_images, test_label, model)

    print("Classification Report:\n", report)
    print("IoU Scores per Class:\n", iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Frequency Weighted IoU: {fwiou:.4f}")


elif args.stage == "infer":
    from utils.inference import inference_mbcnn
    print('Predicting on', city.capitalize(), 'dataset \n')

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
        print(f"Using input modalities: {config.DATASET}")

    year = args.s2.split('/')[-1].split('_')[1].split('.')[0]
    print(f'Year: {year}')

    model = select_model(args.model, config)

    if not args.weight:
        print("Model weight file not provided. Use --weight to specify the path.")
        exit(1)
    if os.path.exists(args.weight):
        model.load_weights(args.weight)
        print("Model weights loaded")
    else:
        print(f"Model weight file not found: {args.weight}")
        exit(1)

    inference = inference_mbcnn(
        model=model,
        image_sources=image_sources,
        save_path=f'{config.PREDICTION_PATH}/{city}.{inputs.lower()}.{model.name}.{year}.tif',
        aoi_path=os.path.join(config.AOI_PATH, f"{city}_aoi.geojson"),
        batch_size=config.BATCH_SIZE,
        stride_ratio=config.STRIDE
    )