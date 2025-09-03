# Training, testing and inference pipeline 

import os
import argparse
import numpy as np
from glob import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import models
import metrics
import losses
import dataloaders as dl
from utils import ideatlas, deeplearning
from tqdm.keras import TqdmCallback
import math

import configs as config
config = config.Config()


parser = argparse.ArgumentParser(description="Train, evaluate or save a deep learning model.")
parser.add_argument("--stage", choices=["train", "test", "infer"], required=True, help="Select mode: train, test or save")
parser.add_argument("--city", type=str, required=True, help="Specify the city name")
parser.add_argument("--model", type=str, required=True, choices=['unet', 'fcndk6', 'deeplab', 'glavitu', 'mbcnn', 'lightunet', 'fpn'], help="Choose one of the model to use")
parser.add_argument("--epochs", type=int, required=False, help="Specify the number of epochs")
parser.add_argument("--batch", type=int, required=False, help="Specify the batch size")
parser.add_argument("--s2", type=str, required=False, help="Specify the Sentinel-2 image path")
parser.add_argument("--bd", type=str, required=False, help="Specify the building density raster path")
parser.add_argument("--weight", type=str, required=False, help="Specify path to the model weight")

args = parser.parse_args()


def print_info(name, data):
    if isinstance(data, list):  # Multiple inputs (list of arrays)
        print(f"{name} Shape: {[d.shape for d in data]}")
    else:  # Single input case
        print(f"{name} Shape: {data.shape}")


batch = args.batch if args.batch else config.BATCH_SIZE
epochs = args.epochs if args.epochs else config.EPOCHS
city = args.city
dir = os.path.join(config.DATA_PATH, city)


inputs = config.DATASET
inputs_str = "_".join(inputs)

input_shapes = {input: config.IN_SHAPE[input] for input in inputs}

h, w, _ = config.IN_SHAPE[inputs[0]]
c = sum(config.IN_SHAPE[input][2] for input in inputs)  # Sum channel dimensions
in_shape = [h, w, c]

# for input, shape in input_shapes.items():
#     print(f"Shape of {input} is {shape}")

print(f"Input shape: {in_shape}") 


if args.stage == "train":

    print('Training model on', city.capitalize(), 'dataset')
    print('Input: ', inputs)

    train_images = [dl.load_data(os.path.join(dir, 'train'), h, w, input) for input in inputs]
    val_images = [dl.load_data(os.path.join(dir, 'val'), h, w, input) for input in inputs]

    if len(inputs) == 1:
        train_images = train_images[0]
        val_images = val_images[0]

    train_label = dl.load_data(os.path.join(dir, 'train'), h, w, 'RF', config.N_CLASSES)
    val_label = dl.load_data(os.path.join(dir, 'val'), h, w, 'RF', config.N_CLASSES)

    print_info("Train Images", train_images)
    print_info("Validation Images", val_images)
    print_info("Train Label", train_label)
    print_info("Validation Label", val_label)

    # trainData = glob(os.path.join(config['vhr'] + 'train_im', 'HR*.tif'))
    # validData = glob(os.path.join(config['vhr'] + 'valid_im', 'HR*.tif'))

    # train_datagen = ideatlas.image_generator(trainData, batch, (h,w,4), config['n_classes'], config['vhr'] + 'train_gt')
    # validation_datagen = ideatlas.image_generator(validData, batch, (h,w,4), config['n_classes'], config['vhr'] + 'valid_gt')

    # ideatlas.patch_class_proportion(train_masks)
    cl_weights = dl.calculate_class_weights(train_label)
    print(f'Class weight: {cl_weights}')
    import segmentation_models as sm
    dice_loss = sm.losses.DiceLoss(class_weights=cl_weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    t_loss =  dice_loss + (2 * focal_loss)
    j_loss = sm.losses.JaccardLoss(class_weights=cl_weights, class_indexes=None, per_image=False, smooth=1e-05)
    
    dice_focal = losses.CombinedDiceFocalLoss(class_idx = 2, gamma=2.0, alpha=0.25, dice_weight=0.25, focal_weight=0.75, class_weights=cl_weights)
    focal = losses.FocalLoss(gamma=2.0, alphas=0.25)

    model = models.select_model(args.model, config)
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
    loss=t_loss,
    metrics=[sm.metrics.FScore(threshold=0.5, class_indexes=[0,1,2], name='f1', class_weights=cl_weights)]         
    )

    print(f'Model -> {model.name} \n'
          f'Parameters -> {model.count_params():,} \n')

    # model.compile(
    #     optimizer='adam',
    #     loss={
    #         'regression': 'mean_squared_error',
    #         'segmentation': losses.FocalLoss()
    #     },
    #     loss_weights={
    #         'regression': 1.0,
    #         'segmentation': 1.0
    #     },
    #     metrics={
    #         'regression': ['mean_squared_error'],
    #         'segmentation': [metrics.IoU()]
    #     }
    # )

    # Check if CHECKPOINT_PATH exists and creates it if it doesn't
    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)

    # Remove old checkpoint files
    for i in os.listdir(config.CHECKPOINT_PATH):
        if i.endswith('.h5'):
            os.remove(os.path.join(config.CHECKPOINT_PATH, i))

    # Check if LOG_PATH exists and creates it if it doesn't
    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)

    # Remove old log files
    for j in os.listdir(config.LOG_PATH):
        if j.endswith('.csv'):
            os.remove(os.path.join(config.LOG_PATH, j))
    
    # define callbacks for saving logs, model weights, early stopping and learning rate schedule
    callbacks = [
        # deeplearning.LRWarmup(
        #     warmup_steps=len(train_s2) // batch,
        #     target=config['lr'],
        #     verbose=0,
        # ),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss",
        #     mode="min",
        #     factor=0.1,
        #     patience=10,
        #     verbose=0,
        # ),
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     mode="min",
        #     patience=30,
        #     verbose=1,
        # ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(f'./{config.CHECKPOINT_PATH}/{city}.{inputs_str}.{model.name}.weights.h5'),
            monitor=f"val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join("log", f"{city}.{inputs_str}.{model.name}.log.csv"),
        ),
        # deeplearning.training_progress(epochs, steps_per_epoch)
        TqdmCallback()
    ]

    ## Fit with multi task learning
    # hist = model.fit(train_images, [train_bd, train_masks],
    #                 batch_size = batch,
    #                 steps_per_epoch = len(train_s2) // batch,
    #                 epochs=epochs,
    #                 callbacks=callbacks, 
    #                 validation_data=(val_images, [valid_bd, valid_masks]),
    #                 validation_steps = len(valid_s2) // batch)

    #Fit with dataloader
    # hist = model.fit(train_datagen,
    #                     steps_per_epoch=len(trainData) // batch,
    #                     epochs=epochs,
    #                     callbacks=[callbacks],
    #                     validation_data=validation_datagen,
    #                     validation_steps=len(validData) // batch,
    #                     verbose=0)

    #Regular Fit
    model.fit(train_images, train_label,
            batch_size=batch,
            steps_per_epoch = math.ceil(len(train_label) / batch),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(val_images, val_label),
            validation_steps = len(val_label) // batch,
            verbose=0)
                
elif args.stage == "test":
    print('Evaluating model on', city.capitalize(), 'dataset \n')
    print('Input: ', inputs)
    model = models.select_model(args.model, config)
    weight = args.weight if args.weight else (f'./{config.CHECKPOINT_PATH}/{city}.{inputs_str}.{model.name}.weights.h5')
    # weight = './checkpoint/best/buenos_aires_s2_morph_mbcnn.weights.h5'
    print(f'Model weight: {weight}')

    # Raise an exception if the weights file does not exist
    if not os.path.exists(weight):
        raise FileNotFoundError(f"Weight file not found: {weight}\n"
                            "Check --weight, config.CHECKPOINT_PATH, inputs in config.json, or run training to create this file.")
    
    model.load_weights(weight)

    test_images = [dl.load_data(os.path.join(dir, 'train'), h, w, input) for input in inputs]
    if len(inputs) == 1:
        test_images = test_images[0]

    test_label = dl.load_data(os.path.join(dir, 'test'), h, w, 'RF', config.N_CLASSES)

    print_info("Test Images", test_images)
    print_info("Test Label", test_label)

    report, conf_matrix, iou_scores, mean_iou, fwiou = metrics.class_report(test_images, test_label, model)

    print("Classification Report:\n", report)
    print("IoU Scores per Class:\n", iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Frequency Weighted IoU: {fwiou:.4f}")


elif args.stage == "infer":
    print('Predicting on', city.capitalize(), 'dataset \n')
    s2_path = args.s2
    bd_path = args.bd
    input_rasters = [s2_path, bd_path]
    year = s2_path.split('/')[-1].split('_')[1].split('.')[0]
    print(f'Year: {year}')

    model = models.select_model(args.model, config)
    weight = args.weight if args.weight else (f'./{config.CHECKPOINT_PATH}/{city}.{inputs_str}.{model.name}.weights.h5')
    
    # Raise an exception if the weights file does not exist
    if not os.path.exists(weight):
        raise FileNotFoundError(f"Weight file not found: {weight}\n"
                            "Check --weight, config.CHECKPOINT_PATH, inputs in config.json, or run training to create this file.")
    
    model.load_weights(weight)
    aoi_path = f"{config.AOI}/{city}_aoi.geojson"
    save_path = f'{config.PREDICTION_PATH}/{city}_{inputs_str}_{model.name}_{year}.tif'
    ideatlas.full_inference_mbcnn(config.N_CLASSES, input_rasters, model, save_path, aoi_path, batch_size=32)
    # ideatlas.slide_window_inference(config.N_CLASSES, s2_path, model, save_path)