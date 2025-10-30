import os
import argparse
from utils.callbacks import build_callbacks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import segmentation_models as sm
import models
import dataloaders as dl
import configs as config


parser = argparse.ArgumentParser(description="Train, evaluate or perform inference on a deep learning model.")
parser.add_argument("--stage", choices=["train", "test", "infer"], required=True, help="Select model stage: train, test or infer")
parser.add_argument("--city", type=str, required=True, help="Specify the city name")
parser.add_argument("--s2", type=str, required=False, help="Sentinel-2 image path (only for inference)")
parser.add_argument("--weight", type=str, required=False, help="Specify path to the model weight")
args = parser.parse_args()

#enable memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)
        pass

# import yaml

# class Config:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)

# # Load YAML file
# with open('config.yaml', 'r') as file:
#     config_dict = yaml.safe_load(file)
#     config = Config(**config_dict)

config = config.Config()
city = args.city
dir = os.path.join(config.DATA_PATH, city)
inputs = ".".join(config.DATASET)

input_shapes = {input: config.IN_SHAPE[input] for input in config.DATASET}
h, w = config.IN_SHAPE[config.DATASET[0]][:2]

if args.stage == "train":
    print('Training model on', city.capitalize(), 'dataset')
    print('Input modality: ', config.DATASET)

    train_dataset = dl.MTCNNDataset(os.path.join(dir, "train"), target_size=(h, w), input_modalities=config.DATASET)
    val_dataset = dl.MTCNNDataset(os.path.join(dir, "val"), target_size=(h, w), input_modalities=config.DATASET)

    train_loader = dl.create_dataloader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = dl.create_dataloader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    cl_dict, cl_list = train_dataset.compute_class_weights()

    dice_loss = sm.losses.DiceLoss(class_weights=cl_list) 
    focal_loss = sm.losses.CategoricalFocalLoss(alpha=cl_list, gamma=2.0)
    t_loss =  dice_loss + (2 * focal_loss)
    j_loss = sm.losses.JaccardLoss(class_weights=cl_list, per_image=False, smooth=1e-05)

    model = models.mtcnn(input_shapes=input_shapes, 
                         classes=config.N_CLASSES, 
                         drop_train=True, 
                         drop_rate=0.25)
    print(f'Model -> {model.name.upper()} | Parameters -> {model.count_params():,}')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
        loss={'reg': 'mae','seg': t_loss},
        loss_weights={'reg': 0.25, 'seg': 0.75},
        metrics={'reg': ['mae'],'seg': [sm.metrics.FScore(name='f1', class_weights=cl_list)]}
    )

    callbacks = build_callbacks(city, inputs, model, train_dataset, config, monitor="val_seg_loss", patience=20)

    model.fit(train_loader, 
            epochs=config.EPOCHS, 
            callbacks=callbacks, 
            validation_data=val_loader, 
            verbose=0)


elif args.stage == "test":
    from metrics.metrics import class_report_mtcnn
    print('Testing model on', city.capitalize(), 'dataset')
    print('Modalities: ', config.DATASET)

    model = models.mtcnn(input_shapes=input_shapes, 
                         classes=config.N_CLASSES, 
                         drop_train=False, 
                         drop_rate=0)
    model.load_weights(f'./{config.CHECKPOINT_PATH}/{city}.{inputs.lower()}.{model.name}.weights.h5')

    test_dataset = dl.MTCNNDataset(os.path.join(dir, "test"), target_size=(h, w), input_modalities=config.DATASET)
    test_loader = dl.create_dataloader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    report, conf_matrix, iou_scores, mean_iou, fwiou = class_report_mtcnn(test_loader, model)

    print("Classification Report:\n", report)
    print("IoU Scores per Class:", iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Frequency Weighted IoU: {fwiou:.4f}")


elif args.stage == "infer":
    from utils.ideatlas.inference import inference_mtcnn
    print('Running inference on', city.capitalize(), 'dataset')
    year = args.s2.split('/')[-1].split('_')[1].split('.')[0]
    print(f'Year: {year}')

    image_sources = {}
    for modality in config.DATASET:
        if modality == 'S1' and args.s1:
            image_sources['S1'] = args.s1
        elif modality == 'S2' and args.s2:
            image_sources['S2'] = args.s2
    
    missing_modalities = set(config.DATASET) - set(image_sources.keys())
    if missing_modalities:
        print(f"Missing required inputs: {missing_modalities}")
        exit(1)
    else:
        print(f"Using input modalities: {list(image_sources.keys())}")
    
    model = models.mtcnn(input_shapes=input_shapes, classes=config.N_CLASSES, drop_train=False, drop_rate=0)
    if not args.weight:
        print("Model weight file not provided. Use --weight to specify the path.")
        exit(1)
    if os.path.exists(args.weight):
        model.load_weights(args.weight)
        print("Model weights loaded")
    else:
        print(f"Model weight file not found: {args.weight}")
        exit(1)
        
    inference = inference_mtcnn(
        model = model,
        image_sources = image_sources,
        save_path = f'{config.PREDICTION_PATH}/{city}.{inputs.lower()}.{model.name}.{year}.tif',
        aoi_path = os.path.join(config.AOI, f"{city}_aoi.geojson"),
        batch_size = config.BATCH_SIZE,
        stride_ratio = config.STRIDE
    )