# Training, testing and inference pipeline 

import os
import argparse
from utils.callbacks import build_callbacks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import models
import losses
import dataloaders as dl
import configs as config



parser = argparse.ArgumentParser(description="Train, evaluate or save a deep learning model.")
parser.add_argument("--stage", choices=["train", "test", "infer"], required=True, help="Select mode: train, test or save")
parser.add_argument("--city", type=str, required=True, help="Specify the city name")
parser.add_argument("--model", type=str, required=True, choices=['unet', 'fcndk6', 'deeplab', 'glavitu', 'mbcnn', 'lightunet', 'fpn'], help="Choose one of the model to use")
parser.add_argument("--s2", type=str, required=False, help="Sentinel-2 image path (only for inference)")
parser.add_argument("--bd", type=str, required=False, help="Building density raster path (only for inference)")
parser.add_argument("--weight", type=str, required=False, help="Specify path to the model weight")
args = parser.parse_args()

config = config.Config()
city = args.city
dir = os.path.join(config.DATA_PATH, city)
inputs = ".".join(config.DATASET)
h, w, _ = config.IN_SHAPE[config.DATASET[0]]

if args.stage == "train":

    print('Training model on', city.capitalize(), 'dataset')
    print('Input modality: ', config.DATASET)

    train_dataset = dl.MBCNNDataset(os.path.join(dir, "train"), target_size=(h, w))
    val_dataset = dl.MBCNNDataset(os.path.join(dir, "val"), target_size=(h, w))

    train_loader = dl.mbcnn_loader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = dl.mbcnn_loader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    sample_inputs, sample_mask = next(iter(val_loader))
    if len(inputs) == 1:
        print(f"Single input shape: {sample_inputs.shape}")
    else:
        for i, input_tensor in enumerate(sample_inputs):
            print(f"Input {inputs[i]} shape: {input_tensor.shape}")
    print(f"Mask shape: {sample_mask.shape}")

    cl_dict, cl_list = train_dataset.compute_class_weights()

    import segmentation_models as sm
    dice_loss = sm.losses.DiceLoss(class_weights=cl_list) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    t_loss =  dice_loss + (2 * focal_loss)
    # j_loss = sm.losses.JaccardLoss(class_weights=cl_weights, class_indexes=None, per_image=False, smooth=1e-05)
    # dice_focal = losses.CombinedDiceFocalLoss(class_idx = 2, gamma=2.0, alpha=0.25, dice_weight=0.25, focal_weight=0.75, class_weights=cl_weights)
    # focal = losses.FocalLoss(gamma=2.0, alphas=0.25)

    model = models.select_model(args.model, config)
    print(f'Model -> {model.name.upper()} | Parameters -> {model.count_params():,}')

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
    loss=t_loss,
    metrics=[sm.metrics.FScore(average='macro', name='f1')]         
    )

    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)

    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)
    
    callbacks = build_callbacks(city, inputs, model, None, config, monitor="val_loss", patience=20)

    model.fit(train_loader,
            epochs=config.EPOCHS,
            callbacks=[callbacks],
            validation_data=val_loader,
            verbose=0)
                
elif args.stage == "test":

    from metrics import metrics
    print('Evaluating model on', city.capitalize(), 'dataset \n')
    print('Modalities: ', config.DATASET)

    model = models.select_model(args.model, config)
    weight = args.weight if args.weight else (f'./{config.CHECKPOINT_PATH}/{city}.{inputs.lower()}.{model.name}.weights.h5')
    # weight = './checkpoint/best/buenos_aires_s2_morph_mbcnn.weights.h5'
    print(f'Model weight: {weight}')

    # Raise an exception if the weights file does not exist
    if not os.path.exists(weight):
        raise FileNotFoundError(f"Weight file not found: {weight}\n"
                            "Check --weight, config.CHECKPOINT_PATH, inputs in config.json, or run training to create this file.")
    
    model.load_weights(weight)

    # test_dataset = dl.RasterDataset(os.path.join(dir, "test"))
    # test_loader = dl.create_dataloader(test_dataset, batch_size=batch, shuffle=False)

    test_images = [dl.load_data(os.path.join(dir, 'test'), None, None, input) for input in inputs]
    if len(inputs) == 1:
        test_images = test_images[0]

    test_label = dl.load_data(os.path.join(dir, 'test'), None, None, 'RF', config.N_CLASSES)

    report, conf_matrix, iou_scores, mean_iou, fwiou = metrics.class_report(test_images, test_label, model)

    print("Classification Report:\n", report)
    print("IoU Scores per Class:\n", iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Frequency Weighted IoU: {fwiou:.4f}")


elif args.stage == "infer":
    from utils.ideatlas.inference import inference_mbcnn
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

    model = models.select_model(args.model, config)

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
        aoi_path=os.path.join(config.AOI, f"{city}_aoi.geojson"),
        batch_size=config.BATCH_SIZE,
        stride_ratio=config.STRIDE
    )
