import models    
import segmentation_models as sm

def select_model(model_name, config):

    inputs = config.DATASET
    input_shapes = {input: config.IN_SHAPE[input] for input in inputs}

    h, w, _ = config.IN_SHAPE[inputs[0]]
    c = sum(config.IN_SHAPE[input][2] for input in inputs)  # Sum channel dimensions
    
    n_outputs = config.N_CLASSES
    if model_name == 'mbcnn':
        model = models.mbcnn(CL=n_outputs, input_shapes=input_shapes, dropout_rate=0.2, batch_norm=True, drop_train=False)
    elif model_name == 'lightunet':
        model = models.lightunet(input_shape = (h,w,c), CL=n_outputs, dropout_rate=0.2, batch_norm=True)
    elif model_name == 'fcndk6':
        model = models.fcndk6(input_shape=(h,w,c), CL=n_outputs)
    elif model_name == 'glavitu':
        model = models.GlaViTU(input_shapes=input_shapes, n_outputs=n_outputs)[1]
    elif model_name == 'deeplab':
        # model = models.DeepLabV3Plus(input_shapes=input_shapes, n_outputs=n_outputs)[1]
        model = models.DeepLabV3Plus(input_shapes=input_shapes, num_classes=n_outputs).model
    elif model_name == 'unet':
        model = sm.Unet(input_shape=(h,w,c), classes=n_outputs, activation='softmax',  encoder_weights=None)
    elif model_name == 'fpn':
        model = sm.FPN(input_shape=(h,w,c), classes=n_outputs, activation='softmax', encoder_weights=None, backbone_name='resnet34')
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model