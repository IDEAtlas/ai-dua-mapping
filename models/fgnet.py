import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

def FGNet(input_shapes, classes, drop_train=True, drop_rate=0.25):
    nfilters = np.array([16, 32, 64])
    if not input_shapes:
        raise ValueError("input_shapes cannot be empty")
    
    # Create input tensors dynamically
    input_tensors = [tf.keras.Input(shape=shape, name=name) 
                    for name, shape in input_shapes.items()]
    
    #------------------------------------
    # Early Fusion
    #------------------------------------
    def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', 
                   name_prefix=""):
        """Reusable conv block for processing each input"""
        x = layers.Conv2D(filters, kernel_size, padding="same", 
                         name=f"{name_prefix}_conv")(inputs)
        x = layers.BatchNormalization(name=f'{name_prefix}_BN')(x)
        x = layers.Activation(activation, name=f'{name_prefix}_Activation')(x)
        return x
    
    # Process each input separately
    conv_blocks = []
    for i, input_tensor in enumerate(input_tensors):
        input_name = list(input_shapes.keys())[i]  # Get input name (S1, S2, etc.)
        processed = conv_block(input_tensor, filters=10, name_prefix=f"Input{i+1}_{input_name}")
        conv_blocks.append(processed)
    
    # Fusion: depend on number of inputs
    if len(conv_blocks) > 1:
        c = layers.concatenate(conv_blocks, axis=-1, name="input_concat")
    else:
        c = conv_blocks[0]  # Single input case
    
    # Fusion convolution
    c = layers.Conv2D(15, kernel_size=1, name="data_fusion")(c)
    c = layers.BatchNormalization(name='data_fusion_BN')(c)
    c = layers.Activation('relu', name='data_fusion_Activation')(c)

    #------------------------------------
    # Encoder 
    #------------------------------------
    # block 0
    e0 = layers.Conv2D(filters=nfilters[0], kernel_size=(3, 3), padding='same')(c)
    e0 = layers.BatchNormalization(axis=-1)(e0)
    e0 = layers.Activation('relu')(e0)
    e0 = layers.Conv2D(filters=nfilters[0], kernel_size=(3, 3), padding='same')(e0)
    e0 = layers.BatchNormalization(axis=-1)(e0)
    e0 = layers.Activation('relu')(e0)

    # block 1
    e1 = layers.MaxPooling2D((2, 2))(e0)
    e1 = layers.Conv2D(filters=nfilters[1], kernel_size=(3, 3), padding='same')(e1)
    e1 = layers.BatchNormalization(axis=-1)(e1)
    e1 = layers.Activation('relu')(e1)
    e1 = layers.Conv2D(filters=nfilters[1], kernel_size=(3, 3), padding='same')(e1)
    e1 = layers.BatchNormalization(axis=-1)(e1)
    e1 = layers.Activation('relu')(e1)

    # bottleneck
    b = layers.Dropout(drop_rate)(e1, training=drop_train)
    b = layers.MaxPooling2D((2, 2))(b)
    b = layers.Conv2D(filters=nfilters[2], kernel_size=(3, 3), padding='same')(b)
    b = layers.BatchNormalization(axis=-1)(b)
    b = layers.Activation('relu')(b)
    b = layers.Conv2D(filters=nfilters[2], kernel_size=(3, 3), padding='same')(b)
    b = layers.BatchNormalization(axis=-1)(b)
    b = layers.Activation('relu')(b)

    #------------------------------------
    # Decoder
    #------------------------------------
    # block 1
    d1 = layers.Dropout(drop_rate)(b, training=drop_train)
    d1 = layers.UpSampling2D((2, 2))(d1)
    d1 = layers.concatenate([e1, d1], axis=-1)  # Skip connection
    d1 = layers.Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
    d1 = layers.BatchNormalization(axis=-1)(d1)
    d1 = layers.Activation('relu')(d1)
    d1 = layers.Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
    d1 = layers.BatchNormalization(axis=-1)(d1)
    d1 = layers.Activation('relu')(d1)

    # block 0
    d0 = layers.UpSampling2D((2, 2))(d1)
    d0 = layers.concatenate([e0, d0], axis=-1)  # Skip connection
    d0 = layers.Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
    d0 = layers.BatchNormalization(axis=-1)(d0)
    d0 = layers.Activation('relu')(d0)
    d0 = layers.Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
    d0 = layers.BatchNormalization(axis=-1)(d0)
    d0 = layers.Activation('relu')(d0)

    # Output
    out_reg = layers.Conv2D(1, (1, 1), padding='same')(d0)
    out_reg = layers.Activation('sigmoid', name='reg')(out_reg)
    out_class = layers.concatenate([d0, out_reg], axis=-1)
    out_class = layers.Conv2D(classes, kernel_size=1, name="final_conv")(out_class)
    out_class = layers.Activation(activation="softmax", name="seg")(out_class)
    
    # Create model
    model = tf.keras.Model(inputs=input_tensors, outputs=[out_reg, out_class], name="fgnet")
    return model