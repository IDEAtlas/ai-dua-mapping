import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Flatten, Dense, BatchNormalization, Dropout, Lambda, Activation, LeakyReLU, ZeroPadding2D, Convolution2D
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import numpy as np


##############################################################

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def UNet(input_shape, CL=1, dropout_rate=0.0, batch_norm=True):

    # network structure
    FILTER_NUM = 64 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    

    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
   
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
   
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
   
    conv_final = layers.Conv2D(CL, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model 
    model = models.Model(inputs, conv_final, name="unet")
    # print(model.summary())
    return model



###--------------------------------------------------------------------------------------####

def lightunet(input_shape, CL, dropout_rate=0.0, batch_norm=True):
    
    H, W, CH = input_shape
    inputs = Input((H, W, CH))
    s = inputs

    def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same', dropout_rate=0.0, batch_norm=True):
        x = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(inputs)
        if batch_norm:
            x = BatchNormalization()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        return x

    # Contraction path
    c1 = conv_block(s, 64, dropout_rate=dropout_rate, batch_norm=batch_norm)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128, dropout_rate=dropout_rate, batch_norm=batch_norm)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256, dropout_rate=dropout_rate, batch_norm=batch_norm)

    # Expansive path
    u4 = UpSampling2D((2, 2))(c3)
    u4 = conv_block(u4, 128, kernel_size=(2, 2), dropout_rate=dropout_rate, batch_norm=batch_norm)
    u4 = concatenate([u4, c2])
    c4 = conv_block(u4, 128, dropout_rate=dropout_rate, batch_norm=batch_norm)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = conv_block(u5, 64, kernel_size=(2, 2), dropout_rate=dropout_rate, batch_norm=batch_norm)
    u5 = concatenate([u5, c1])
    c5 = conv_block(u5, 64, dropout_rate=dropout_rate, batch_norm=batch_norm)

    outputs = Conv2D(CL, (1, 1), activation='softmax')(c5)

    model = Model(inputs=[inputs], outputs=[outputs], name="lightunet")

    return model




'''
Multi-Branch Convolutional Neural Network


███╗   ███╗██████╗         ██████╗███╗   ██╗███╗   ██╗
████╗ ████║██╔══██╗       ██╔════╝████╗  ██║████╗  ██║
██╔████╔██║██████╔╝ ████║ ██║     ██╔██╗ ██║██╔██╗ ██║
██║╚██╔╝██║██╔══██╗       ██║     ██║╚██╗██║██║╚██╗██║
██║ ╚═╝ ██║██████╔╝       ╚██████╗██║ ╚████║██║ ╚████║
╚═╝     ╚═╝╚═════╝         ╚═════╝╚═╝  ╚═══╝╚═╝  ╚═══╝


'''


def mbcnn(CL=3, input_shapes=None, dropout_rate=0.2, batch_norm=False, drop_train=False):
   
    nfilters = np.array([16, 32, 64])
    # nfilters = (nfilters / 8).astype('int')
    # Input tensors for the images
    input_tensors = [Input(shape=shape) for shape in input_shapes.values()]

    def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same', dropout_rate=0.0, batch_norm=True):
        x = Conv2D(filters, kernel_size, activation=None, kernel_initializer=kernel_initializer, padding=padding)(inputs)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x, training=drop_train)
        return x

    # Convolutions on each input
    conv_blocks = []
    for input_tensor in input_tensors:
        conv_blocks.append(conv_block(input_tensor, nfilters[1], dropout_rate=0, batch_norm=batch_norm))

    concat_input = concatenate(conv_blocks)

    e0 = conv_block(concat_input, nfilters[0], dropout_rate=dropout_rate, batch_norm=batch_norm)
    e0 = conv_block(e0, nfilters[0], dropout_rate=dropout_rate, batch_norm=batch_norm)

    e1 = MaxPooling2D((2, 2))(e0)
    e1 = conv_block(e1, nfilters[1], dropout_rate=dropout_rate, batch_norm=batch_norm)
    e1 = conv_block(e1, nfilters[1], dropout_rate=dropout_rate, batch_norm=batch_norm)

    e2 = Dropout(dropout_rate)(e1, training=drop_train)
    e2 = MaxPooling2D((2, 2))(e2)
    e2 = conv_block(e2, nfilters[2], dropout_rate=dropout_rate, batch_norm=batch_norm)
    e2 = conv_block(e2, nfilters[2], dropout_rate=dropout_rate, batch_norm=batch_norm)

    d2 = Dropout(dropout_rate)(e2, training=drop_train)
    d2 = UpSampling2D((2, 2))(d2)
    d2 = concatenate([e1, d2], axis=-1)  # Skip connection
    d2 = Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d2)
    if batch_norm:
        d2 = BatchNormalization(axis=-1)(d2)
    d2 = Activation('relu')(d2)
    d2 = Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d2)
    if batch_norm:
        d2 = BatchNormalization(axis=-1)(d2)
    d2 = Activation('relu')(d2)

    d1 = UpSampling2D((2, 2))(d2)
    d1 = concatenate([e0, d1], axis=-1)  # Skip connection
    d1 = Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d1)
    if batch_norm:
        d1 = BatchNormalization(axis=-1)(d1)
    d1 = Activation('relu')(d1)
    d1 = Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d1)
    if batch_norm:
        d1 = BatchNormalization(axis=-1)(d1)
    d1 = Activation('relu')(d1)

    # Output
    out_class = Conv2D(CL, (1, 1), padding='same')(d1)
    out_class = Activation('softmax', name='output')(out_class)

    model = Model(inputs=input_tensors, outputs=[out_class], name='mbcnn')
    return model



def mtcnn(CL=3, input_shapes=None, dropout_rate=0.2, batch_norm=True, drop_train=True):
    
    nfilters = np.array([64, 128, 256])
    nfilters = (nfilters / 8).astype('int')

    # Input tensors for the images
    input_tensors = [Input(shape=shape) for shape in input_shapes.values()]

    def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same', dropout_rate=0.0, batch_norm=True):
        x = layers.Conv2D(filters, kernel_size, activation=None, kernel_initializer=kernel_initializer, padding=padding)(inputs)
        if batch_norm:
            x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation(activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x, training=drop_train)
        return x

    # Convolutions on each input
    conv_blocks = []
    for input_tensor in input_tensors:
        conv_blocks.append(conv_block(input_tensor, nfilters[0], dropout_rate=0, batch_norm=batch_norm))

    concat_input = layers.concatenate(conv_blocks)

    e0 = conv_block(concat_input, nfilters[0], dropout_rate=dropout_rate, batch_norm=batch_norm)
    e0 = conv_block(e0, nfilters[0], dropout_rate=dropout_rate, batch_norm=batch_norm)

    e1 = layers.MaxPooling2D((2, 2))(e0)
    e1 = conv_block(e1, nfilters[1], dropout_rate=dropout_rate, batch_norm=batch_norm)
    e1 = conv_block(e1, nfilters[1], dropout_rate=dropout_rate, batch_norm=batch_norm)

    e2 = layers.Dropout(dropout_rate)(e1, training=drop_train)
    e2 = layers.MaxPooling2D((2, 2))(e2)
    e2 = conv_block(e2, nfilters[2], dropout_rate=dropout_rate, batch_norm=batch_norm)
    e2 = conv_block(e2, nfilters[2], dropout_rate=dropout_rate, batch_norm=batch_norm)

    d2 = layers.Dropout(dropout_rate)(e2, training=drop_train)
    d2 = layers.UpSampling2D((2, 2))(d2)
    d2 = layers.concatenate([e1, d2], axis=-1)  # Skip connection
    d2 = layers.Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d2)
    if batch_norm:
        d2 = layers.BatchNormalization(axis=-1)(d2)
    d2 = layers.Activation('relu')(d2)
    d2 = layers.Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d2)
    if batch_norm:
        d2 = layers.BatchNormalization(axis=-1)(d2)
    d2 = layers.Activation('relu')(d2)

    d1 = layers.UpSampling2D((2, 2))(d2)
    d1 = layers.concatenate([e0, d1], axis=-1)  # Skip connection
    d1 = layers.Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d1)
    if batch_norm:
        d1 = layers.BatchNormalization(axis=-1)(d1)
    d1 = layers.Activation('relu')(d1)
    d1 = layers.Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d1)
    if batch_norm:
        d1 = layers.BatchNormalization(axis=-1)(d1)
    d1 = layers.Activation('relu')(d1)

    # Output for regression
    out_reg = layers.Conv2D(1, (1, 1), padding='same')(d1)
    out_reg = layers.Activation('sigmoid', name='regression')(out_reg)
    
    # Output for classification
    out_class = layers.concatenate([d1, out_reg], axis=-1)
    out_class = layers.Conv2D(CL, (1, 1), padding='same')(out_class)
    out_class = layers.Activation('softmax', name='segmentation')(out_class)

    model = models.Model(inputs=input_tensors, outputs=[out_reg, out_class], name='mtcnn')
    return model