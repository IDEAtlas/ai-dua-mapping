import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Activation
from tensorflow.keras import models, layers
import numpy as np



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

        return x

    # Convolutions on each input
    conv_blocks = []
    for input_tensor in input_tensors:
        conv_blocks.append(conv_block(input_tensor, nfilters[0], dropout_rate=0, batch_norm=batch_norm))

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