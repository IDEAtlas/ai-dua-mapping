import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout
from tensorflow.keras import models, layers

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


