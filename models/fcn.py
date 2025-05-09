import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, BatchNormalization, Activation

def fcn(input_shape, CL):
    """Function to create FCN model using Functional API."""
    
    inputs = Input(shape=input_shape)

    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(filters=32, kernel_size=(7, 7), dilation_rate=(1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)

    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), dilation_rate=(1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)

    outputs = Conv2D(filters=CL, kernel_size=(1, 1), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fcn")
    
    return model

def fcndk3(input_shape, CL):
    """Function to create FCN-DK3 model using Functional API."""
    
    inputs = Input(shape=input_shape)

    x = ZeroPadding2D((2, 2))(inputs)
    x = Conv2D(filters=16, kernel_size=(5, 5), dilation_rate=(1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((2, 2))(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(1, 1))(x)

    x = ZeroPadding2D((4, 4))(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), dilation_rate=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((4, 4))(x)
    x = MaxPooling2D(pool_size=(9, 9), strides=(1, 1))(x)

    x = ZeroPadding2D((6, 6))(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), dilation_rate=(3, 3))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((6, 6))(x)
    x = MaxPooling2D(pool_size=(13, 13), strides=(1, 1))(x)

    outputs = Conv2D(filters=CL, kernel_size=(1, 1), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fcndk3")
    
    return model

def fcndk6(input_shape, CL):
    """Function to create FCN-DK6 model using Functional API."""
    
    inputs = Input(shape=input_shape)

    x = ZeroPadding2D((2, 2))(inputs)
    x = Conv2D(filters=16, kernel_size=(5, 5), dilation_rate=(1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((2, 2))(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(1, 1))(x)

    x = ZeroPadding2D((4, 4))(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), dilation_rate=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((4, 4))(x)
    x = MaxPooling2D(pool_size=(9, 9), strides=(1, 1))(x)

    x = ZeroPadding2D((6, 6))(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), dilation_rate=(3, 3))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((6, 6))(x)
    x = MaxPooling2D(pool_size=(13, 13), strides=(1, 1))(x)

    x = ZeroPadding2D((8, 8))(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), dilation_rate=(4, 4))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D((8, 8))(x)
    x = MaxPooling2D(pool_size=(17, 17), strides=(1, 1))(x)

    outputs = Conv2D(filters=CL, kernel_size=(1, 1), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fcndk6")
    
    return model