# import tensorflow as tf
# import models.misc
# import layers.general


# def DeepLabV3Plus(
#     input_shapes, n_outputs, last_activation="softmax", dropout=0, inference_dropout=False, 
#     backbone_builder=models.misc.ResNeSt101, name="DeepLabv3plus", **kwargs
# ):
#     inputs = []
#     for input_name, input_shape in input_shapes.items():
#         input_layer = tf.keras.layers.Input(input_shape, name=input_name)
#         inputs.append(input_layer)

#     fused = layers.general.FusionBlock(
#         64, 
#         spatial_dropout=dropout, 
#         inference_dropout=inference_dropout,
#     )(inputs)

#     backbone = backbone_builder(
#         fused.shape[1:],
#         use_stem=False,
#         return_low_level=True,
#         dropout=dropout,
#         inference_dropout=inference_dropout,
#     )

#     low_level_features, high_level_features = backbone(fused)
#     high_level_features = layers.general.DilatedSpatialPyramidPooling(256)(high_level_features)

#     outputs = layers.general.DeepLabv3PlusHead(256)([low_level_features, high_level_features])
#     outputs = tf.keras.layers.Dense(
#         n_outputs, activation=last_activation, name="output"
#     )(outputs)

#     model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
#     return model

import tensorflow as tf
from tensorflow.keras import layers, Model

class DeepLabV3Plus:
    def __init__(self, input_shapes, num_classes=1, backbone='resnet101', output_stride=16):
        """
        input_shapes: dict, e.g., {"S1": (128, 128, 2), "S2": (128, 128, 10)}
        num_classes: number of output classes
        backbone: currently only supports 'resnet101'
        output_stride: 8 or 16 (controls ASPP dilation rates)
        """
        self.input_shapes = input_shapes
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.model = self.build_model()

    def build_model(self):
        input_list = [layers.Input(shape=v) for v in self.input_shapes.values()]
        x = layers.Concatenate(axis=-1)(input_list)

        # Backbone: Use pretrained ResNet101 with custom input
        base_model = tf.keras.applications.ResNet101(
            include_top=False,
            weights=None,  # Set to "imagenet" if you want pretrained
            input_tensor=x,
            pooling=None
        )

        # Extract feature maps
        high_level_feature = base_model.get_layer('conv4_block23_out').output  # stride 16
        low_level_feature = base_model.get_layer('conv2_block3_out').output   # stride 4

        # ASPP
        aspp = self._build_aspp(high_level_feature, dilation_rates=[6, 12, 18] if self.output_stride == 16 else [12, 24, 36])

        # Decoder
        decoder_output = self._build_decoder(aspp, low_level_feature)

        # Final output layer
        outputs = layers.Conv2D(self.num_classes, kernel_size=1, activation='softmax' if self.num_classes > 1 else 'sigmoid')(decoder_output)

        return Model(inputs=input_list, outputs=outputs, name='DeepLabV3Plus')

    def _build_aspp(self, x, dilation_rates):
        dims = x.shape[-1]

        pool = layers.GlobalAveragePooling2D()(x)
        pool = layers.Reshape((1, 1, dims))(pool)
        pool = layers.Conv2D(256, 1, padding='same', use_bias=False)(pool)
        pool = layers.BatchNormalization()(pool)
        pool = layers.Activation('relu')(pool)
        pool = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(pool)

        conv_1x1 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        conv_1x1 = layers.BatchNormalization()(conv_1x1)
        conv_1x1 = layers.Activation('relu')(conv_1x1)

        conv_3x3_1 = layers.Conv2D(256, 3, padding='same', dilation_rate=dilation_rates[0], use_bias=False)(x)
        conv_3x3_1 = layers.BatchNormalization()(conv_3x3_1)
        conv_3x3_1 = layers.Activation('relu')(conv_3x3_1)

        conv_3x3_2 = layers.Conv2D(256, 3, padding='same', dilation_rate=dilation_rates[1], use_bias=False)(x)
        conv_3x3_2 = layers.BatchNormalization()(conv_3x3_2)
        conv_3x3_2 = layers.Activation('relu')(conv_3x3_2)

        conv_3x3_3 = layers.Conv2D(256, 3, padding='same', dilation_rate=dilation_rates[2], use_bias=False)(x)
        conv_3x3_3 = layers.BatchNormalization()(conv_3x3_3)
        conv_3x3_3 = layers.Activation('relu')(conv_3x3_3)

        x = layers.Concatenate()([pool, conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3])
        x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return x

    def _build_decoder(self, x, low_level_feat):
        low_level_feat = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_feat)
        low_level_feat = layers.BatchNormalization()(low_level_feat)
        low_level_feat = layers.Activation('relu')(low_level_feat)

        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = layers.Concatenate()([x, low_level_feat])
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        return x