import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import (Input, Dense, Flatten, Conv2D, Conv2DTranspose, Concatenate,
                                     BatchNormalization, Add, Reshape, Dropout, LeakyReLU, UpSampling2D)

initializer = tf.random_normal_initializer(0, 0.02)

def get_vgg():
    VGG = VGG19(include_top=True, weights='imagenet')
    for layer in VGG.layers:
        layer.trainable=False
    return Model(inputs=VGG.input, outputs=VGG.get_layer("fc1").output)


def vgg_preprocess(image):
    # center crop the image
    image = image[ : , 16:240, 16:240, : ]
    return image


def encoder_unit(input, filter):
    x = input
    x = Conv2D(filer, kernel_size=3, strides=(1, 1), padding='same', 
                use_bias=False, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    skip = x

    x = Conv2D(filter, kernel_size=4, strides=(2, 2), padding='same',
                use_bias=False, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    return x, skip


def decoder_unit(input, skip, filter):
    x = input
    x = Conv2DTranspose(filter, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=initializer)(x)
    x = Conv2D(filter, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filter, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Concatenate()([x, skip])

    return x


def generator_model():
    input1= Input((256, 256, 3), name='content')
    input2= Input((4096, ), name='style')
    vgg = Reshape((1, 1, -1))(input2)
    vgg = UpSampling2D(size=(2, 2), name='up1')(vgg)

    """Start encoder"""
    x, skip1 = encoder_unit(input1, 16)
    x, skip2 = encoder_unit(x, 32)
    x, skip3 = encoder_unit(x, 64)
    x, skip4 = encoder_unit(x, 128)
    x, skip5 = encoder_unit(x, 256)
    """end encoder"""

    # bottleneck
    # TODO: fix scaling->get 8x8x2048 somehow
    # maybe add another layer to vgg and retrain
    x = Reshape((2, 2, 4096))(x)
    x = Add(name="add")([x, vgg])
    x = Reshape((8, 8, 256))(x)

    """start decoder""" 
    x = decoder_unit(x, skip5, 256)
    x = decoder_unit(x, skip4, 128)
    x = decoder_unit(x, skip3, 64)
    x = decoder_unit(x, skip2, 32)
    x = decoder_unit(x, skip1, 16)
    """end decoder"""

    x = Conv2D(3, 3, strides=(1, 1), padding='same', name='final_c', activation='tanh', kernel_initializer=inititalizer)(x)
    
    return Model(inputs=[input1, input2], outputs=x, name='UNet')


def discriminator_model():
    input1 = Input((256, 256, 3))
    x = Conv2D(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=inititalizer)(input1)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=inititalizer)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=inititalizer)(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    model = Model(inputs=input1, outputs=x, name='discrim')
    return model