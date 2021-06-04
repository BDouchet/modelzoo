import tensorflow as tf
from tensorflow.keras import layers

def ASPP(result,nbr=8):
    b0 = layers.DepthwiseConv2D(3, activation='relu', padding='same')(result)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b0)
    b0 = layers.BatchNormalization()(b0)

    b1 = layers.DepthwiseConv2D(3, dilation_rate=(6, 6), activation='relu', padding='same')(result)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b1)
    b1 = layers.BatchNormalization()(b1)

    b2 = layers.DepthwiseConv2D(3, dilation_rate=(12, 12), activation='relu', padding='same')(result)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b2)
    b2 = layers.BatchNormalization()(b2)

    b3 = layers.DepthwiseConv2D(3, dilation_rate=(18, 18), activation='relu', padding='same')(result)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b3)
    b3 = layers.BatchNormalization()(b3)

    b4 = layers.AveragePooling2D()(result)
    b4 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.UpSampling2D(interpolation='bilinear')(b4)

    out = layers.Concatenate()([b4, b0, b1, b2, b3])
    
    return out
