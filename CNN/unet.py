import tensorflow as tf
from tensorflow.keras import layers, models


def unet(height=400,width=400,nbr_mask=10,nbr=64,activation='softmax'):
    
    #Inputimage
    entree = layers.Input(shape=(height, width, 3), dtype='float32')

    #Level 0
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(entree)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result1 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result1)

    #Level -1
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result2 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result2)

    #Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result3 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result3)

    #Level -3
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result4 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result4)

    #Level -4
    result = layers.Conv2D(16 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(16 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(8*nbr,2,strides=(2,2),activation='relu',padding='same')(result)

    # Level -3
    result = tf.concat([result, result4], axis=3)
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(4 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    #Level -2
    result = tf.concat([result, result3], axis=3)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(2*nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)
    #result = layers.UpSampling2D()(result)

    #Level -1
    result = tf.concat([result, result2], axis=3)
    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    #Level 0
    result = tf.concat([result, result1], axis=3)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    #Output
    sortie = layers.Conv2D(nbr_mask, 1, activation=activation, padding='same')(result)

    model = models.Model(inputs=entree, outputs=sortie)
    return model
