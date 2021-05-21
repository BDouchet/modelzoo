"""
Useful loss functions for Keras:
- Weighted Categorical Cross Entropy
- Dice Loss
- Chamfer Loss
"""

import tensorflow.keras.backend as K

def wcce(weights):
    weights = K.variable(weights,dtype='float32')
    def loss(y_true, y_pred):
        y_true=K.cast(y_true,dtype='float32')
        y_pred=K.cast(y_pred,dtype='float32')
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        return -K.sum(loss, -1)
    return loss
  
def dice_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    inter = K.sum(y_pred * y_true)
    dice = (2*inter + K.epsilon()) / (K.sum(y_pred) + K.sum(y_true) + K.epsilon())
    return 1 - dice
  
def chamfer_loss(y_true,y_pred):
    # taken from https://www.tensorflow.org/graphics/api_docs/python/tfg/nn/loss/chamfer_distance/evaluate
    point_set_a = tf.convert_to_tensor(value=y_true)
    point_set_b = tf.convert_to_tensor(value=y_pred)

    difference = (
        tf.expand_dims(point_set_a, axis=-2) -
        tf.expand_dims(point_set_b, axis=-3))

    square_distances = tf.einsum("...i,...i->...", difference, difference)

    minimum_square_distance_a_to_b = tf.reduce_min(
        input_tensor=square_distances, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_min(
        input_tensor=square_distances, axis=-2)

    return (
        tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))
