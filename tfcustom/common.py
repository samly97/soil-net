import tensorflow as tf


def get_masked_conc_map(y_true, y_pred, dim=128):
    r""" `get_masked_conc_map` is common to the custom loss and metric which
    computes the Mean Square Error (MSE) only on the void space within a porous
    media, as this is the phase of interest.

    This function takes in `y_true` and `y_pred` which are TensorFlow tensors
    representing the concentration maps of a steady-state diffusion problem.
    """

    # "Recover" the porous media from `y_true`
    p_media = tf.math.divide_no_nan(y_true, y_true)
    p_media = tf.cast(p_media, tf.bool)

    p_media.set_shape([None, dim, dim, dim])

    # Mask out the void space. Note that the number of elements is different
    # between each porous media. If low porosity, then less number of elements.
    y_true = tf.boolean_mask(y_true, p_media)
    y_pred = tf.boolean_mask(y_pred, p_media)

    return tf.reduce_mean(tf.square(y_true - y_pred))
