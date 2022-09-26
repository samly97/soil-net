import tensorflow as tf


def predict_tortuosity(
    model: tf.keras.Model,
    soil: tf.types.experimental.TensorLike,
    skel: tf.types.experimental.TensorLike,
    C_in: float = 1.0,
    C_out: float = 0.5,
    dim: int = 128,
):
    r""" `predict_tortuosity` takes the inputs to the Neural Network and
    estimates the tortuosity from the predicted concentration map.

    The user could specify the inlet and outlet concentrations as well as
    the length of the voxel here. Here, it is assumed that the image is
    cubic.
    """

    # dC, L, A are scalars - used for effective diffusivity later on
    dC = C_in - C_out
    L = dim
    A = dim ** 2

    # Predict concentration map and estimate derivative
    conc_map = model.predict((soil, skel))

    # Get the batch size (how many images being handled)
    batch_size = tf.shape(conc_map)[0]

    # Create a boolean mask to selectively choose valid fluxes to use to
    # compute the total flux from the inlet.
    flux_mask = tf.ones((1, dim, dim), dtype=tf.bool)
    flux_mask = tf.tile(flux_mask, (batch_size, 1, 1))

    # Only evaluate flux on non-solid inlets and "non-blocked" inlets (void
    # inlets with solid space below)
    soil_bool = tf.identity(soil)
    soil_bool = tf.cast(soil_bool, tf.bool)

    flux_mask = tf.math.logical_and(
        flux_mask, tf.gather(soil_bool, 0, axis=1),
    )
    flux_mask = tf.math.logical_and(
        flux_mask, tf.gather(soil_bool, 1, axis=1),
    )

    # Ragged boolean mask keeps the shape of the original tensor
    temp_pix0 = tf.ragged.boolean_mask(
        tf.gather(conc_map, 0, axis=1),
        flux_mask,
    )
    temp_pix1 = tf.ragged.boolean_mask(
        tf.gather(conc_map, 1, axis=1),
        flux_mask,
    )

    # Compute the flux. `reduce_sum` is used twice to reduce the `y` and
    # `z` dimensions.
    #
    # J = -D * (C(1) - C(0)) / dx
    #   D  = 1
    #   dx = 1
    flux = tf.math.subtract(temp_pix0, temp_pix1)
    flux = tf.math.reduce_sum(flux, axis=1)
    flux = tf.math.reduce_sum(flux, axis=1)

    # Get the effective porosities of the samples
    eps = tf.math.reduce_sum(soil, axis=1)
    eps = tf.math.reduce_sum(eps, axis=1)
    eps = tf.math.reduce_sum(eps, axis=1)
    eps = tf.math.scalar_mul(1 / (dim ** 3), eps)

    # Compute effective diffusivity
    Deff = tf.math.scalar_mul((L-1)/A/dC, flux)

    # Compute tortuosity
    tau = tf.math.divide(eps, Deff)

    return tau


def evaluate_abs_rel_err(
    model: tf.keras.Model,
    soil: tf.types.experimental.TensorLike,
    skel: tf.types.experimental.TensorLike,
    target: tf.types.experimental.TensorLike
):
    r""" `evaluate_abs_rel_err` assumes that inputs to the Neural Network is
    given as their raw inputs, i.e., it does not work on a
    `tf.data.Dataset`.

    This method returns a Relative Absolute Error of the predicted
    concentration map from the Neural Network output against the
    ground-truth results.
    """

    # Predict concentration map using Neural Network
    predictions = model.predict((soil, skel))

    # Add arbtirary number to avoid divide by 0
    soil = tf.math.scalar_mul(5., soil)
    target = tf.math.add(soil, target)
    predictions = tf.math.add(soil, predictions)

    # Compute Relative Absolute Error (decimal form)
    relative_err = tf.math.subtract(predictions, target)
    relative_err = tf.math.abs(relative_err)
    relative_err = tf.math.divide(relative_err, target)

    return relative_err
