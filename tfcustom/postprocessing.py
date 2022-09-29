import tensorflow as tf


def predict_tortuosity(
    conc_map: tf.types.experimental.TensorLike,
    soil: tf.types.experimental.TensorLike,
    C_in: float = 1.0,
    C_out: float = 0.5,
    dim: int = 128,
):
    r""" `predict_tortuosity` takes a batch of concentration maps as well as the
    porous media image to estimate tortuosity. `conc_map` from the Neural
    Network prediction.

    The user could specify the inlet and outlet concentrations as well as
    the number of pixel in an axis. Here, it is assumed that the image is
    cubic.
    """

    def _get_flux_mask(batch_size, porous_media):
        # Create a boolean mask to selectively choose valid fluxes to use to
        # compute the total flux from the inlet.
        flux_mask = tf.ones((1, dim - 1, dim, dim), dtype=tf.bool)
        flux_mask = tf.tile(flux_mask, (batch_size, 1, 1, 1))

        # Only evaluate flux on non-solid inlets and "non-blocked" inlets (void
        # inlets with solid space below)
        soil_bool = tf.identity(porous_media)
        soil_bool = tf.cast(soil_bool, tf.bool)

        flux_mask = tf.math.logical_and(
            flux_mask, soil_bool[:, :-1],
        )
        flux_mask = tf.math.logical_and(
            flux_mask, soil_bool[:, 1:],
        )

        # Convert to float32 since multiplication preserves tensor shape
        # whereas a logical operation flattens or turns the tensor into a
        # ragged tensor.
        flux_mask = tf.cast(flux_mask, tf.float32)

        return flux_mask

    def _compute_flux_layer_by_layer(from_inlet, layer_adj_2_inlet):
        # Compute the flux. `reduce_sum` is used twice to reduce the `y` and
        # `z` dimensions.
        #
        # J = -D * (C(1) - C(0)) / dx
        #   D  = 1
        #   dx = 1

        # Computes (dim - 2) differences
        flux = tf.math.subtract(from_inlet, layer_adj_2_inlet)
        flux = tf.math.reduce_sum(flux, axis=3)
        flux = tf.math.reduce_sum(flux, axis=2)

        return flux

    def _compute_porosity(porous_media):
        # Get the effective porosities of the samples
        eps = tf.math.reduce_sum(porous_media, axis=1)
        eps = tf.math.reduce_sum(eps, axis=1)
        eps = tf.math.reduce_sum(eps, axis=1)
        eps = tf.math.scalar_mul(1 / (dim ** 3), eps)

        return eps

    # dC, L, A are scalars - used for effective diffusivity later on
    dC = C_in - C_out
    L = dim
    A = dim ** 2

    # Get the batch size (how many images being handled)
    batch_size = tf.shape(conc_map)[0]

    # Flux masks ignores consecutive pixels corresponding to the solid phase
    # vertically in the x-axis
    flux_mask = _get_flux_mask(batch_size, soil)

    # Ragged boolean mask keeps the shape of the original tensor
    temp_pix0 = tf.math.multiply(
        conc_map[:, :-1],
        flux_mask,
    )
    temp_pix1 = tf.math.multiply(
        conc_map[:, 1:],
        flux_mask,
    )

    flux = _compute_flux_layer_by_layer(temp_pix0, temp_pix1)
    eps = _compute_porosity(soil)

    # Compute effective diffusivity
    Deff = tf.math.scalar_mul((L-1)/A/dC, flux)
    Deff = tf.math.reduce_mean(Deff, axis=1)

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
    given as their raw inputs, i.e., it does not work on a `tf.data.Dataset`.

    This method returns a Relative Absolute Error of the predicted
    concentration map from the Neural Network output against the ground-truth
    results.
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
