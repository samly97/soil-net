import tensorflow as tf


def predict_tortuosity(
    conc_map: tf.types.experimental.TensorLike,
    soil: tf.types.experimental.TensorLike,
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

    def _get_flux_mask(batch_size, porous_media):
        # Create a boolean mask to selectively choose valid fluxes to use to
        # compute the total flux from the inlet.
        flux_mask = tf.ones((1, dim, dim), dtype=tf.bool)
        flux_mask = tf.tile(flux_mask, (batch_size, 1, 1))

        # Only evaluate flux on non-solid inlets and "non-blocked" inlets (void
        # inlets with solid space below)
        soil_bool = tf.identity(porous_media)
        soil_bool = tf.cast(soil_bool, tf.bool)

        flux_mask = tf.math.logical_and(
            flux_mask, tf.gather(soil_bool, 0, axis=1),
        )
        flux_mask = tf.math.logical_and(
            flux_mask, tf.gather(soil_bool, 1, axis=1),
        )

        return flux_mask

    def _compute_flux_from_inlet_layer(inlet_layer, next_to_inlet):
        # Compute the flux. `reduce_sum` is used twice to reduce the `y` and
        # `z` dimensions.
        #
        # J = -D * (C(1) - C(0)) / dx
        #   D  = 1
        #   dx = 1
        flux = tf.math.subtract(inlet_layer, next_to_inlet)
        flux = tf.math.reduce_sum(flux, axis=1)
        flux = tf.math.reduce_sum(flux, axis=1)

        return flux

    def _compute_porosity(porous_media):
        # Get the effective porosities of the samples
        eps = tf.math.reduce_sum(porous_media, axis=1)
        eps = tf.math.reduce_sum(eps, axis=1)
        eps = tf.math.reduce_sum(eps, axis=1)
        eps = tf.math.scalar_mul(1 / (dim ** 3), eps)

        return eps

    def _average_porous_media(porous_media):
        # https://github.com/tldr-group/taufactor/blob/main/taufactor/taufactor.py
        # Lines 74-86

        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1]])

        # Work off a copy
        porous_media = tf.identity(porous_media)

        porous_media = tf.pad(porous_media, paddings,
                              "CONSTANT", constant_values=1)
        porous_media = tf.pad(porous_media, paddings,
                              "CONSTANT", constant_values=0)

        ret = tf.zeros_like(porous_media, tf.float32)

        # Iterate through x, y, z
        for dim in range(1, 4):
            for roll in [1, -1]:
                ret = tf.math.add(ret, tf.roll(porous_media, roll, dim))

        # Remove the 2 pixels of padding
        ret = ret[:, 2:-2, 2:-2, 2:-2]

        # Multiply `ret` with the 0s from `porous_media` to 0 solids
        porous_media = porous_media[:, 2:-2, 2:-2, 2:-2]
        ret = tf.math.multiply(ret, porous_media)

        return ret

    def _average_conc_map(conc_map, rolled_porous_media):
        # https://github.com/tldr-group/taufactor/blob/main/taufactor/taufactor.py
        # Lines 123-129

        no_flux_paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
        top_bc_padding = tf.constant([[0, 0], [1, 0], [0, 0], [0, 0]])
        bot_bc_padding = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]])

        conc_map = tf.pad(conc_map, no_flux_paddings,
                          "CONSTANT", constant_values=0)
        conc_map = tf.pad(conc_map, top_bc_padding,
                          "CONSTANT", constant_values=C_in)
        conc_map = tf.pad(conc_map, bot_bc_padding,
                          "CONSTANT", constant_values=C_out)

        ret = conc_map[:, 2:, 1:-1, 1:-1]
        ret = tf.math.add(ret, conc_map[:, :-2, 1:-1, 1:-1])
        ret = tf.math.add(ret, conc_map[:, 1:-1, 2:, 1:-1])
        ret = tf.math.add(ret, conc_map[:, 1:-1, :-2, 1:-1])
        ret = tf.math.add(ret, conc_map[:, 1:-1, 1:-1, 2:])
        ret = tf.math.add(ret, conc_map[:, 1:-1, 1:-1, :-2])

        # Average the concentration map out. Use `divide_no_nan` just in case
        # the rolled porous media has unexpected 0s outside the solid phase.
        ret = tf.math.divide_no_nan(ret, rolled_porous_media)

        return ret

    # dC, L, A are scalars - used for effective diffusivity later on
    dC = C_in - C_out
    L = dim
    A = dim ** 2

    # Shifted `conc_map` is averaged with shifted porous_media
    rolled_porous_media = _average_porous_media(soil)
    conc_map = _average_conc_map(conc_map, rolled_porous_media)

    # Get the batch size (how many images being handled)
    batch_size = tf.shape(conc_map)[0]

    # Flux masks ignores consecutive pixels corresponding to the solid phase
    # vertically in the x-axis
    flux_mask = _get_flux_mask(batch_size, soil)

    # Ragged boolean mask keeps the shape of the original tensor
    temp_pix0 = tf.ragged.boolean_mask(
        tf.gather(conc_map, 0, axis=1),
        flux_mask,
    )
    temp_pix1 = tf.ragged.boolean_mask(
        tf.gather(conc_map, 1, axis=1),
        flux_mask,
    )

    flux = _compute_flux_from_inlet_layer(temp_pix0, temp_pix1)
    eps = _compute_porosity(soil)

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
