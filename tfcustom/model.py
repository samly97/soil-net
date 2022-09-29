import tensorflow as tf
import numpy as np


class TauNet():

    def __init__(
        self,
        filters=[16, 32, 64, 128, 256, 512],
        dim: int = 128,
        latentDim: int = 16,
        C_in: float = 1.0,
        C_out: float = 0.5,
    ):
        self.filters = filters
        self.dim = dim
        self.shape = (dim, dim, dim)
        self.latentDim = latentDim

        self.analytical = self._get_analytical(C_in, C_out)

    def _get_analytical(self, C_in, C_out):

        # Return analytical solution as shape = (1, dim, dim, dim, 1)
        ret_arr = tf.ones((self.dim, self.dim, self.dim), dtype=tf.float32)

        # Compute analytical solution
        x_arr = tf.linspace(0, self.dim, self.dim)
        analytical = -(C_in - C_out)/self.dim * x_arr + C_in
        analytical = tf.cast(analytical, tf.float32)
        analytical = tf.reshape(analytical, (self.dim, 1, 1))

        # Now analytical solution is "repeated" across the x axis
        ret_arr = tf.math.multiply(ret_arr, analytical)

        # Reshape the dimensions now so it's ready for tiling
        ret_arr = tf.reshape(ret_arr, (1, self.dim, self.dim, self.dim, 1))

        return ret_arr

    def _enforce_hard_bcs(self, pred, soil, inBC, outBC):
        dim = self.dim

        # Use porous media as a boolean mask
        soil_bool = tf.identity(soil)
        soil_bool = tf.cast(soil_bool, tf.bool)

        # Enforce Boundary conditions as hard constraints
        left_boundary = tf.zeros((1, dim - 1, dim, dim), dtype=tf.bool)
        left_boundary = tf.pad(left_boundary, [[0, 0], [1, 0], [
                               0, 0], [0, 0]], constant_values=True)

        right_boundary = tf.zeros((1, dim - 1, dim, dim), dtype=tf.bool)
        right_boundary = tf.pad(right_boundary, [[0, 0], [0, 1], [
                                0, 0], [0, 0]], constant_values=True)

        # 0 out all the boundaries that are solids
        left_boundary = tf.logical_and(left_boundary, soil_bool)
        right_boundary = tf.logical_and(right_boundary, soil_bool)

        # Cast boundaries to scalar now
        left_BC = tf.cast(left_boundary, tf.float32)
        right_BC = tf.cast(right_boundary, tf.float32)

        left_BC = tf.math.scalar_mul(inBC, left_BC)
        right_BC = tf.math.scalar_mul(outBC, right_BC)

        # Delete the boundaries from the prediction
        body = tf.ones((1, dim-2, dim, dim), dtype=tf.float32)
        body = tf.pad(body, [[0, 0], [1, 1], [0, 0],
                      [0, 0]], constant_values=0)

        # Get rid of boundaries from prediction
        pred = tf.math.multiply(pred, body)

        return pred, left_BC, right_BC

    def _get_encoder(self, soil_input, skel_input):

        # Porous Media Input
        x = soil_input
        x = tf.keras.layers.Reshape(self.shape + (1,))(x)

        # Multiply porous media input with the "analytical solution"
        x = tf.math.multiply(x, self.analytical)

        ###########
        # ENCODER #
        ###########

        # Skeleton of Porous Media
        x2 = skel_input
        x2 = tf.keras.layers.Reshape(self.shape + (1,))(x2)

        # For U-Net
        skips = []

        for f in self.filters:
            x = tf.keras.layers.Conv3D(
                f, (3, 3, 3), strides=(2, 2, 2), padding='same',
                activation='relu')(x)

            skips.append(x)

        for f in self.filters:
            x2 = tf.keras.layers.Conv3D(
                f, (3, 3, 3), strides=(2, 2, 2), padding='same',
                activation='relu')(x2)

        # For reshaping later
        volumeSize = tf.keras.backend.int_shape(x)

        ##############
        # BOTTLENECK #
        ##############

        # Flatten the convoluted porous media and skeleton
        x = tf.keras.layers.Flatten()(x)
        x2 = tf.keras.layers.Flatten()(x2)

        # Append both information from porous media and skeleton
        x = tf.keras.layers.Concatenate()([x, x2])

        # Reduce input space to latent dimension
        latent_inputs = tf.keras.layers.Dense(
            self.latentDim, activation='relu')(x)

        return latent_inputs, volumeSize, skips

    def _get_decoder(self, latent_inputs, volumeSize, skips):

        # Start building decoder
        x = tf.keras.layers.Dense(np.prod(volumeSize[1:]))(latent_inputs)
        x = tf.keras.layers.Reshape(
            (volumeSize[1], volumeSize[2], volumeSize[3], volumeSize[4]))(x)

        # Reverse for decoder
        for (f, skip) in zip(self.filters[::-1], skips[::-1]):
            x = tf.concat([x, skip], axis=-1)

            x = tf.keras.layers.Conv3DTranspose(
                f, (3, 3, 3), strides=(2, 2, 2), padding='same',
                activation='relu')(x)

        # Last layer - activation is None to let it do whatever
        last = tf.keras.layers.Conv3DTranspose(
            1, (3, 3, 3), padding="same")(x)
        decoder_output = tf.keras.layers.Reshape(self.shape)(last)

        return decoder_output

    def _post_process_model_output(self, output, soil_input):

        # Zero out boundaries on `last` tensor & get boundary condition tensors
        last, left_BC, right_BC = self._enforce_hard_bcs(
            output, soil_input, 1.0, 0.5)

        # Zero out regions representing solid phase
        last = tf.math.multiply(last, soil_input)

        # Add boundary conditions
        last = tf.math.add(last, left_BC)
        last = tf.math.add(last, right_BC)

        return last

    def get_model(self):

        tf.no_gradient("self.enforce_hard_bcs")

        # Porous Media Input
        soil_input = tf.keras.Input(shape=self.shape, name="soil")

        # Porous Media Skeleton Input
        skel_input = tf.keras.Input(shape=self.shape, name="skeleton")

        ##################
        # RETRIEVE U-Net #
        ##################
        latent_inputs, volumeSize, skips = self._get_encoder(
            soil_input, skel_input
        )
        decoder_outputs = self._get_decoder(
            latent_inputs, volumeSize, skips
        )

        ###################
        # MATH OPERATIONS #
        ###################
        last = self._post_process_model_output(
            decoder_outputs, soil_input
        )

        # Return model as a `tf.keras.Model` object
        model = tf.keras.Model(
            inputs=[soil_input, skel_input], outputs=last, name="soil_studies")

        return model
