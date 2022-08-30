import tensorflow as tf
from findiff import FinDiff


class SoilNet(tf.keras.Model):

    def predict_tortuosity(
        self,
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
        A = dim ** 3

        # First derivative in the x-axis with O(h^2) accuracy
        # -- Apparently FinDiff doesn't do odd integer accuracy
        d_dx = FinDiff(0, 1, 1, acc=2)

        # Predict concentration map and estimate derivative
        conc_map = self.predict((soil, skel))
        deriv = tf.numpy_function(d_dx, [conc_map], tf.float32)

        # Gather the inlet derivatives, axis = (batch, x, y, z)
        inlet_deriv = tf.gather(deriv, 0, axis=1)

        # Reduce twice to get rid of y and z axes
        flux = tf.math.reduce_sum(inlet_deriv, axis=1)
        flux = tf.math.reduce_sum(flux, axis=1)

        # Get the effective porosities of the samples
        eps = tf.math.reduce_sum(soil, axis=1)
        eps = tf.math.reduce_sum(eps, axis=1)
        eps = tf.math.reduce_sum(eps, axis=1)
        eps = tf.math.scalar_mul(1/A, eps)

        # Compute effective diffusivity
        Deff = tf.math.scalar_mul((L-1)/A/dC, flux)

        # Compute tortuosity
        tau = tf.math.divide(eps, Deff)

        return tau

    def evaluate_abs_rel_err(
        self,
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
        predictions = self.predict((soil, skel))

        # Add arbtirary number to avoid divide by 0
        soil = tf.math.scalar_mul(5., soil)
        target = tf.math.add(soil, target)
        predictions = tf.math.add(soil, predictions)

        # Compute Relative Absolute Error (decimal form)
        relative_err = tf.math.subtract(predictions, target)
        relative_err = tf.math.abs(relative_err)
        relative_err = tf.math.divide(relative_err, target)

        return relative_err
