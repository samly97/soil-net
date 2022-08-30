import tensorflow as tf


class SoilNet(tf.keras.Model):

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
