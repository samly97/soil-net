import tensorflow as tf
from .common import get_masked_conc_map


class MaskedMSEMetric(tf.keras.metrics.Metric):

    def __init__(self, name="masked_mse"):
        super().__init__(name=name)
        self.masked_mse = self.add_weight(name="mmse", initializer="zeros")

        # Used to keep a running average
        self.n_epoch = self.add_weight(
            name="internal_var", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        # This equals (batch_size) * x * y * z elements after masking
        n_batch = tf.size(y_true)
        n_batch = tf.cast(n_batch, tf.float32)

        # Batch averaged masked MSE
        batch_avg = get_masked_conc_map(y_true, y_pred)

        # Use these two variables to keep a running average over the epoch.
        # When we want to output results, we simply divide the two state
        # variables.
        self.n_epoch.assign_add(n_batch)
        self.masked_mse.assign_add(
            n_batch * batch_avg
        )

    def result(self):
        return self.masked_mse / self.n_epoch

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.masked_mse.assign(0.0)
        self.n_epoch.assign(0.0)
