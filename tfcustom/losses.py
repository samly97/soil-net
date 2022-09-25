import tensorflow as tf
from scipy.ndimage import binary_erosion


class LaplacianLoss(tf.keras.losses.Loss):

    def __init__(self, dim: int = 128, name="laplacian_mse"):
        super().__init__(name=name)

        self.laplacian = tf.constant([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                      [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                                     tf.float32)

        # Shape (3, 3, 3, 1, 1) so convolution would work
        self.laplacian = tf.expand_dims(self.laplacian, axis=-1)
        self.laplacian = tf.expand_dims(self.laplacian, axis=-1)

        # `unborder` has borders of 0 to zero out the border edges. The
        # laplacian convolution adds 0s to the border to convolve, which
        # incorrectly handles the finite difference method on those pixels.
        self.unborder = tf.ones((dim - 2, dim - 2, dim - 2), dtype=tf.float32)
        padding = tf.constant([[1, 1], [1, 1], [1, 1]])
        self.unborder = tf.pad(self.unborder, padding,
                               "CONSTANT", constant_values=0)
        self.unborder = tf.reshape(self.unborder, (1, dim, dim, dim, 1))

    def call(self, y_true, y_pred):

        # Dims are now (Batch, Height, Width, Depth, Channels)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        # "Recover" the porous media from concentration map
        p_media = tf.math.divide(y_true, y_true)
        p_media = tf.cast(p_media, tf.bool)

        # Compute the Laplacian and get rid of borders
        pred_lp = self._compute_laplacian(y_pred)

        # Mask internal grain boundaries
        pred_lp = self._mask_internal_grain_boundaries(pred_lp, p_media)

        # (laplacian - 0) ^ 2
        return tf.reduce_mean(tf.square(pred_lp))

    def _compute_laplacian(self, y):
        # Computes the Laplacian (note that edges are weird - laplacian on edge
        # and vals of 0)
        ret = tf.nn.conv3d(y, self.laplacian, (1, 1, 1, 1, 1), "SAME")

        # Zero out border pixels - the gradient at these values should be 0 too
        # due to chain rule.
        ret = tf.math.multiply(ret, self.unborder)

        return ret

    def _mask_internal_grain_boundaries(self, pred_lp, p_media):
        # Create a mask which effectively removes the internal boundary in the
        # void space: (solid/boundary (eroded void)/void)
        grain_boundaries = tf.numpy_function(
            binary_erosion, [p_media], tf.bool
        )

        p_media = tf.cast(p_media, tf.float32)
        grain_boundaries = tf.cast(grain_boundaries, tf.float32)
        grain_boundaries = tf.math.multiply(grain_boundaries, p_media)

        # Expand `grain_boundaries` to have the same shape as `pred_lp`, namely
        # [batch, dim, dim, dim, 1]
        grain_boundaries = tf.expand_dims(grain_boundaries, axis=-1)

        # Mask out internal grain boundaries
        pred_lp = tf.math.multiply(pred_lp, grain_boundaries)

        return pred_lp

    def get_config(self):
        cfg = super().get_config()
        return cfg


class OverallLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        lossA,
        lossB,
        W_a: float = 1.,
        W_b: float = 1.,
        name="overall_loss",
    ):
        super().__init__(name=name)

        self.weight_A = tf.cast(W_a, tf.float32)
        self.weight_B = tf.cast(W_b, tf.float32)

        self.lossA = lossA
        self.lossB = lossB

    def call(self, y_true, y_pred):
        return self.weight_A * self.lossA(y_true, y_pred) \
            + self.weight_B * self.lossB(y_true, y_pred)

    def get_config(self):
        cfg = super().get_config()
        cfg["weight_A"] = self.weight_A.numpy()
        cfg["weight_B"] = self.weight_B.numpy()
        cfg["lossA"] = tf.keras.losses.serialize(self.lossA)
        cfg["lossB"] = tf.keras.losses.serialize(self.lossB)

        return cfg
