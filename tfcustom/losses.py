import tensorflow as tf


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
        _ = y_true

        # Dims are now (Batch, Height, Width, Depth, Channels)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        shape = tf.shape(y_pred)
        batch_size = tf.cast(shape[0], tf.int32)
        batch_size = tf.concat([[batch_size], [1], [1], [1], [1]], 0)

        # Computes the Laplacian (note that edges are weird - laplacian on edge
        # and vals of 0)
        ret = tf.nn.conv3d(y_pred, self.laplacian, (1, 1, 1, 1, 1), "SAME")

        # Tile `unborder` `batch_size` times
        temp = tf.tile(self.unborder, batch_size)

        # Zero out border pixels - the gradient at these values should be 0 too
        # due to chain rule.
        ret = tf.math.multiply(ret, temp)

        # (laplacian - 0) ^ 2
        return tf.reduce_mean(tf.square(ret))

    def get_config(self):
        return {
            "laplacian": self.laplacian,
            "unborder": self.unborder,
        }


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
        return {
            "weight_A": self.weight_A,
            "weight_B": self.weight_B,
            "lossA": self.lossA,
            "lossB": self.lossB,
        }
