import tensorflow as tf


class LaplacianLoss(tf.keras.losses.Loss):

    def __init__(self, name="laplacian_mse"):
        super().__init__(name=name)

        self.laplacian = tf.constant([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                      [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                                     tf.float32)

        # Shape (3, 3, 3, 1, 1) so convolution would work
        self.laplacian = tf.expand_dims(self.laplacian, axis=-1)
        self.laplacian = tf.expand_dims(self.laplacian, axis=-1)

    def call(self, y_true, y_pred):
        _ = y_true

        # Dims are now (Batch, Height, Width, Depth, Channels)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        # Computes the Laplacian (note that edges are weird - laplacian on edge
        # and vals of 0)
        ret = tf.nn.conv3d(y_pred, self.laplacian, (1, 1, 1, 1, 1), "SAME")

        # (laplacian - 0) ^ 2
        return tf.reduce_mean(tf.square(ret))

    def get_config(self):
        return {
            "laplacian": self.laplacian,
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
