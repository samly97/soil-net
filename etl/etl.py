from typing import List, Tuple
import numpy as np

import tensorflow as tf


class ETL():

    def __init__(
        self,
        criteria_arr: List[int],
        batch_size: int,
        data_dir: str,
        soil_dir: str,
        skel_dir: str,
        target_dir: str,
    ):
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.batch_size = batch_size

        self.data_dir = data_dir
        self.soil_dir = soil_dir
        self.skel_dir = skel_dir
        self.target_dir = target_dir

        self.starting_ds = tf.data.Dataset.from_tensor_slices(
            criteria_arr
        )

    def get_ml_dataset(self):
        def configure_for_performance(ds):
            # https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
            ds = ds.cache()
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(buffer_size=self.AUTOTUNE)
            return ds

        inp_ds = self.starting_ds.map(
            self._process_input_path, num_parallel_calls=self.AUTOTUNE)
        out_ds = self.starting_ds.map(
            self._process_output_path, num_parallel_calls=self.AUTOTUNE)

        inp_ds = configure_for_performance(inp_ds)
        out_ds = configure_for_performance(out_ds)

        ret_ds = tf.data.Dataset.zip((inp_ds, out_ds))
        return ret_ds

    def _process_input_path(
        self,
        arr_idx
    ) -> Tuple[tf.types.experimental.TensorLike,
               tf.types.experimental.TensorLike]:
        # Load from porous media and skeleton from .npy
        soil = self.load_numpy_arr(
            arr_idx,
            self.soil_dir,
            tf.bool,
            tf.float32
        )
        skel = self.load_numpy_arr(
            arr_idx,
            self.skel_dir,
            tf.bool,
            tf.float32
        )
        return soil, skel

    def _process_output_path(
        self,
        arr_idx
    ) -> tf.types.experimental.TensorLike:
        conc_map = self.load_numpy_arr(
            arr_idx,
            self.target_dir,
            tf.float64,
            tf.float32,
        )
        return conc_map

    def load_numpy_arr(
        self,
        arr_idx: int,
        from_dir: str,
        load_dtype,
        save_dtype,
    ) -> tf.types.experimental.TensorLike:
        path = tf.strings.join(
            [self.data_dir, "/", from_dir]
        )
        fname = tf.strings.join(
            [path, "/", tf.strings.as_string(arr_idx), ".npy"])

        img = tf.numpy_function(
            np.load, [fname], load_dtype,
        )
        img = tf.cast(img, dtype=save_dtype)

        return img
