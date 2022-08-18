import os
from typing import List, Tuple

import numpy as np
from random import shuffle
from math import ceil
import tensorflow as tf

from etl.etl import ETL


def parse_raw_data(
    data_dir: str,
    input_dir: str,
) -> List[int]:

    data_dir = os.path.join(os.getcwd(), data_dir)
    input_im_dir = os.path.join(data_dir, input_dir)
    input_im_fname = np.array([fname for fname in os.listdir(input_im_dir)])

    pic_num = [int(fname.split(".npy")[0]) for fname in input_im_fname]

    return pic_num


def shuffle_dataset(
    pic_num: List[int],
) -> List[int]:

    # Shuffle list
    list_idx = list(range(len(pic_num)))
    shuffle(list_idx)

    # Shuffle particle number list
    pic_num_arr = np.array(pic_num)
    pic_num_arr = pic_num_arr[list_idx]

    return pic_num_arr.tolist()


def get_split_indices(
    trn_split: float,
    pic_num: List[int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:

    trn_idx = ceil(len(pic_num) * trn_split)
    val_idx = len(pic_num)

    return ((0, trn_idx), (trn_idx, val_idx))


def create_etl() -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    pic_num = parse_raw_data("dataset", "soil")  # cfg file?
    pic_num = shuffle_dataset(pic_num)

    trn_idx, val_idx = get_split_indices(
        0.8,  # cfg file?
        pic_num,
    )

    datasets = [None, None]

    for idx, tup in enumerate([trn_idx, val_idx]):
        start_idx, end_idx = tup
        criteria_arr = pic_num[start_idx:end_idx]

        etl = ETL(
            criteria_arr,
            batch_size=8,  # cfg file?
            data_dir=os.path.join(os.getcwd(), "dataset"),  # cfg file?
            soil_dir="soil",  # cfg file?
            skel_dir="skeleton",  # cfg file?
            target_dir="target",  # cfg file?
        )
        datasets[idx] = etl.get_ml_dataset()

    trn_dataset = datasets[0]
    val_dataset = datasets[1]

    return (
        trn_dataset,
        val_dataset,
    )


if __name__ == "__main__":
    create_etl()
