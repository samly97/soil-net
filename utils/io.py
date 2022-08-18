import numpy as np

import json
import os


def load_json(
    filename: str,
    path: str = "",
):
    if path == "":
        filepath = os.path.join(os.getcwd(), filename)
    else:
        filepath = os.path.join(path, filename)

    f = open(filepath, "r")
    return json.load(f)


def save_numpy_arr(
    img: np.ndarray,
    fname: str,
) -> None:
    np.save(fname, img)


def create_dir(
    path: str,
    dirname: str,
) -> str:
    out_dir = os.path.join(path, dirname)

    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    return out_dir
