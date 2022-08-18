import porespy as ps
import numpy as np

import json
import os

# For skeletonizing the images
import scipy.ndimage as spim
from skimage.morphology import skeletonize_3d, ball


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


def save_micro_png(
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


def create_blob_soil(
    pixels: int = 128,
    min_porosity: float = 0.1,
    max_porosity: float = 0.9,
) -> np.ndarray:

    # Scales randomly generated porosity from: min_porosity - max_porosity
    porosity = min_porosity + np.random.rand() * max_porosity

    # Blobiness of 1-3 (high is exclusive), see documentation for reference:
    # https://porespy.org/modules/generated/porespy.generators.blobs.html
    blobiness = np.random.randint(1, high=4)

    soil = ps.generators.blobs(
        shape=[pixels, pixels, pixels], porosity=porosity, blobiness=blobiness,
    )

    return soil


def create_overlapping_sphere_soil(
    pixels: int = 128,
    r: int = 5,
    min_porosity: float = 0.1,
    max_porosity: float = 0.9,
    max_iters=15,
) -> np.ndarray:

    # Scales randomly generated porosity from: min_porosity - max_porosity
    porosity = min_porosity + np.random.rand() * max_porosity

    soil = ps.generators.overlapping_spheres(
        shape=[pixels, pixels, pixels], r=r, porosity=porosity,
        maxiter=max_iters,
    )

    return soil


def skeletonize_porous_media(
    im: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    if len(im.shape) < 3:
        return ValueError("Image needs to be 3-dimensional!")

    # Work off of a copy to be safe
    im = np.copy(im)

    # Change swap axis to skeletonize in the desired direction
    im = np.swapaxes(im, 0, axis)

    # Default is the 0 axis
    pw = [[20, 20], [0, 0], [0, 0]]
    strel = ball

    # Extend pore channels in the z-axis
    temp = np.pad(im, pw, mode="edge")
    temp = np.pad(temp, pw, mode="constant", constant_values=True)

    # Create the skeleton
    sk = skeletonize_3d(temp) > 0

    # Remove the previously applied padding
    sk = ps.tools.unpad(sk, pw)
    sk = ps.tools.unpad(sk, pw)

    # Createa a fatter skeleton
    sk2 = spim.binary_dilation(sk, structure=strel(1))

    # Get rid of non-percolating void space

    # Inlets = outlets in the axis of interest (z here)
    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    outlets = np.zeros_like(im)
    outlets[-1, ...] = True

    # Remove pores that are not connected to inlets/outlets
    sk2 = ps.filters.trim_nonpercolating_paths(sk2, inlets, outlets)

    return sk2


if __name__ == '__main__':

    # Load config file

    """
    Algorithm:

    1. Define N_os (overlapping spheres) and N_b (blobs) to create and start
       generating porous media.
    2a. Conduct tortuosity simulations using `tortuosity_fd` and save the input
        -target pairs.
    2b. Concurrently, write the resulting metadata into something, saving the
        porosity, tortuosity, formation factors.
    """

    path = os.getcwd()

    def get_fname(idx): return str(idx) + ".npy"

    soil_dir = "soil"
    skeleton_dir = "skeleton"
    target_dir = "target"

    N_os = 150
    N_b = 150

    ret_dict = {}

    data_i = 0

    for i in range(0, N_os):
        # Dummy text message

        data_i += 1

    for i in range(0, N_b):
        data_i += 1
        print(data_i)
