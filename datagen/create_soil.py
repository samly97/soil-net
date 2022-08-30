import porespy as ps
import numpy as np

# For skeletonizing the images
import scipy.ndimage as spim
from skimage.morphology import skeletonize_3d, ball


def clean_soil(
    porous_media: np.ndarray,
    axis: int = 0,
) -> np.ndarray:

    # Get inlets and outlets of image
    inlets = ps.generators.faces(porous_media.shape, inlet=axis)
    outlets = ps.generators.faces(porous_media.shape, outlet=axis)

    # Get rid of blind pores
    porous_media = ps.filters.trim_nonpercolating_paths(
        porous_media, inlets=inlets, outlets=outlets)

    return porous_media


def create_blob_soil(
    pixels: int = 128,
    min_porosity: float = 0.1,
    max_porosity: float = 0.8,
) -> np.ndarray:

    # Scales randomly generated porosity from: min_porosity - max_porosity
    porosity = min_porosity + np.random.rand() * (max_porosity - min_porosity)

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
    max_porosity: float = 0.8,
    max_iters=15,
) -> np.ndarray:

    # Scales randomly generated porosity from: min_porosity - max_porosity
    porosity = min_porosity + np.random.rand() * (max_porosity - min_porosity)

    soil = ps.generators.overlapping_spheres(
        shape=[pixels, pixels, pixels], r=r, porosity=porosity,
        maxiter=max_iters,
    )

    return soil


def create_fibrous_soil(
    pixels: int = 128,
    r: int = 5,
    phi_max: int = 0,
    theta_max: int = 90,
    min_porosity: float = 0.1,
    max_porosity: float = 0.8,
    max_iters=15,
) -> np.ndarray:

    # Scales randomly generated porosity from: min_porosity - max_porosity
    porosity = min_porosity + np.random.rand() * (max_porosity - min_porosity)

    soil = ps.generators.cylinders(
        shape=[pixels, pixels, pixels],
        r=r, phi_max=phi_max, theta_max=theta_max,
        porosity=porosity,
        maxiter=max_iters,
    )

    return soil


def skeletonize_soil(
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
