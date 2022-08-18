from utils import io
from datagen import create_soil

import porespy as ps

from tqdm import tqdm
import os
import json


if __name__ == '__main__':
    """
    Algorithm:

    1. Define N_os (overlapping spheres) and N_b (blobs) to create and start
       generating porous media.
    2a. Conduct tortuosity simulations using `tortuosity_fd` and save the input
        -target pairs.
    2b. Concurrently, write the resulting metadata into something, saving the
        porosity, tortuosity, formation factors.
    """

    # Naming scheme for data: 1.npy
    def get_fname(idx): return str(idx) + ".npy"

    # Data directory names
    path = os.getcwd()

    soil_dir = "soil"
    skeleton_dir = "skeleton"
    target_dir = "target"

    # Create directories
    data_path = io.create_dir(path, "dataset")

    soil_dir = io.create_dir(data_path, soil_dir)
    skeleton_dir = io.create_dir(data_path, skeleton_dir)
    target_dir = io.create_dir(data_path, target_dir)

    N_os = 5
    N_b = 5

    ret_dict = {}

    data_i = 0

    for i in tqdm(range(0, N_os + N_b)):
        data_i += 1

        # Porous media (Input data)
        if data_i <= N_os:
            soil = create_soil.create_overlapping_sphere_soil()
            media_type = "overlapping spheres"
        else:
            soil = create_soil.create_blob_soil()
            media_type = "blobs"

        # If removing non-percolating pores throws an error, discard
        # the generated porous media and continue
        try:
            soil = create_soil.clean_soil(soil)
        except Exception:
            data_i -= 1
            continue

        # Skeleton of porous media (Input data)
        skel = create_soil.skeletonize_soil(soil, axis=0)

        # Generate ground-truth data
        tort_sim = ps.simulations.tortuosity_fd(soil, axis=0)

        # Save INPUT - TARGET Data
        io.save_numpy_arr(
            soil,
            os.path.join(soil_dir, get_fname(data_i))
        )
        io.save_numpy_arr(
            skel,
            os.path.join(skeleton_dir, get_fname(data_i))
        )
        io.save_numpy_arr(
            tort_sim.concentration,
            os.path.join(target_dir, get_fname(data_i))
        )

        # Write metadata to dictionary
        ret_dict[get_fname(data_i)] = {
            "media": media_type,
            "effective_porosity": tort_sim.effective_porosity,
            "tortuosity": tort_sim.tortuosity,
            "formation_factor": tort_sim.formation_factor,
        }

    with open(
        os.path.join(data_path, "dataset.json"),
        "w",
    ) as outfile:
        json.dump(ret_dict, outfile, indent=4)
