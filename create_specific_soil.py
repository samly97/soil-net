from utils import io
from datagen import create_soil

import porespy as ps

from tqdm import tqdm
import os
import json


def create_specific_soil(
    soil_fn,
    samples: int,
    path: str,
    dataset_dir: str,
):
    # Naming scheme for data: 1.npy
    def get_fname(idx): return str(idx) + ".npy"

    # Data directory names
    soil_dir = "soil"
    skeleton_dir = "skeleton"
    target_dir = "target"

    # Create directories
    data_path = io.create_dir(path, dataset_dir)

    soil_dir = io.create_dir(data_path, soil_dir)
    skeleton_dir = io.create_dir(data_path, skeleton_dir)
    target_dir = io.create_dir(data_path, target_dir)

    # Write metadata to this dictionary
    ret_dict = {}

    data_i = 0

    for i in tqdm(range(0, samples)):
        data_i += 1

        soil = soil_fn()

        # If removing non-percolating pores throws an error, discard
        # the generated porous media and continue
        try:
            soil = create_soil.clean_soil(soil)
        except Exception:
            data_i -= 1
            continue

        # Skeleton of porous media (Input data)
        skel = create_soil.skeletonize_soil(soil, axis=0)

        # May throw exception if inlet and outlet fluxes do not match
        try:
            # Generate ground-truth data
            tort_sim = ps.simulations.tortuosity_fd(soil, axis=0)
        except Exception:
            data_i -= 1
            continue

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
            "effective_porosity": tort_sim.effective_porosity,
            "tortuosity": tort_sim.tortuosity,
            "formation_factor": tort_sim.formation_factor,
        }

    with open(
        os.path.join(data_path, "dataset.json"),
        "w",
    ) as outfile:
        json.dump(ret_dict, outfile, indent=4)


if __name__ == '__main__':
    create_specific_soil()
