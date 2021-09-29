# !/usr/bin/env python3
"""
Generate look up tables to cache during distance estimation training.

# usage: ./generate_luts.py --config params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import glob
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from main import collect_tupperware


def get_intrinsics(args):
    jsons = sorted(list(glob.glob(os.path.join(args.dataset_dir, "calibration_data", "calibration", "*.json"))))

    cam_intrinsics = list()
    indicies = list()
    camera_sides = list()

    for idx, file in enumerate(jsons):
        data = json.load(open(file))
        list_of_dict_values = list(data['intrinsic'].values())
        if list_of_dict_values in cam_intrinsics:
            continue
        cam_intrinsics.append(list_of_dict_values)
        indicies.append(idx)
        camera_sides.append(data['name'])

    # add file indexes & camera side to the beginning & end of the list
    for idx, (index_values, cam_side) in enumerate(zip(indicies, camera_sides)):
        cam_intrinsics[idx].insert(0, index_values)
        cam_intrinsics[idx].insert(len(cam_intrinsics), cam_side)

    return cam_intrinsics


def scale_intrinsic(args, intrinsic, crop_coords):
    """Scales the intrinsics from original res to the network's initial input res"""
    K = np.copy(intrinsic[1:4])
    D = np.copy(intrinsic[5:9])
    # Compensate for resizing
    if args.crop:
        D *= args.input_width / (crop_coords[2] - crop_coords[0])
        K[1] *= args.input_width / (crop_coords[2] - crop_coords[0])
        K[2] *= args.input_height / (crop_coords[3] - crop_coords[1])
    else:
        D *= args.input_width / intrinsic[-4]
        K[1] *= args.input_width / intrinsic[-4]
        K[2] *= args.input_height / intrinsic[-2]
    return K, D


def inverse_poly_lut(args, intrinsics, cropped_coords):
    """Create LUTs for the polynomial projection model as there is no analytical inverse"""
    LUTs = dict()

    total_car1_images = 6054

    for intrinsic in intrinsics:
        intrinsic[2] += intrinsic[-2] / 2
        intrinsic[3] += intrinsic[4] / 2

        if args.crop:
            # Adjust the offset of the cropped intrinsic around the width and height.
            if intrinsic[0] < total_car1_images:
                coords = cropped_coords["Car1"][intrinsic[-1]]
            else:
                coords = cropped_coords["Car2"][intrinsic[-1]]
            intrinsic[2] -= coords[0]
            intrinsic[3] -= coords[1]

        # Scale intrinsics to adjust image resize.
        K, D = scale_intrinsic(args, intrinsic, coords)

        x = np.linspace(0, args.input_width - 1, args.input_width)
        y = np.linspace(0, args.input_height - 1, args.input_height)
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)

        x_cam = (mesh_x - K[1]) / K[0]
        y_cam = (mesh_y - K[2]) / K[0]

        r = np.sqrt(x_cam * x_cam + y_cam * y_cam)
        theta_LUT = np.arctan2(y_cam, x_cam).astype(np.float32)
        angle_LUT = np.zeros_like(r, dtype=np.float32)

        for i, _r in tqdm(enumerate(r)):
            a = np.roots([D[3], D[2], D[1], D[0], -_r])
            a = np.real(a[a.imag == 0])
            try:
                a = np.min(a[a >= 0])
                angle_LUT[i] = a
            except ValueError:  # raised if `a` is empty.
                print(f"Field angle of incident ray is empty")
                pass

        LUTs[intrinsic[5]] = dict(theta=theta_LUT, angle_maps=angle_LUT)

    return LUTs


if __name__ == "__main__":
    args = collect_tupperware()
    camera_intrinsics = get_intrinsics(args)

    # Cropping to remove the black regions of the car (impacts the distance estimation task)
    # Solution would be to use masks and ignore them during the loss calculation
    cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                    MVL=(343, 5, 1088, 411),
                                    MVR=(185, 5, 915, 425),
                                    RV=(186, 203, 1105, 630)),
                          Car2=dict(FV=(160, 272, 1030, 677),
                                    MVL=(327, 7, 1096, 410),
                                    MVR=(175, 4, 935, 404),
                                    RV=(285, 187, 1000, 572)))

    LUTs = inverse_poly_lut(args, camera_intrinsics, cropped_coords)

    with open('LUTs.pkl', 'wb') as f:
        pickle.dump(LUTs, f, pickle.HIGHEST_PROTOCOL)
