# !/usr/bin/env python3
"""
Creates dataset path for training and validation on WoodScape dataset.

# usage: datapath_create [--data_path]
# ./datapath_create.py -d path/to/WoodScape/WoodScape_ICCV19/rgb_images/

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import argparse
import glob
import random


def woodscape_paths_to_txt(root_path):
    dir_images = sorted(list(glob.glob(root_path + "*.png")))
    paths = [paths.replace(root_path, '') for paths in dir_images]

    train_file = "train.txt"
    val_file = "val.txt"

    tf_total = len(paths)
    percent = lambda part, whole: float(whole) / 100 * float(part)
    test_count = percent(2.5, tf_total)  # 2.5% of training data is allocated for validation as there is less data :D

    random.seed(777)
    test_frames = random.sample(paths, int(test_count))
    frames = set(paths) - set(test_frames)

    print(f'=> Total number of training frames: {len(frames)} and validation frames: {len(test_frames)}')

    with open(train_file, 'w') as tf:
        for image_path in sorted(frames):
            tf.write(image_path + "\n")

    with open(val_file, 'w') as vf:
        for image_path in sorted(test_frames):
            vf.write(image_path + "\n")

    print(f'Wrote {tf.name} and {vf.name} into disk')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a file with dataset path.')
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help='add woodscape dataset "rgb_images" folder location')
    args = parser.parse_args()
    woodscape_paths_to_txt(args.data_path)
