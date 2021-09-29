#!/usr/bin/env python3
"""
Semantic class weights calculation for OmniDet

# usage: ./compute_class_weights.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import json
import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.woodscape_loader import WoodScapeRawDataset
from main import collect_tupperware

printj = lambda dic: print(json.dumps(dic, indent=4))


def main():
    args = collect_tupperware()
    printj(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or -1

    print(f"=> Loading {args.dataset.upper()} training dataset")

    # --- Load Data ---
    train_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                        path_file=args.train_file,
                                        is_train=False,
                                        config=args)

    train_loader = DataLoader(train_dataset,
                              args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    # Get class weights from the selected weighing technique
    print(f"=> Weighing technique: {args.weighing}  \n"
          f"Computing class weights... \n"
          f"This can take a while depending on the dataset size")

    if args.weighing.lower() == 'enet':
        class_weights = enet_weighing(train_loader, args.num_classes)
    elif args.weighing.lower() == 'mfb':
        class_weights = median_freq_balancing(train_loader, args.num_classes)
    else:
        class_weights = None

    with np.printoptions(precision=2, suppress=True):
        print(f"Class weights: {class_weights}")


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper: w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class: propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    :param dataloader: A data loader to iterate over the dataset.
    :param num_classes: The number of classes.
    :param c: An additional hyper-parameter which restricts the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for inputs in tqdm(dataloader):
        flat_label = inputs["motion_labels", 0, 0].cpu().numpy().flatten()

        # Sum up the number of pixels of each class and the total pixel counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described in https://arxiv.org/abs/1411.4734:
        w_class = median_freq / freq_class,
    where freq_class is the number of pixels of a given class divided by the total number of pixels in images where
    that class is present, and median_freq is the median of freq_class.
    :param dataloader: A data loader to iterate over the dataset whose weights are going to be
    computed.
    :param num_classes: The number of classes
    """
    class_count = 0
    total = 0
    for inputs in tqdm(dataloader):
        flat_label = inputs["motion_labels", 0, 0].cpu().numpy().flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has one element for each class.
        # The value is either 0 (if the class does not exist in the label)
        # or equal to the pixel count (if the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq


if __name__ == "__main__":
    main()
