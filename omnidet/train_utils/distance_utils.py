"""
Distance training utils for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import ListedColormap


def bilinear_sampler(im: torch.Tensor, flow_field: torch.Tensor,
                     mode='bilinear', padding_mode='border') -> torch.Tensor:
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel in https://arxiv.org/abs/1506.02025.
    flow_field is the tensor specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    :param im: Batch of images with shape -- [B x 3 x H x W]
    :param flow_field: Tensor of normalized x, y coordinates in [-1, 1], with shape -- [B x 2 x H * W]
    :param mode: interpolation mode to calculate output values 'bilinear' | 'nearest'.
    :param padding_mode: "zeros" use "0" for out-of-bound grid locations,
           padding_mode="border: use border values for out-of-bound grid locations,
           padding_mode="reflection": use values at locations reflected by the border
                        for out-of-bound grid locations.
    :return: Sampled image with shape -- [B x 3 x H x W]
    """
    batch_size, channels, height, width = im.shape
    flow_field = flow_field.permute(0, 2, 1).reshape(batch_size, height, width, 2)
    output = F.grid_sample(im, flow_field, mode=mode, padding_mode=padding_mode, align_corners=True)
    return output


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]"""
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higher resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


# Support only preceptually uniform sequential colormaps
# https://matplotlib.org/examples/color/colormaps_reference.html
COLORMAPS = dict(plasma=cm.get_cmap('plasma', 10000),
                 magma=high_res_colormap(cm.get_cmap('magma')),
                 viridis=cm.get_cmap('viridis', 10000))


def tensor2array(tensor, colormap='magma'):
    norm_array = normalize_image(tensor).detach().cpu()
    array = COLORMAPS[colormap](norm_array).astype(np.float32)
    return array.transpose(2, 0, 1)
