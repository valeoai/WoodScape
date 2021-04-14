#!/usr/bin/env python3
# Copyright 2021 Valeo Schalter und Sensoren GmbH and contributors
#
# Author: Christian Witt <christian.witt@valeo.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pathlib
import os

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
from matplotlib import pyplot as plt

from projection import Camera, RadialPolyCamProjection, CylindricalProjection, read_cam_from_json, \
    create_img_projection_maps


def make_cylindrical_cam(cam: Camera):
    """generates a cylindrical camera with a centered horizon"""
    assert isinstance(cam.lens, RadialPolyCamProjection)
    # creates a cylindrical projection
    lens = CylindricalProjection(cam.lens.coefficients[0])
    rot_zxz = SciRot.from_matrix(cam.rotation).as_euler('zxz')
    # adjust all angles to multiples of 90 degree
    rot_zxz = np.round(rot_zxz / (np.pi / 2)) * (np.pi / 2)
    # center horizon
    rot_zxz[1] = np.pi / 2
    # noinspection PyArgumentList
    return Camera(
        rotation=SciRot.from_euler(angles=rot_zxz, seq='zxz').as_matrix(),
        translation=cam.translation,
        lens=lens,
        size=cam.size, principle_point=(cam.cx_offset, cam.cy_offset),
        aspect_ratio=cam.aspect_ratio
    )


root = pathlib.Path(os.path.realpath(__file__)).parent

# generate camera instances
fisheye_cam = read_cam_from_json(root / 'front.json')
cylindrical_cam = make_cylindrical_cam(fisheye_cam)

# load example image and re-project it to a central cylindrical projection
fisheye_image = cv2.imread(str(root / 'front.jpg'))
map1, map2 = create_img_projection_maps(fisheye_cam, cylindrical_cam)
cylindrical_image = cv2.remap(fisheye_image, map1, map2, cv2.INTER_CUBIC)

# draw some example boxes into both images
for i, pts in enumerate([(592, 443, 95, 75),
                         (723, 458, 112, 54),
                         (91, 428, 95, 65)]):
    roi_x, roi_y, roi_width, roi_height = pts
    cyl_points = np.array([
        (roi_x, roi_y),
        (roi_x + roi_width, roi_y),
        (roi_x + roi_width, roi_y + roi_height),
        (roi_x, roi_y + roi_height)])

    world_point = cylindrical_cam.project_2d_to_3d(cyl_points, norm=np.ones(cyl_points.shape[0]))
    fisheye_points = fisheye_cam.project_3d_to_2d(world_point)

    cv2.polylines(cylindrical_image, np.int32([cyl_points]), True, (128, 255, 64), 1)
    cv2.polylines(fisheye_image, np.int32([fisheye_points]), True, (128, 255, 64), 1)

plt.imshow(cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(cylindrical_image, cv2.COLOR_BGR2RGB))
plt.show()
