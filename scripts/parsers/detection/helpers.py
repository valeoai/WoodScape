#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for box2d extraction from polygons

# author: Ganesh Sistu
# reviewers:

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""
import csv

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_2d_boxes(parser, class_names, class_colors, class_ids, class_obj_thresh,
                      occluded_ann=True, image_rgb=None, debug=False):
    # generate 2d boxes
    if image_rgb is None and debug:
        raise Exception('Image is None!')

    skip_count = 0
    class_stats = [0 for _ in class_ids]
    class_dict = dict(zip(class_names, class_ids))
    img_box_annotations = list()
    for polygon_object in parser.get_objects():
        # object class name
        class_name = polygon_object.get_class_name()
        class_id = class_dict[class_name.lower()]
        obj_thresh_pixels = class_obj_thresh
        # Bounding box
        b_box = polygon_object.get_box()

        # ignore objects of size threshold pixels
        occlusion_level = polygon_object.get_metadata().get_occlusion()
        depicted_object = polygon_object.get_metadata().is_depiction()
        covered_by_glass = polygon_object.get_metadata().is_cover_by_glass()

        if depicted_object or covered_by_glass:
            skip_count += 1
            continue
        obj_w = b_box['x_max'] - b_box['x_min']
        obj_h = b_box['y_max'] - b_box['y_min']
        area = obj_w * obj_h
        if (area < obj_thresh_pixels) or (obj_w < 0.1 * obj_h) or (obj_h < 0.1 * obj_w):
            skip_count += 1
            continue
        # occlusion_levels --> {0: 0%, 1: (1-25)%, 2: (26-50)%, 3: (51-75)%, 4: (76-99)%}
        if occluded_ann and occlusion_level > 1:
            skip_count += 1
            continue

        boundary = polygon_object.get_poly_cods()
        boundary_numpy = np.array(boundary, dtype=np.int32)
        # valid polygon should have a 2 dims
        if not len(boundary_numpy.shape) > 1:
            continue
        # uniform_points = boundary_numpy
        min_x = np.min(boundary_numpy[:, 0])
        max_x = np.max(boundary_numpy[:, 0])
        min_y = np.min(boundary_numpy[:, 1])
        max_y = np.max(boundary_numpy[:, 1])
        b_box['x_min'] = min_x
        b_box['x_max'] = max_x
        b_box['y_min'] = min_y
        b_box['y_max'] = max_y

        if debug:
            # for xi, yi in zip(boundary_numpy[:, 0], boundary_numpy[:, 1]):
            #     point = (xi, yi)
            #     cv2.circle(image_rgb, (int(point[0]), int(
            #         point[1])), 1, (0, 255, 0), -1)
            cv2.rectangle(image_rgb, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)
            x_center = (b_box['x_min'] + b_box['x_max']) / 2
            y_center = (b_box['y_min'] + b_box['y_max']) / 2
            cv2.putText(image_rgb, str(class_id) + ': ' + class_name, (int(x_center), int(y_center)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, class_colors[class_id], 2)

        obj_poly_annotation = [class_name, class_id, int(b_box['x_min']), int(b_box['y_min']),
                               int(b_box['x_max']), int(b_box['y_max'])]

        img_box_annotations.append(obj_poly_annotation)
        class_stats[class_id] += 1

    if debug:
        print('skipped objects due to filtering: ', skip_count)
        plt.close('all')
        plt.imshow(image_rgb)
        plt.show()

    return img_box_annotations, class_stats


def read_image_as_rgb(file_name):
    """
    read image from file name as RGB
    """
    return np.asarray(Image.open(file_name))


def write_to_file(file_name, data):
    """
    write list to a csv
    """
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
