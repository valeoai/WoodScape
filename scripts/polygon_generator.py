#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate polygon points from polygon annotations

# usage: box2d_generator [--src_path] [--dst_path] [--box_2d_class_mapping] [--instance_class_mapping] [--rgb_image_path]
# arguments:
--src_path : instance annotations directory ex: .\\woodscape\\instance_annotations
--dst_path : semantic annotations directory ex: .\\woodscape\\box_2d_annotations
--box_2d_class_mapping : modify the mapping file in .\\woodscape\\scripts\\configs\\box_2d_mapping_x_classes.json
--instance_class_mapping : in .\\woodscape\\scripts\\mappers\\class_names.json
--rgb_image_path : rgb image directory ex: .\\woodscape\\rgb_images

# author: Ganesh Sistu
# reviewers:

#Values    Name      Description
----------------------------------------------------------------------------
   1    name         Describes the object by name from config file:
                     Ex: 'Car', 'Pedestrian', 'Cyclist'
   1    index        Describes the type of object by index from config file:
                     Ex: 'Car' - 0, 'Pedestrian' - 1, 'Cyclist' -2
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel
   N    polygon      Polygon (x,y) co-ordinates 

Example: output.txt
car 0 10, 20, 10, 40, 0,5, 2, 11,..... 
rider 1 30, 40, 10, 40, 0,5, 2, 11,..... 
.... ... .. .. .. ..
.... ... .. .. .. ..

Note: To generate boxes with more classes modify
.\\woodscape\\scripts\\parsers\\detection\\class_names.py

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""
import csv
import os
import argparse
import json

import numpy as np
from tqdm import tqdm
from ann_utils import create_dir, absolute_file_paths
from parsers.detection.filter_params import FilterParams
from parsers.detection.helpers import read_image_as_rgb, generate_polygons, write_to_file
from parsers.detection.annotation_detection_parser import AnnotationDetectionParser


def parser_arguments():
    # arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--src_path", type=str,
                            help="instance annotations directory")
    arg_parser.add_argument("--dst_path", type=str,
                            help="box_2d annotations directory")
    arg_parser.add_argument("--box_2d_class_mapping", type=str,
                            help="box_2d_mapping_x_classes.json")
    arg_parser.add_argument("--instance_class_mapping", type=str,
                            help="class_names.json")
    arg_parser.add_argument("--rgb_image_path", type=str, nargs="?",
                            help="rgb_image directory")

    args = arg_parser.parse_args()
    # create destination dir
    create_dir(args.dst_path)

    if os.path.isdir(args.src_path) and \
            os.path.isfile(args.instance_class_mapping) and \
            os.path.isfile(args.box_2d_class_mapping):

        instance_classes_info = open(args.instance_class_mapping).read()
        instance_classes_mapping = json.loads(instance_classes_info)

        box_2d_classes_info = open(args.box_2d_class_mapping).read()
        box_2d_classes_mapping = json.loads(box_2d_classes_info)

        return args.src_path, args.dst_path, box_2d_classes_mapping, \
            instance_classes_mapping, args.rgb_image_path
    else:
        raise Exception("Error: Check if the files or dirs in arguments exit!")


if __name__ == "__main__":

    # parser arguments
    src_path, dst_path, box_2d_info, instance_classes_mapping, rgb_image_path = parser_arguments()

    class_names = box_2d_info["classes_to_extract"]
    class_colors = box_2d_info["class_colors"]
    class_ids = box_2d_info["class_indexes"]
    class_obj_thresh = box_2d_info["min_obj_size"]
    depiction_ann = box_2d_info["filter_depict_objects"]
    glass_cover_ann = box_2d_info["filter_glass_cover_objects"]
    # by default filters objects with occlusion more than 25%
    occluded_ann = box_2d_info["filter_occluded_objects"]
    debug = box_2d_info["debug"]

    # Filter configuration
    # filter annotations that are depicted or on glass
    filter_ann_params = FilterParams(class_names=class_names,
                                     depiction=depiction_ann,
                                     cover_by_glass=glass_cover_ann)

    # Get the list of json files in the dir
    trace_list = [file_name for file_name in absolute_file_paths(
        src_path) if '.json' in file_name]

    # generate box annotations from polygon annotations
    stats_overall = list()
    for source_file_name in tqdm(trace_list):
        file_name = os.path.basename(source_file_name)
        text_file_name = file_name.replace('.json', '.txt')
        dst_file_name = os.path.join(dst_path, text_file_name)
        png_image_name = file_name.replace('.json', '.png')
        rgb_image_file_name = os.path.join(rgb_image_path, png_image_name)
        image = read_image_as_rgb(rgb_image_file_name)
        shape = image.shape
        if debug:
            image_debug = image.copy()
        else:
            image_debug = None
        json_ann_name = str(source_file_name)
        parser = AnnotationDetectionParser(json_ann_name,
                                           (shape[1], shape[0]),
                                           filter_ann_params)
        img_poly_annotations, stats = generate_polygons(parser, class_names, class_colors, class_ids, class_obj_thresh,
                                                        image_rgb=image_debug, debug=debug)
        write_to_file(dst_file_name, img_poly_annotations)
        stats_overall.append(stats)

    stats_overall = np.sum(np.array(stats_overall), axis=0)

    print('\n ********* Stats *********')
    for class_name, class_stats in zip(class_names, stats_overall):
        print(class_name, ':', np.sum(class_stats))
    print('Done!')
