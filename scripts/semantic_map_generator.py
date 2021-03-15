#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate semantic maps from polygon annotations

# usage: semantic_map_generator [--src_path] [--dst_path] [--class_mapping]
# arguments:
--src_path : instance annotations directory ex: .\\woodscape\\instance_annotations
--dst_path : semantic annotations directory ex: .\\woodscape\\semantic_annotations
--class_mapping : modify the mapping file in .\\woodscape\\scripts\\configs\\semantic_mapping_x_classes.json
--instance_class_mapping : in .\\woodscape\\scripts\\mappers\\class_names.json

# author: Ganesh Sistu
# reviewers:

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""

import os
import argparse
import json
from tqdm import tqdm

from parsers.segmentation.image_annotator import ImageAnnotator, parse_annotation
from ann_utils import create_dir, absolute_file_paths


def parser_arguments():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str,
                        help="instance annotations directory")
    parser.add_argument("--dst_path", type=str,
                        help="semantic annotations directory")
    parser.add_argument("--semantic_class_mapping", type=str,
                        help="semantic_mapping_x_classes.json")
    parser.add_argument("--instance_class_mapping", type=str,
                        help="woodscape_instance_classes.json")

    args = parser.parse_args()
    # create destination dir
    create_dir(args.dst_path)

    if os.path.isdir(args.src_path) and \
            os.path.isfile(args.instance_class_mapping) and \
            os.path.isfile(args.semantic_class_mapping):

        instance_classes_info = open(args.instance_class_mapping).read()
        instance_classes_mapping = json.loads(instance_classes_info)

        semantic_classes_info = open(args.semantic_class_mapping).read()
        semantic_classes_mapping = json.loads(semantic_classes_info)

        return args.src_path, args.dst_path, semantic_classes_mapping, instance_classes_mapping
    else:
        raise Exception("Error: Check if the files or dirs in arguments exit!")


if __name__ == "__main__":

    # parser arguments
    src_path, dst_path, semantic_classes_mapping, instance_classes_mapping = parser_arguments()

    semantic_class_names = list(semantic_classes_mapping['classes_to_extract'])
    semantic_class_indexes = dict(zip(semantic_class_names, semantic_classes_mapping['class_indexes']))
    semantic_class_colors = dict(zip(semantic_class_names, semantic_classes_mapping['class_colors']))

    # Get the list of json files in the dir
    trace_list = [file_name for file_name in absolute_file_paths(
        src_path) if '.json' in file_name]

    # Iterate over all json files to generate semantic maps
    paths_dict = {"save_dir_seg": dst_path}

    for source_file_name in tqdm(trace_list):
        # for each annotation file
        file_name = os.path.basename(source_file_name)
        dst_file_name = os.path.join(dst_path, file_name)

        json_data = open(source_file_name).read()
        data = json.loads(json_data)

        for k, v in data.items():
            annotation_info = parse_annotation(data[k]["annotation"])
            instance_annotation = ImageAnnotator(file_name, annotation_info, semantic_class_names,
                                                 semantic_class_indexes, semantic_class_colors)

            instance_annotation.save_semantic(instance_classes_mapping,
                                              dest_file_name=file_name,
                                              paths_dict=paths_dict,
                                              w=v["image_width"],
                                              h=v["image_height"])

print('Done!')
