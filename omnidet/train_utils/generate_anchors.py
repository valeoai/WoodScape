"""
Object Detection Anchor Box Generation Script

# author: Ganesh Sistu <ganesh.sistu@valeo.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import argparse
import glob
import json
import os

import numpy as np


def get_objects_width_and_height(annot_dir):
    """Create two lists for all objects width and height in dataset
    :param annot_dir: directory for bounding boxes annotation files
    :return: objects_width_list: list of all objects widths
    :return: objects_height_list: list of all objects heights
    """

    # List the annotation files
    annot_files_list = glob.glob(annot_dir + "/*.txt")

    # Initiate output lists
    objects_width_list = list()
    objects_height_list = list()

    # Read the annotation files
    for file_counter, annot_file in enumerate(annot_files_list):
        with open(annot_file, 'r') as annotation_file:
            annotation = annotation_file.readlines()

            for annot_object in annotation:
                tokens = annot_object.split(",")
                x1 = int(tokens[2])
                y1 = int(tokens[3])
                x2 = int(tokens[4])
                y2 = int(tokens[5].strip("\n"))

                width = x2 - x1
                height = y2 - y1

                objects_width_list.append(width)
                objects_height_list.append(height)

        print("Objects of file", file_counter, "are read successfully!")
    return objects_width_list, objects_height_list


def generate_anchors(results_dir, run_num, anchor_num, objects_w, objects_h):
    """Generate anchor boxes from object properties
    :param results_dir: output dir
    :param run_num: Number of k-means runs
    :param anchor_num: Number of anchors
    :return: None
    """

    if results_dir is None:
        raise Exception('Results dir is not provided')

    objects_wh = np.transpose(np.vstack((objects_w, objects_h)))

    distance = None
    centroids = None

    for i in range(run_num):
        # Calculate centroids and distances for i-th iteration
        print("Running kmeans, run", i + 1, "of", run_num)
        centroids_i, distances_i = run_kmeans(objects_wh, anchor_num)

        # Order i-th centroids
        centroids_i = centroids_i[np.argsort(centroids_i[:, 0])]

        # Store only the best centroids (those with the smallest distance)
        if distance is None or distance > np.sum(distances_i):
            distance = np.sum(distances_i)
            centroids = centroids_i
            print("Updated best centroids.")

    print("Anchors in ascending order (this goes in the decoder config file):")
    print(centroids)

    out_file = os.path.join(results_dir, "anchor_boxes.json")

    data = dict(anchors=centroids.tolist())
    with open(out_file, 'w') as outfile:
        json.dump(data, outfile)
    print(f"Anchors in ascending order are dumped in file: {out_file}")


def run_kmeans(ann_dims, anchor_num):
    """ Run k-means algorithm.
    Code taken/modified from
    https://github.com/experiencor/keras-yolo2/blob/master/gen_anchors.py
    :param ann_dims: data points.
    :param anchor_num: number anchors
    :return: list of means.
    """
    ann_num = ann_dims.shape[0]
    prev_assignments = np.full(ann_num, -1, dtype=np.int)
    centroids = ann_dims[np.random.randint(0, ann_dims.shape[0], anchor_num)]

    while True:
        # Calculate distances
        distances = np.zeros((ann_num, anchor_num), dtype=np.float)
        for i in range(ann_num):
            distances[i, :] = 1 - calc_iou(ann_dims[i], centroids)

        # Assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids, distances

        # Calculate new centroids
        for i in range(anchor_num):
            centroids[i] = np.sum(ann_dims[assignments == i], axis=0)
            centroids[i] = centroids[i] / (np.sum(assignments == i) + 1e-6)
        prev_assignments = assignments.copy()


def calc_iou(ann, centroids):
    """ Calculate intersection over union between two boxes.
    Code modified from https://github.com/experiencor/keras-yolo2/blob/master/gen_anchors.py
    :param ann: data points
    :param centroids: centroid coordinates
    :return: iou value
    """
    w, h = ann
    similarities = list()

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def main():
    parser = argparse.ArgumentParser(description='Argument parser for anchor generation')
    parser.add_argument('-a', '--anchors_num', help='number of anchors', default=9, type=int)
    parser.add_argument('-kr', '--k_mean_runs', help='run for a few images, useful for debug', default=3, type=int)
    parser.add_argument('-rd', '--results_dir', help='directory for output json file', type=str, required=True, )
    parser.add_argument('-ad', '--annot_dir', help='directory for bounding boxes annotation', type=str, required=True)
    args = parser.parse_args()

    # Parse objects width and heights from dataset annotation into lists
    objects_w, objects_h = get_objects_width_and_height(annot_dir=args.annot_dir)

    # Generate the anchors
    generate_anchors(results_dir=args.results_dir,
                     run_num=args.k_mean_runs,
                     anchor_num=args.anchors_num,
                     objects_w=objects_w,
                     objects_h=objects_h)


if __name__ == '__main__':
    main()
