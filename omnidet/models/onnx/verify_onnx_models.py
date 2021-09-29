#!/usr/bin/env python3
"""
ONNX model verification of perception models for OmniDet.

#usage: ./verify_onnx_models.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import onnxruntime
import torch
from PIL import Image
from matplotlib import pyplot as plt

from eval.qualitative_detection import non_max_suppression, get_contour
from main import collect_tupperware
from utils import semantic_color_encoding
from eval.qualitative_detection import color_encoding_woodscape_detection

ALPHA = 0.5


def motion_color_encoding():
    motion_classes = dict(static=(0, 0, 0), motion=(255, 0, 0))
    motion_color_encoding = np.zeros((2, 3), dtype=np.uint8)
    for i, (k, v) in enumerate(motion_classes.items()):
        motion_color_encoding[i] = v
    return motion_color_encoding


def post_process_distance(output):
    inv_norm = 1 / np.asarray(output[0])
    vmax = np.percentile(inv_norm, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_norm.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cm.get_cmap('magma'))
    colormapped_im = (mapper.to_rgba(inv_norm)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def post_process_semantic_motion(predictions, input_img, color_coding):
    color_output = input_img.copy()
    predictions = np.argmax(predictions, 0)
    not_background = predictions != 0
    color_output[not_background, ...] = (color_output[not_background, ...] * (1 - ALPHA) +
                                         color_coding[predictions[not_background]] * ALPHA)
    return color_output


def post_process_detection(outputs, args, input_img, color_coding):
    outputs = non_max_suppression(outputs, conf_thres=args.detection_conf_thres, nms_thres=args.detection_nms_thres)
    if outputs[0] is None:
        print("There are no boxes, please train more epochs, or check your data")
        return outputs
    detection_outputs = torch.cat(outputs, dim=0)

    color_output = input_img.copy()
    for box in detection_outputs:  # box shape is (x1, y1, x2, y2, conf, theta, cls_conf, cls_pred)
        # Get class name and color
        cls_pred = int(box[7]) if args.theta or args.theta_1_2 else int(box[6])
        class_color = (color_coding[cls_pred]).tolist()
        class_name = args.classes_names[cls_pred]
        x1, y1, conf = box[0], box[1], box[4]
        box = get_contour([box[0], box[1], box[2], box[3]], box[5]).exterior.coords
        boxes = np.int0(box)[0:4]
        box = np.int0([[b[0], b[1]] for b in boxes])
        cv2.drawContours(color_output, [box], 0, class_color, thickness=2)
        cv2.putText(color_output, str(f"{conf:.2f}"), (x1 - 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5e-3 * color_output.shape[0], (0, 255, 0), 1)
    return color_output


def pre_image_op(args, index, frame_index, cam_side):
    total_car1_images = 6054
    cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                    MVL=(343, 5, 1088, 411),
                                    MVR=(185, 5, 915, 425),
                                    RV=(186, 203, 1105, 630)),
                          Car2=dict(FV=(160, 272, 1030, 677),
                                    MVL=(327, 7, 1096, 410),
                                    MVR=(175, 4, 935, 404),
                                    RV=(285, 187, 1000, 572)))
    if args.crop:
        if int(frame_index[1:]) < total_car1_images:
            cropped_coords = cropped_coords["Car1"][cam_side]
        else:
            cropped_coords = cropped_coords["Car2"][cam_side]
    else:
        cropped_coords = None

    cropped_image = get_image(args, index, cropped_coords, frame_index, cam_side)
    resized_image = cv2.resize(np.array(cropped_image), (args.input_width, args.input_height),
                               cv2.INTER_LANCZOS4).transpose((2, 0, 1))
    resized_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return resized_image / 255


def get_image(args, index, cropped_coords, frame_index, cam_side):
    recording_folder = "rgb_images" if index == 0 else "previous_images"
    file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
    path = os.path.join(args.dataset_dir, recording_folder, file)
    image = Image.open(path).convert('RGB')
    if args.crop:
        return image.crop(cropped_coords)
    return image


def verify_onnx_model(args):
    ort_session = onnxruntime.InferenceSession(args.onnx_load_model)
    previous_frames = ort_session.get_inputs()[0].name
    current_frames = ort_session.get_inputs()[1].name

    semantic_color_coding = semantic_color_encoding(args)
    motion_color_coding = motion_color_encoding()
    detection_color_coding = color_encoding_woodscape_detection()

    image_paths = [line.rstrip('\n') for line in open(args.val_file)]
    for path in image_paths:
        frame_index, cam_side = path.split('.')[0].split('_')
        previous_frame = pre_image_op(args, -1, frame_index, cam_side)
        current_frame = pre_image_op(args, 0, frame_index, cam_side)
        output = ort_session.run(None, {previous_frames: previous_frame, current_frames: current_frame})
        # Note: Output list order is Depth, Semantic, Motion and Object Detection
        distance_image = post_process_distance(output[0])
        current_frame = np.squeeze(current_frame * 255, 0).transpose(1, 2, 0).astype(np.uint8)
        semantic_image = post_process_semantic_motion(output[1], current_frame, semantic_color_coding)
        motion_image = post_process_semantic_motion(output[2], current_frame, motion_color_coding)
        detection_output = torch.from_numpy(output[3])
        detection_image = post_process_detection(detection_output, args, current_frame, detection_color_coding)
        concat_output_1 = np.concatenate([distance_image, detection_image], 1)
        concat_output_2 = np.concatenate([semantic_image, motion_image], 1)
        neurall = np.concatenate([concat_output_1, concat_output_2], 0)
        plt.imshow(neurall)
        plt.show()


if __name__ == "__main__":
    # load your predefined ONNX model
    args = collect_tupperware()
    verify_onnx_model(args)
