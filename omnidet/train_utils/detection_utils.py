"""
Detection utils for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import shapely
import torch
from shapely.geometry import Polygon


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def to_cpu(tensor):
    return tensor.detach().cpu()


def get_tensor_value(tensor):
    if isinstance(tensor, torch.Tensor):
        return to_cpu(tensor).item()
    else:
        return tensor


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = (list(), list(), list())
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    :param recall:    The recall curve (list).
    :param precision: The precision curve (list).
    :return The average precision.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold, args):
    """
    Compute true positives, predicted scores and predicted labels per sample
    outputs: (x1, y1, x2, y2, conf, cls_conf, cls_pred)
    """
    batch_metrics = list()
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_thetas = output[:, 4]  # this is dummy value will not be used
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label, pred_theta) in enumerate(zip(pred_boxes, pred_labels, pred_thetas)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def get_contour(b_box, theta):
    # b_box in shape of (x_min, y_min, x_max, y_max)
    cont = shapely.geometry.box(b_box[0], b_box[1], b_box[2], b_box[3])
    rot_cont = shapely.affinity.rotate(cont, theta * -1, use_radians=False)
    return rot_cont


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    :return Detections with the shape (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, args):
    """
    :param pred_boxes:
    :param pred_cls:
    :param target: (img_id, Class_type, x, y ,w ,h, yaw [if exist])
    :param anchors:
    :param ignore_thres:
    :param args: input arguments from params file
    :return:
    """
    device = args.device

    nB = pred_boxes.size(0)  # Batch size
    nA = pred_boxes.size(1)  # Anchor size
    nC = pred_cls.size(-1)  # Number of classes
    nG_y = pred_boxes.size(2)  # Grid size vertical
    nG_x = pred_boxes.size(3)  # Grid size horizontal

    # Output tensors
    obj_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device)
    noobj_mask = torch.ones([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device=obj_mask.device)
    class_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    iou_scores = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tx = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    ty = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tw = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    th = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tcls = torch.zeros([nB, nA, nG_y, nG_x, nC], dtype=torch.float).to(device=obj_mask.device)

    # Handel negative samples
    target = target[target[:, 4] > 0]
    if target.nelement() == 0:
        tconf = obj_mask.float()
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    # Assign each object to corresponding grid cell
    target_boxes = torch.cat(((target[:, 2] * nG_x).unsqueeze(1), (target[:, 3] * nG_y).unsqueeze(1),
                              (target[:, 4] * nG_x).unsqueeze(1), (target[:, 5] * nG_y).unsqueeze(1)), 1)
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    # ious shape is (3, number_of_boxes)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # Handel corner cases(boundaries cases)
    gi = torch.clamp(gi, 0, nG_x - 1)
    gj = torch.clamp(gj, 0, nG_y - 1)

    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Normalize coordinates relative to the cell (x & y in range [0,1])
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    # Assign the class type
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def log_metrics(outputs, targets, losses):
    target = targets[0]
    output = outputs[0]
    # Metrics
    cls_acc = 100 * target["class_mask"][target["obj_mask"]].mean()
    conf_obj = output["obj_conf"].mean()
    conf_noobj = output["no_obj_conf"].mean()
    conf50 = (output["pred_conf"] > 0.5).float()
    iou50 = (target["iou_scores"] > 0.5).float()
    iou75 = (target["iou_scores"] > 0.75).float()
    detected_mask = conf50 * target["class_mask"] * target["tconf"]
    precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
    recall50 = torch.sum(iou50 * detected_mask) / (target["obj_mask"].sum() + 1e-16)
    recall75 = torch.sum(iou75 * detected_mask) / (target["obj_mask"].sum() + 1e-16)

    metrics = dict(detection_loss=get_tensor_value(losses["detection_loss"]),
                   x=get_tensor_value(losses["x"]), y=get_tensor_value(losses["y"]),
                   w=get_tensor_value(losses["w"]), h=get_tensor_value(losses["h"]),
                   conf=get_tensor_value(losses["conf"]),
                   cls=get_tensor_value(losses["cls"]),
                   cls_acc=get_tensor_value(cls_acc),
                   recall50=get_tensor_value(recall50), recall75=get_tensor_value(recall75),
                   precision=get_tensor_value(precision),
                   conf_obj=get_tensor_value(conf_obj), conf_noobj=get_tensor_value(conf_noobj))
    return metrics


def scale_annotation(box, scaled_size, image_shape, start_box_idx):
    """
    Scale the output boxes to the desired shape (up-sampling / down-sampling)
    :param box: the predected boxes on the scale of the feed images (labels ,x1, y1, x2, y2, yaw)
    :param scaled_size: the desired shape (width, height)
    :param image_shape: the original size (height, width)
    :param start_box_idx: the start index that the box coordinates starts from
    :return:
    """
    # Parse the shape
    height = image_shape[0]
    width = image_shape[1]

    # Compute ratio --> Fraction means downsizing
    ratio_height = scaled_size[1] / height
    ratio_width = scaled_size[0] / width

    # Multiply box with scale
    box[:, start_box_idx + 1] = np.multiply(box[:, start_box_idx + 1], ratio_height)
    box[:, start_box_idx] = np.multiply(box[:, start_box_idx], ratio_width)
    box[:, start_box_idx + 3] = np.multiply(box[:, start_box_idx + 3], ratio_height)
    box[:, start_box_idx + 2] = np.multiply(box[:, start_box_idx + 2], ratio_width)

    return box


def crop_annotation(box, cropping, accepted_crop_ratio, img_size: tuple,
                    orginial_image_size, enable_scaling=False):
    """
    The function takes the cropping and applies it to the bounding boxes.
    :param box: box (labels, x, y, w ,h, yaw)
    :param cropping: desired crop from left,top,right,bottom
    :param accepted_crop_ratio: determines the percentage of accepted area after cropping.
    :param img_size: [w, h] the desired size
    :param orginial_image_size: Original image size before cropping [w,h] (as loaded from the disk)
    :param enable_scaling: scale annotation to desired size after cropping.
    :returns box: The Cropped and scaled box.
    """
    image_width = img_size[0]
    image_height = img_size[1]
    org_image_width = orginial_image_size[0]
    org_image_height = orginial_image_size[1]
    box_xyxy = box.clone()

    # Parse the shape
    height = abs(cropping["top"] - cropping["bottom"])
    width = abs(cropping["left"] - cropping["right"])

    # Compute x1,y1 and x2,y2 un-normalized
    box_xyxy[:, 1] = (box[:, 1] - (box[:, 3] / 2)) * org_image_width
    box_xyxy[:, 2] = (box[:, 2] - (box[:, 4] / 2)) * org_image_height
    box_xyxy[:, 3] = (box[:, 1] + (box[:, 3] / 2)) * org_image_width
    box_xyxy[:, 4] = (box[:, 2] + (box[:, 4] / 2)) * org_image_height

    org_box_width = box_xyxy[:, 3] - box_xyxy[:, 1]
    org_box_height = box_xyxy[:, 4] - box_xyxy[:, 2]

    # Subtract x and y from the box points and Handle the boundaries
    box_xyxy[:, 1] = box_xyxy[:, 1] - cropping["left"]
    box_xyxy[:, 3] = box_xyxy[:, 3] - cropping["left"]
    box_xyxy[:, 2] = box_xyxy[:, 2] - cropping["top"]
    box_xyxy[:, 4] = box_xyxy[:, 4] - cropping["top"]

    # Compute area of the overlapped box
    new_boxes_area = (box_xyxy[:, 3] - box_xyxy[:, 1]) * (box_xyxy[:, 4] - box_xyxy[:, 2])

    # Apply filtering according to area relative to the original box
    skipbox1 = new_boxes_area < (accepted_crop_ratio * (org_box_width * org_box_height))

    # Check area of the output
    # skipbox2 = new_boxes_area < min_accepted_area
    skipbox2_w = abs(box_xyxy[:, 1] - box_xyxy[:, 3]) < 30
    skipbox2_h = abs(box_xyxy[:, 2] - box_xyxy[:, 4]) < 20
    skipbox2 = torch.mul(skipbox2_w, skipbox2_h)

    # Filter boxes according to area conditions
    skipBox = torch.mul(skipbox1, skipbox2)
    box_xyxy_filtered = box_xyxy[skipBox == False]
    if len(box_xyxy_filtered) == 0:
        # Handle negative samples
        box_xyxy_filtered = torch.zeros(box_xyxy[0].unsqueeze(0).shape, dtype=torch.float64)
    box = box_xyxy_filtered.clone()

    # scaling the boxes to the desired image size
    if enable_scaling:
        box_xyxy_filtered = scale_annotation(box_xyxy_filtered,
                                             [image_width, image_height], [height, width],
                                             start_box_idx=1)

    # Convert xyxy to center, width and height and normalize them
    w = (box_xyxy_filtered[:, 3] - box_xyxy_filtered[:, 1])
    h = (box_xyxy_filtered[:, 4] - box_xyxy_filtered[:, 2])
    box[:, 1] = (box_xyxy_filtered[:, 1] + (w / 2)) / image_width
    box[:, 2] = (box_xyxy_filtered[:, 2] + (h / 2)) / image_height
    box[:, 3] = w / image_width
    box[:, 4] = h / image_height

    return box
