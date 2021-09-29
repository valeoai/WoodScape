"""
Detection decoder model for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

Parts of the code adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

from collections import OrderedDict

import torch.nn as nn

from train_utils.detection_utils import *


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, args):
        super().__init__()
        self.args = args
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.num_classes = self.args.num_classes_detection
        self.ignore_thres = 0.5
        self.metrics = dict()
        self.img_dim = [self.args.input_width, self.args.input_height]
        self.grid_size = [0, 0]  # grid size
        self.stride = [0, 0]  # grid size

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        # grid shape is [Height, Width]
        g = self.grid_size
        self.stride[1] = self.img_dim[1] / self.grid_size[0]  # Height
        self.stride[0] = self.img_dim[0] / self.grid_size[1]  # Width

        # Calculate offsets for each grid using loops for onnx generation.
        grid_x = []
        for i in range(g[0]):
            grid_x.append(torch.arange(g[1]))
        self.grid_x = torch.stack(grid_x).view([1, 1, g[0], g[1]]).to(self.args.device)

        grid_y = []
        for i in range(g[1]):
            grid_y.append(torch.arange(g[0]))
        self.grid_y = torch.stack(grid_y).t().view([1, 1, g[0], g[1]]).to(self.args.device)

        self.scaled_anchors = torch.Tensor([(a_w / self.stride[0], a_h / self.stride[1])
                                            for a_w, a_h in self.anchors]).to(self.args.device)

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def _post_proccess_output_target(self, targets):
        if targets is None:
            return self.output, 0, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=self.pred_boxes,
                pred_cls=self.pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
                args=self.args)

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            target_dict = dict(x=tx[obj_mask], y=ty[obj_mask],
                               w=tw[obj_mask], h=th[obj_mask],
                               obj_conf=tconf[obj_mask], no_obj_conf=tconf[noobj_mask],
                               cls=tcls[obj_mask],
                               iou_scores=iou_scores, tconf=tconf,
                               class_mask=class_mask, obj_mask=obj_mask)
            output_dict = dict(x=self.x[obj_mask], y=self.y[obj_mask],
                               w=self.w[obj_mask], h=self.h[obj_mask],
                               obj_conf=self.pred_conf[obj_mask], no_obj_conf=self.pred_conf[noobj_mask],
                               cls=self.pred_cls[obj_mask], pred_conf=self.pred_conf)

            return self.output, output_dict, target_dict

    def forward(self, x, targets=None, img_dim=None):
        if torch.is_tensor(img_dim):
            img_dim = img_dim.cpu().numpy()

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = [x.size(2), x.size(3)]  # Height, Width

        # x, y, w, h, conf
        num_of_output = 5
        prediction = (x.view(num_samples, self.num_anchors, self.num_classes + num_of_output,
                             grid_size[0], grid_size[1]).permute(0, 1, 3, 4, 2).contiguous())

        # For embedded deployment, use softmax as sigmoid
        self.x = torch.softmax(torch.stack([prediction[..., 0], torch.zeros_like(prediction[..., 0])],
                                           dim=prediction[..., 0].dim()), dim=prediction[..., 0].dim())[..., 0]
        self.y = torch.softmax(torch.stack([prediction[..., 1], torch.zeros_like(prediction[..., 1])],
                                           dim=prediction[..., 1].dim()), dim=prediction[..., 1].dim())[..., 0]
        self.w = prediction[..., 2]  # Width
        self.h = prediction[..., 3]  # Height

        self.pred_conf = torch.softmax(
            torch.stack([prediction[..., 4], torch.zeros_like(prediction[..., 4])],
                        dim=prediction[..., 4].dim()), dim=prediction[..., 4].dim())[..., 0]  # Conf
        self.pred_cls = torch.softmax(
            torch.stack([prediction[..., 5:], torch.zeros_like(prediction[..., 5:])],
                        dim=prediction[..., 5:].dim()), dim=prediction[..., 5:].dim())[..., 0]  # Cls pred

        # If grid size does not match current we compute new offsets
        if grid_size[0] != torch.tensor(self.grid_size[0]) or grid_size[1] != torch.tensor(self.grid_size[1]):
            self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        self.pred_boxes = torch.Tensor(prediction[..., :4].shape).to(self.args.device)

        self.pred_boxes = torch.cat(((self.x + self.grid_x).unsqueeze(4), (self.y + self.grid_y).unsqueeze(4),
                                     (torch.exp(self.w) * self.anchor_w).unsqueeze(4),
                                     (torch.exp(self.h) * self.anchor_h).unsqueeze(4)), dim=4)

        self.output = torch.cat((self.pred_boxes.view(num_samples, -1, 4) * self.stride[0],
                                 self.pred_conf.view(num_samples, -1, 1),
                                 self.pred_cls.view(num_samples, -1, self.num_classes)), -1)

        return self._post_proccess_output_target(targets)


######################################################################################################################
# #############################              Define YOLO Decoder             #########################################
######################################################################################################################


class YoloDecoder(nn.Module):
    def __init__(self, _out_filters, args):
        super(YoloDecoder, self).__init__()
        self.args = args
        output_shape = 5 + args.num_classes_detection

        #  embedding0
        final_out_filter0 = len(args.anchors1) * output_shape
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)

        #  embedding1
        final_out_filter1 = len(args.anchors1) * output_shape
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)

        #  embedding2
        final_out_filter2 = len(args.anchors1) * output_shape
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

        self._final_layer()

    def _final_layer(self):
        # Define final layer
        self.final_layer0 = YOLOLayer(self.args.anchors1, self.args)
        self.final_layer1 = YOLOLayer(self.args.anchors2, self.args)
        self.final_layer2 = YOLOLayer(self.args.anchors3, self.args)

    def _make_cbl(self, _in, _out, ks):
        """ cbl = conv + batch_norm + leaky_relu"""
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1,
                                                             padding=pad, bias=False)),
                                          ("bn", nn.BatchNorm2d(_out)),
                                          ("relu", nn.ReLU())]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([self._make_cbl(in_filters, filters_list[0], 1),
                           self._make_cbl(filters_list[0], filters_list[1], 3),
                           self._make_cbl(filters_list[1], filters_list[0], 1),
                           self._make_cbl(filters_list[0], filters_list[1], 3),
                           self._make_cbl(filters_list[1], filters_list[0], 1),
                           self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True))
        return m

    def forward(self, backbone_features, img_dim, targets=None):
        outputs = dict()
        yolo_outputs = list()
        yolo_output_dict_list = list()
        yolo_target_dict_list = list()

        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        x2, x1, x0 = backbone_features[-3:]

        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)

        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)

        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        # Calculate the loss and do some post-processing on the outputs
        x, output_dict, target_dict = self.final_layer0(out0, targets, img_dim)
        yolo_outputs.append(x)
        yolo_output_dict_list.append(output_dict)
        yolo_target_dict_list.append(target_dict)

        x, output_dict, target_dict = self.final_layer1(out1, targets, img_dim)
        yolo_outputs.append(x)
        yolo_output_dict_list.append(output_dict)
        yolo_target_dict_list.append(target_dict)

        x, output_dict, target_dict = self.final_layer2(out2, targets, img_dim)
        yolo_outputs.append(x)
        yolo_output_dict_list.append(output_dict)
        yolo_target_dict_list.append(target_dict)

        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()

        if torch.onnx.is_in_onnx_export():
            return yolo_outputs
        else:
            outputs["yolo_outputs"] = yolo_outputs
            outputs["yolo_output_dicts"] = yolo_output_dict_list if targets is not None else None
            outputs["yolo_target_dicts"] = yolo_target_dict_list if targets is not None else None
            return outputs
