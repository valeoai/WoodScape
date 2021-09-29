"""
PoseNet model for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/nianticlabs/monodepth2
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = self.num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.squeeze = nn.Conv2d(self.num_ch_enc[-1], 256, kernel_size=1)
        self.pose_0 = nn.Conv2d(self.num_input_features * 256, 256, kernel_size=3, stride=stride, padding=1)
        self.pose_1 = nn.Conv2d(256, 256, kernel_size=3, stride=stride, padding=1)
        self.pose_2 = nn.Conv2d(256, 6 * self.num_frames_to_predict_for, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.squeeze(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        out = cat_features

        out = self.pose_0(out)
        out = self.relu(out)

        out = self.pose_1(out)
        out = self.relu(out)

        out = self.pose_2(out)

        pose = out.mean([2, 3])
        pose = 0.01 * pose.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = pose[..., :3]
        translation = pose[..., 3:]
        return axisangle, translation
