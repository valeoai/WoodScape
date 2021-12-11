"""
Motion Decoder model for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import torch
import torch.nn as nn

from models.normnet_decoder import PixelShuffleICNR, conv3x3
from models.semantic_decoder import convblock


class MotionDecoder(nn.Module):
    def __init__(self, num_ch_enc, n_classes=2, siamese_net=False):
        super().__init__()
        self.n_classes = n_classes

        self.num_ch_enc = num_ch_enc  # [64, 64, 128, 256, 512]
        # [64, 64, 128, 256, 512] for motion_decoder and [128, 128, 256, 512, 1024] for siamese net
        self.num_ch_enc = num_ch_enc if not siamese_net else self.num_ch_enc * 2
        self.num_ch_dec = np.array([16, 32, 64, 128, 256]) if not siamese_net else np.array([16, 32, 64, 128, 256]) * 2

        # decoder
        self.upconv_4_0 = convblock(self.num_ch_enc[-1], self.num_ch_dec[4])
        self.upconv_4_1 = convblock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])

        self.upconv_3_0 = convblock(self.num_ch_dec[4], self.num_ch_dec[3])
        self.upconv_3_1 = convblock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])

        self.upconv_2_0 = convblock(self.num_ch_dec[3], self.num_ch_dec[2])
        self.upconv_2_1 = convblock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])

        self.upconv_1_0 = convblock(self.num_ch_dec[2], self.num_ch_dec[1])
        self.upconv_1_1 = convblock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])

        self.upconv_0_0 = convblock(self.num_ch_dec[1], self.num_ch_dec[0])
        self.upconv_0_1 = convblock(self.num_ch_dec[0], self.num_ch_dec[0])

        self.motion_conv_0 = conv3x3(self.num_ch_dec[0], self.n_classes)

        self.shuffle_conv_4_0 = PixelShuffleICNR(self.num_ch_dec[4], self.num_ch_dec[4] * 4)
        self.shuffle_conv_3_0 = PixelShuffleICNR(self.num_ch_dec[3], self.num_ch_dec[3] * 4)
        self.shuffle_conv_2_0 = PixelShuffleICNR(self.num_ch_dec[2], self.num_ch_dec[2] * 4)
        self.shuffle_conv_1_0 = PixelShuffleICNR(self.num_ch_dec[1], self.num_ch_dec[1] * 4)
        self.shuffle_conv_0_0 = PixelShuffleICNR(self.num_ch_dec[0], self.num_ch_dec[0] * 4)

    def forward(self, input_features):
        outputs = dict()
        x = input_features[-1]

        x = self.upconv_4_0(x)
        x = self.shuffle_conv_4_0(x)
        x = torch.cat((x, input_features[3]), dim=1)
        x = self.upconv_4_1(x)

        x = self.upconv_3_0(x)
        x = self.shuffle_conv_3_0(x)
        x = torch.cat((x, input_features[2]), dim=1)
        x = self.upconv_3_1(x)

        x = self.upconv_2_0(x)
        x = self.shuffle_conv_2_0(x)
        x = torch.cat((x, input_features[1]), dim=1)
        x = self.upconv_2_1(x)

        x = self.upconv_1_0(x)
        x = self.shuffle_conv_1_0(x)
        x = torch.cat((x, input_features[0]), dim=1)
        x = self.upconv_1_1(x)

        x = self.upconv_0_0(x)
        x = self.shuffle_conv_0_0(x)

        if torch.onnx.is_in_onnx_export():
            return self.motion_conv_0(x)
        else:
            outputs[("motion", 0)] = self.motion_conv_0(x)
            return outputs
