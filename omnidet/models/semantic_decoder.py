"""
Semantic Decoder model for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import torch
import torch.nn as nn

from models.normnet_decoder import PixelShuffleICNR, conv3x3


def convblock(in_planes, out_planes):
    """Layer to perform a convolution, BatchNorm followed by ReLU"""
    return nn.Sequential(conv3x3(in_planes, out_planes),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))


class SemanticDecoder(nn.Module):
    def __init__(self, num_ch_enc, n_classes=20):
        super().__init__()
        self.n_classes = n_classes

        self.num_ch_enc = num_ch_enc  # [64, 64, 128, 256, 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

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

        self.semantic_conv_0 = conv3x3(self.num_ch_dec[0], self.n_classes)

        self.shuffle_conv_4_0 = PixelShuffleICNR(self.num_ch_dec[4], self.num_ch_dec[4] * 4)
        self.shuffle_conv_3_0 = PixelShuffleICNR(self.num_ch_dec[3], self.num_ch_dec[3] * 4)
        self.shuffle_conv_2_0 = PixelShuffleICNR(self.num_ch_dec[2], self.num_ch_dec[2] * 4)
        self.shuffle_conv_1_0 = PixelShuffleICNR(self.num_ch_dec[1], self.num_ch_dec[1] * 4)
        self.shuffle_conv_0_0 = PixelShuffleICNR(self.num_ch_dec[0], self.num_ch_dec[0] * 4)

        self.shuffle_conv_3 = PixelShuffleICNR(self.num_ch_dec[3], 64 * self.n_classes, scale=8)
        self.shuffle_conv_2 = PixelShuffleICNR(self.num_ch_dec[2], 16 * self.n_classes, scale=4)
        self.shuffle_conv_1 = PixelShuffleICNR(self.num_ch_dec[1], 4 * self.n_classes, scale=2)

    def forward(self, input_features):
        outputs = dict()
        x = input_features[-1]

        x = self.upconv_4_0(x)
        x = self.shuffle_conv_4_0(x)
        x_g3 = torch.cat((x, input_features[3]), dim=1)
        x = self.upconv_4_1(x_g3)

        x = self.upconv_3_0(x)
        x = self.shuffle_conv_3_0(x)
        x_g2 = torch.cat((x, input_features[2]), dim=1)
        x_3 = self.upconv_3_1(x_g2)

        x = self.upconv_2_0(x_3)
        x = self.shuffle_conv_2_0(x)
        x_g1 = torch.cat((x, input_features[1]), dim=1)
        x_2 = self.upconv_2_1(x_g1)

        x = self.upconv_1_0(x_2)
        x = self.shuffle_conv_1_0(x)
        x_g0 = torch.cat((x, input_features[0]), dim=1)
        x_1 = self.upconv_1_1(x_g0)

        x = self.upconv_0_0(x_1)
        x = self.shuffle_conv_0_0(x)

        if torch.onnx.is_in_onnx_export():
            return self.semantic_conv_0(x)
        else:
            outputs[("semantic", 0)] = self.semantic_conv_0(x)
            return outputs
