"""
Normnet decoder model with PixelShuffle for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/nianticlabs/monodepth2
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import torch
import torch.nn as nn


def convblock(in_planes, out_planes):
    """Layer to perform a convolution followed by ELU"""
    return nn.Sequential(conv3x3(in_planes, out_planes), nn.ELU(inplace=True))


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PixelShuffleICNR(nn.Module):
    def __init__(self, in_planes, out_planes, scale=2):
        super().__init__()
        self.conv = conv1x1(in_planes, out_planes)
        self.shuffle = nn.PixelShuffle(scale)
        kernel = self.ICNR(self.conv.weight, upscale_factor=scale)
        self.conv.weight.data.copy_(kernel)

    @staticmethod
    def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
        """Fills the input Tensor or Variable with values according to the method
        described in "Checkerboard artifact free sub-pixel convolution" https://arxiv.org/abs/1707.02937
        Andrew Aitken et al. (2017), this inizialization should be used in the
        last convolutional layer before a PixelShuffle operation
        :param tensor: an n-dimensional torch.Tensor or autograd.Variable
        :param upscale_factor: factor to increase spatial resolution by
        :param inizializer: inizializer to be used for sub_kernel inizialization
        """
        new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
        sub_kernel = torch.zeros(new_shape)
        sub_kernel = inizializer(sub_kernel)
        sub_kernel = sub_kernel.transpose(0, 1)
        sub_kernel = sub_kernel.contiguous().view(sub_kernel.shape[0], sub_kernel.shape[1], -1)
        kernel = sub_kernel.repeat(1, 1, upscale_factor ** 2)
        transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

    def forward(self, x):
        return self.shuffle(self.conv(x))


class NormDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.sigmoid = nn.Sigmoid()

        self.num_ch_enc = num_ch_enc
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

        self.normconv_0 = conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.shuffle_conv_4_0 = PixelShuffleICNR(self.num_ch_dec[4], self.num_ch_dec[4] * 4)
        self.shuffle_conv_3_0 = PixelShuffleICNR(self.num_ch_dec[3], self.num_ch_dec[3] * 4)
        self.shuffle_conv_2_0 = PixelShuffleICNR(self.num_ch_dec[2], self.num_ch_dec[2] * 4)
        self.shuffle_conv_1_0 = PixelShuffleICNR(self.num_ch_dec[1], self.num_ch_dec[1] * 4)
        self.shuffle_conv_0_0 = PixelShuffleICNR(self.num_ch_dec[0], self.num_ch_dec[0] * 4)

        self.shuffle_conv_3 = PixelShuffleICNR(self.num_ch_dec[3], 64, scale=8)
        self.shuffle_conv_2 = PixelShuffleICNR(self.num_ch_dec[2], 16, scale=4)
        self.shuffle_conv_1 = PixelShuffleICNR(self.num_ch_dec[1], 4, scale=2)

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
        x_3 = self.upconv_3_1(x)

        x = self.upconv_2_0(x_3)
        x = self.shuffle_conv_2_0(x)
        x = torch.cat((x, input_features[1]), dim=1)
        x_2 = self.upconv_2_1(x)

        x = self.upconv_1_0(x_2)
        x = self.shuffle_conv_1_0(x)
        x = torch.cat((x, input_features[0]), dim=1)
        x_1 = self.upconv_1_1(x)

        x = self.upconv_0_0(x_1)
        x = self.shuffle_conv_0_0(x)

        if torch.onnx.is_in_onnx_export():
            return self.sigmoid(self.normconv_0(x))
        else:
            outputs[("norm", 3)] = self.sigmoid(self.shuffle_conv_3(x_3))
            outputs[("norm", 2)] = self.sigmoid(self.shuffle_conv_2(x_2))
            outputs[("norm", 1)] = self.sigmoid(self.shuffle_conv_1(x_1))
            outputs[("norm", 0)] = self.sigmoid(self.normconv_0(x))
            return outputs
