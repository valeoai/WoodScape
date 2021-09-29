"""
InverseWarp for Fisheye for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import torch

from train_utils.distance_utils import bilinear_sampler
from losses.distance_loss import InverseWarp


class PhotometricFisheye(InverseWarp):
    def __init__(self, args):
        """Inverse Warp class for Fisheye
        :param args: input params from config file
        """
        super().__init__(args)
        self.crop = args.crop

    @staticmethod
    @torch.jit.script
    def img2world(norm: torch.Tensor, theta_lut, angle_lut, essential_mat) -> torch.Tensor:
        """Transform coordinates in the pixel frame to the camera frame.
        :param norm: norm values for the pixels -- [B x H * W]
        :param theta_lut: Look up table containing coords for angle in the image plane -- [B x H * W x 1]
        :param angle_lut: Look up table containing coords for angle of incidence -- [B x H * W x 1]
        :param essential_mat: The camera transform matrix -- [B x 4 x 4]
        :return: world_coords: The world based coordinates -- [B x 4 x H * W]
        """
        norm = norm.reshape(norm.size(0), 1, -1)  # B x 1 x H * W
        # angle in the image plane
        theta = theta_lut.permute(0, 2, 1)
        # Obtain angle of incidence from radius
        angle_of_incidence = (angle_lut.permute(0, 2, 1)).to(device=norm.device)
        r_world = torch.sin(angle_of_incidence) * norm
        x = r_world * torch.cos(theta)
        y = r_world * torch.sin(theta)
        # Obtain `z` from the norm
        z = torch.cos(angle_of_incidence) * norm
        cam_coords = torch.cat((x, y, z), 1)
        cam_coords = torch.cat(
            [cam_coords, torch.ones(cam_coords.size(0), 1, cam_coords.shape[2]).to(device=norm.device)], 1)
        world_coords = essential_mat @ cam_coords
        return world_coords

    def world2img(self, world_coords: torch.Tensor, intrinsics: torch.Tensor, distortion_coeffs: torch.Tensor,
                  height: int, width: int) -> tuple:
        """Transform 3D world co-ordinates to the pixel frame.
        :param world_coords: The camera based coords -- [B x 4 x H * W]
        :param intrinsics: The camera intrinsics -- [B x 4]
        :param distortion_coeffs: k1, k2, k3, k4 -- [B x 4]
        :param height: image height
        :param width: image width
        :return: pixel_coords, mask: The pixel coordinates corresponding to points -- [B x 2 x H * W], [B x 1 x H * W]
        """
        x_cam, y_cam, z = [world_coords[:, i, :].unsqueeze(1) for i in range(3)]
        # angle in the image plane
        theta = torch.atan2(y_cam, x_cam)
        # radius from angle of incidence
        r = torch.sqrt(x_cam * x_cam + y_cam * y_cam)
        # Calculate angle using z
        a = np.pi / 2 - torch.atan2(z, r)
        distortion_coeffs = distortion_coeffs.unsqueeze(1).unsqueeze(1)
        r_mapping = sum([distortion_coeffs[:, :, :, i] * torch.pow(a, i + 1) for i in range(4)])

        intrinsics = intrinsics.unsqueeze(1).unsqueeze(1)
        x = r_mapping * torch.cos(theta) * intrinsics[:, :, :, 0] + intrinsics[:, :, :, 2]
        y = r_mapping * torch.sin(theta) * intrinsics[:, :, :, 1] + intrinsics[:, :, :, 3]

        x_norm = 2 * x / (width - 1) - 1
        y_norm = 2 * y / (height - 1) - 1
        pcoords_norm = torch.cat([x_norm, y_norm], 1)  # b x 2 x hw
        if self.ego_mask:
            x_mask = (x_norm > -1) & (x_norm < 1)
            y_mask = (y_norm > -1) & (y_norm < 1)
            mask = (x_mask & y_mask).reshape(pcoords_norm.size(0), 1, height, width).float()
        else:
            mask = None
        return pcoords_norm, mask

    def inverse_warp(self, source_img, norm, car_mask,
                     extrinsic_mat, K, D, theta_lut, angle_lut) -> tuple:
        """Inverse warp a source image to the target image plane for fisheye images
        :param source_img: source image (to sample pixels from) -- [B x 3 x H x W]
        :param norm: Distance map of the target image -- [B x 1 x H x W]
        :param extrinsic_mat: DoF pose vector from target to source -- [B x 4 x 4]
        :param K: Camera intrinsic matrix -- [B x 4]
        :param D: Camera distortion co-efficients k1, k2, k3, k4 -- [B x 4]
        :param theta_lut: Look up table containing coords for angle in the image plane -- [B x H * W x 1]
        :param angle_lut: Look up table containing coords for angle of incidence -- [B x H * W x 1]
        :return: Projected source image -- [B x 3 x H x W]
        """
        batch_size, _, height, width = source_img.size()
        norm = norm.reshape(batch_size, height * width)

        world_coords = self.img2world(norm, theta_lut, angle_lut, extrinsic_mat)  # [B x 4 x H * W]
        image_coords, mask = self.world2img(world_coords, K, D, height, width)

        padding_mode = "border" if self.crop else "zeros"
        projected_img = bilinear_sampler(source_img, image_coords, mode='bilinear', padding_mode=padding_mode)

        if not self.crop:
            # TODO: Check interpolation
            # car_hood = bilinear_sampler(car_mask, image_coords, mode='nearest', padding_mode=padding_mode)
            projected_img = projected_img * car_mask

        return projected_img, mask

    def fisheye_inverse_warp(self, inputs, outputs):
        for scale in range(self.num_scales):
            for frame_id in self.frame_idxs[1:]:
                norm = outputs[("norm", scale)]
                norm = self.scale_norm(norm)

                if not self.crop:
                    car_mask = inputs["mask", 0]
                    norm = norm * car_mask
                    norm[norm == 0] = 0.1
                else:
                    car_mask = None

                image = inputs[("color", frame_id, 0)]
                intrinsic_mat = inputs[("K", frame_id)]
                distortion_coeffs = inputs[("D", frame_id)]
                theta_lut = inputs[("theta_lut", frame_id)]
                angle_lut = inputs[("angle_lut", frame_id)]
                essential_mat = outputs[("cam_T_cam", 0, frame_id)]

                outputs[("color", frame_id, scale)], outputs[("ego_mask", frame_id, scale)] = \
                    self.inverse_warp(image,
                                      norm,
                                      car_mask,
                                      essential_mat,
                                      intrinsic_mat,
                                      distortion_coeffs,
                                      theta_lut,
                                      angle_lut)

    def warp(self, inputs, outputs):
        self.fisheye_inverse_warp(inputs, outputs)
