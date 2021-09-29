"""
Loss function for Distance Estimation for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/nianticlabs/monodepth2
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
import torch.nn as nn


class InverseWarp:
    def __init__(self, args):
        self.ego_mask = args.ego_mask
        self.frame_idxs = args.frame_idxs
        self.num_scales = args.num_scales
        self.min_distance = args.min_distance
        self.max_distance = args.max_distance

    def warp(self, inputs, outputs) -> None:
        raise NotImplementedError("Invalid InverseWarp Attempted!")

    def scale_norm(self, norm):
        """Convert network's sigmoid output into norm prediction"""
        return self.min_distance + self.max_distance * norm


class PhotometricReconstructionLoss(nn.Module):
    def __init__(self, inverse_warp_object: InverseWarp, args):
        """Loss function for unsupervised monocular distance
        :param args: input params from config file
        """
        super().__init__()
        self.warp = inverse_warp_object

        self.frame_idxs = args.frame_idxs
        self.num_scales = args.num_scales
        self.crop = args.crop
        self.seed = 1e-7

        self.disable_auto_mask = args.disable_auto_mask
        self.clip_loss = args.clip_loss_weight
        self.ssim_weight = args.ssim_weight
        self.reconstr_weight = args.reconstr_weight
        self.smooth_weight = args.smooth_weight

    def norm_smoothness(self, norm: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Smoothens the output distance map or distance map
        :param norm: Depth map of the target image -- [B x 1 x H x W]
        :param img: Images from the image_stack -- [B x 3 x H x W]
        :return Mean value of the smoothened image
        """
        norm_gradients_x = torch.abs(norm[:, :, :, :-1] - norm[:, :, :, 1:])
        norm_gradients_y = torch.abs(norm[:, :, :-1, :] - norm[:, :, 1:, :])

        image_gradients_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        image_gradients_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        norm_gradients_x *= torch.exp(-image_gradients_x)
        norm_gradients_y *= torch.exp(-image_gradients_y)

        return norm_gradients_x.mean() + norm_gradients_y.mean()

    @staticmethod
    def ssim(x, y):
        """Computes a differentiable structured image similarity measure."""
        x = nn.ReflectionPad2d(1)(x)
        y = nn.ReflectionPad2d(1)(y)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = nn.AvgPool2d(kernel_size=3, stride=1)(x)
        mu_y = nn.AvgPool2d(kernel_size=3, stride=1)(y)
        sigma_x = nn.AvgPool2d(kernel_size=3, stride=1)(x ** 2) - mu_x ** 2
        sigma_y = nn.AvgPool2d(kernel_size=3, stride=1)(y ** 2) - mu_y ** 2
        sigma_xy = nn.AvgPool2d(kernel_size=3, stride=1)(x * y) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

    def compute_reprojection_loss(self, predicted, target, ego_mask=None):
        """Computes reprojection loss between predicted and target images"""
        if type(ego_mask) == torch.Tensor:
            l1_loss = (torch.abs(target - predicted) * ego_mask).mean(1, True)
            ssim_error = self.ssim(predicted, target)
            ssim_loss = (ssim_error * ego_mask).mean(1, True)
        else:
            l1_loss = torch.abs(target - predicted).mean(1, True)
            ssim_loss = self.ssim(predicted, target).mean(1, True)

        reprojection_loss = self.ssim_weight * ssim_loss + self.reconstr_weight * l1_loss

        if self.clip_loss:
            mean, std = reprojection_loss.mean(), reprojection_loss.std()
            reprojection_loss = torch.clamp(reprojection_loss, max=float(mean + self.clip_loss * std))

        return reprojection_loss, l1_loss, ssim_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses"""
        losses = dict()
        total_loss = 0
        target = inputs[("color", 0, 0)]
        for scale in range(self.num_scales):
            loss = 0

            # --- PHOTO-METRIC LOSS ---
            reprojection_loss = list()
            for frame_id in self.frame_idxs[1:]:
                pred = outputs[("color", frame_id, scale)]
                if self.crop:
                    ego_mask = outputs[("ego_mask", frame_id, scale)]
                else:
                    ego_mask = outputs[("ego_mask", frame_id, scale)] * inputs["mask", 0]
                    outputs[("ego_mask", frame_id, scale)] = ego_mask
                reproj_loss, l1, ssim = self.compute_reprojection_loss(pred, target, ego_mask)
                reprojection_loss.append(reproj_loss)
            reprojection_loss = torch.cat(reprojection_loss, 1)

            # --- AUTO MASK ---
            if not self.disable_auto_mask:
                identity_reprojection_loss = list()
                for frame_id in self.frame_idxs[1:]:
                    target = inputs[("color", 0, 0)]
                    pred = inputs[("color", frame_id, 0)]
                    reproj_loss, l1, ssim = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_loss.append(reproj_loss)
                identity_reprojection_loss = torch.cat(identity_reprojection_loss, 1)
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(
                    device=identity_reprojection_loss.device) * 1e-5
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            # --- COMPUTING MIN FOR MONOCULAR APPROACH ---
            if combined.shape[1] == 1:
                forward_optimise = combined
            else:
                forward_optimise, forward_idxs = torch.min(combined, dim=1)

            loss += forward_optimise.mean()

            # --- SMOOTHNESS LOSS ---
            inv_norm = 1 / outputs[("norm", 0)]
            normalized_norm = (inv_norm / (inv_norm.mean([2, 3], True) + self.seed))
            smooth_loss = self.norm_smoothness(normalized_norm, inputs[("color", 0, 0)])
            loss += self.smooth_weight * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"distance_loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["distance_loss"] = total_loss
        return losses

    def forward(self, inputs, outputs):
        """Loss function for self-supervised norm and pose on monocular videos"""
        self.warp.warp(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return losses, outputs
