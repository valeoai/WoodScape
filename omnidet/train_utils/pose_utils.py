"""
Pose estiamtion training utils for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/nianticlabs/monodepth2
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch


def pose_vec2mat(axisangle, translation, invert=False, rotation_mode='euler'):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix"""
    if rotation_mode == "euler":
        R = euler2mat(axisangle)
    elif rotation_mode == "quat":
        R = quat2mat(axisangle)

    t = translation.clone()
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    T = get_translation_matrix(t)
    if invert:
        essential_mat = torch.matmul(R, T)
    else:
        essential_mat = torch.matmul(T, R)
    return essential_mat


@torch.jit.script
def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix"""
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T


@torch.jit.script
def euler2mat(angle):
    """Convert euler angles to rotation matrix.
    :param angle: rotation angle along 3 axis (in radians) -- [B x 1 x 3]
    :return Rotation matrix corresponding to the euler angles -- [B x 4 x 4]
    """
    batch_size = angle.size(0)
    x, y, z = angle[:, :, 0], angle[:, :, 1], angle[:, :, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z * 0
    ones = zeros + 1
    zmat = torch.stack([cosz, -sinz, zeros, zeros,
                        sinz, cosz, zeros, zeros,
                        zeros, zeros, ones, zeros,
                        zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros,
                        zeros, ones, zeros, zeros,
                        -siny, zeros, cosy, zeros,
                        zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros,
                        zeros, cosx, -sinx, zeros,
                        zeros, sinx, cosx, zeros,
                        zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)

    rotMat = xmat @ ymat @ zmat
    return rotMat


@torch.jit.script
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    :param quat: quat: first three coeffs of quaternion are rotations.
    fourth is then computed to have a norm of 1 -- size = [B x 1 x 3]
    :return: Rotation matrix corresponding to the quaternion -- size = [B x 4 x 4]
    """
    batch_size = quat.size(0)
    norm_quat = torch.cat([quat[:, :, :1] * 0 + 1, quat], dim=2)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=2, keepdim=True)
    w, x, y, z = norm_quat[:, :, 0], norm_quat[:, :, 1], norm_quat[:, :, 2], norm_quat[:, :, 3]
    zeros = z * 0
    ones = zeros + 1

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, zeros,
                           2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, zeros,
                           2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2, zeros,
                           zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)
    return rot_mat
