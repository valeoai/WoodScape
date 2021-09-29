"""
Loss function for 2D Object Detection for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

from torch import nn


class ObjectDetectionLoss(nn.Module):
    """This criterion combines the object detection losses."""

    def __init__(self, config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.loss = dict(x=0, y=0, w=0, h=0, theta=0, obj_conf=0, no_obj_conf=0, conf=0, cls=0, total_loss=0)
        self.config = config
        self.object_scale = 1
        self.no_object_scale = 100

    def forward(self, outputs, targets):
        """
        Compute the detection losses (localization loss, confidence loss, class loss, yaw loss)
        :param outputs: model predictions
        :param targets: ground truth
        :return:
        """
        self.loss = dict(x=0, y=0, w=0, h=0, theta=0, obj_conf=0, no_obj_conf=0, conf=0, cls=0, total_loss=0)
        total_loss = 0
        for i in range(len(outputs)):
            output = outputs[i]
            target = targets[i]
            # Handle the negative samples:
            if len(target['x']) == 0:
                self.loss["no_obj_conf"] += self.bce_loss(output["no_obj_conf"], target["no_obj_conf"])
                self.loss["conf"] += self.no_object_scale * self.loss["no_obj_conf"]
            else:
                self.loss["x"] += self.mse_loss(output["x"], target["x"])
                self.loss["y"] += self.mse_loss(output["y"], target["y"])
                self.loss["w"] += self.mse_loss(output["w"], target["w"])
                self.loss["h"] += self.mse_loss(output["h"], target["h"])
                self.loss["obj_conf"] += self.bce_loss(output["obj_conf"], target["obj_conf"])
                self.loss["no_obj_conf"] += self.bce_loss(output["no_obj_conf"], target["no_obj_conf"])
                self.loss["conf"] += self.object_scale * self.loss["obj_conf"] + self.no_object_scale * self.loss[
                    "no_obj_conf"]
                self.loss["cls"] += self.bce_loss(output["cls"], target["cls"])

        for loss_name, loss_value in self.loss.items():
            total_loss += loss_value
        self.loss["detection_loss"] = total_loss

        return self.loss
