"""
Detection decoder model for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

from models.detection_decoder import YoloDecoder, YOLOLayer


class YOLOLayerDeployment(YOLOLayer):
    """Detection layer"""

    def _post_proccess_output_target(self, targets):
        return self.output, dict(), dict()


class YoloDecoderDeployment(YoloDecoder):
    def _final_layer(self):
        # Define final layer
        self.final_layer0 = YOLOLayerDeployment(self.args.anchors1, self.args)
        self.final_layer1 = YOLOLayerDeployment(self.args.anchors2, self.args)
        self.final_layer2 = YOLOLayerDeployment(self.args.anchors3, self.args)
