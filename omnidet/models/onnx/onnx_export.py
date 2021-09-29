#!/usr/bin/env python3
"""
ONNX export of perception models for OmniDet.

#usage: ./onnx_export.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os

import onnxmltools
import torch
import torch.nn as nn
from pytorch_model_summary import summary

from main import collect_tupperware
from models.detection_decoder_inference import YoloDecoderDeployment
from models.normnet_decoder import NormDecoder
from models.posenet import PoseDecoder
from models.motion_decoder import MotionDecoder
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder


class Encoder:
    def __init__(self, args, input_images):
        super().__init__()

        num_input_images = 1 if args.siamese_net else 2
        self.encoder = ResnetEncoder(args.network_layers,
                                     pretrained=False,
                                     num_input_images=num_input_images).to(args.device)

        if args.init_weights:
            loaded_encoder_dict = torch.load(os.path.join(args.model_path, "encoder.pth"), map_location=args.device)
            filtered_dict_enc = {k: v for k, v in loaded_encoder_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(filtered_dict_enc)
            self.encoder.eval()
            print(f"=> Successfully loaded ResNet{args.network_layers} encoder")

        if args.model_summary:
            # Set max_depth=2 for shorter summary
            print(summary(self.encoder, input_images[1],
                          show_input=True, show_hierarchical=True, print_summary=True,
                          max_depth=None, show_parent_layers=False),
                  file=open(os.path.join(args.onnx_export_path, f"encoder_params.txt"), "w"))


class NormNet(Encoder, nn.Module):
    def __init__(self, args, input_images):
        super().__init__(args, input_images)
        self.args = args

        self.norm_decoder = NormDecoder(self.encoder.num_ch_enc, num_output_channels=1).to(args.device)

        if args.init_weights:
            loaded_norm_dict = torch.load(os.path.join(args.model_path, "norm.pth"), map_location=args.device)
            self.norm_decoder.load_state_dict(loaded_norm_dict)
            self.norm_decoder.eval()
            print(f"=> Successfully loaded {args.onnx_model} decoder")

        if args.model_summary:
            print(summary(self.norm_decoder, self.encoder(input_images[1]),
                          show_input=True, show_hierarchical=True,
                          print_summary=True, max_depth=None, show_parent_layers=False),
                  file=open(os.path.join(args.onnx_export_path, f"{args.onnx_model}_params.txt"), "w"))

    def scale_norm(self, norm):
        """Convert network's sigmoid output into norm prediction"""
        return self.args.min_distance + self.args.max_distance * norm

    def forward(self, input_images):
        return self.scale_norm(self.norm_decoder(self.encoder(input_images[1]))).squeeze(0)


class PoseNet(Encoder, nn.Module):
    def __init__(self, args, input_images):
        super().__init__(args, input_images)
        self.pose_decoder = PoseDecoder(self.encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=2).to(args.device)

        if args.init_weights:
            loaded_pose_dict = torch.load(os.path.join(args.pose_decoder_path, "pose.pth"), map_location=args.device)
            self.pose_decoder.load_state_dict(loaded_pose_dict)
            self.pose_decoder.eval()
            print("=> Successfully loaded PoseNet decoder")

        if args.model_summary:
            print(summary(self.pose_decoder, [self.encoder(input_images[0])],
                          show_input=True, show_hierarchical=True,
                          print_summary=True, max_depth=None, show_parent_layers=False),
                  file=open(os.path.join(args.onnx_export_path, f"{args.onnx_model}_params.txt"), "w"))

    def forward(self, input_images):
        return self.pose_decoder([self.encoder(input_images[1])])


class SemanticNet(Encoder, nn.Module):
    def __init__(self, args, input_images):
        super().__init__(args, input_images)

        self.semantic_decoder = SemanticDecoder(self.encoder.num_ch_enc,
                                                n_classes=args.semantic_num_classes).to(args.device)
        if args.init_weights:
            loaded_semantic_dict = torch.load(os.path.join(args.model_path, "semantic.pth"), map_location=args.device)
            self.semantic_decoder.load_state_dict(loaded_semantic_dict)
            self.semantic_decoder.eval()
            print("=> Successfully loaded Semantic decoder")

        if args.model_summary:
            print(summary(self.semantic_decoder, self.encoder(input_images[1]),
                          show_input=True, show_hierarchical=True, print_summary=True, max_depth=None,
                          show_parent_layers=False),
                  file=open(os.path.join(args.onnx_export_path, f"{args.onnx_model}_params.txt"), "w"))

    def forward(self, input_images):
        return self.semantic_decoder(self.encoder(input_images[1])).squeeze(0)


class MotionNet(Encoder, nn.Module):
    def __init__(self, args, input_images):
        super().__init__(args, input_images)
        self.args = args

        self.motion_decoder = MotionDecoder(self.encoder.num_ch_enc,
                                            n_classes=2,
                                            siamese_net=args.siamese_net).to(args.device)

        if args.init_weights:
            loaded_motion_dict = torch.load(os.path.join(args.model_path, "motion.pth"), map_location=args.device)
            self.motion_decoder.load_state_dict(loaded_motion_dict)
            self.motion_decoder.eval()
            print("=> Successfully loaded Motion decoder")

        if self.args.model_summary:
            if args.siamese_net:
                features_0 = self.encoder(input_images[0])
                features_1 = self.encoder(input_images[1])
                features = [[torch.cat([i, j], dim=1) for i, j in zip(features_0, features_1)]]
            else:
                features = [self.encoder(input_images[1])]

            print(summary(self.motion_decoder, features[0], show_input=True,
                          show_hierarchical=True, print_summary=True, max_depth=None, show_parent_layers=False),
                  file=open(os.path.join(args.onnx_export_path, f"{args.onnx_model}_params.txt"), "w"))

    def forward(self, input_images):
        if self.args.siamese_net:
            features_0 = self.encoder(input_images[0])
            features_1 = self.encoder(input_images[1])
            features = [[torch.cat([i, j], dim=1) for i, j in zip(features_0, features_1)]]
        else:
            features = [self.encoder(input_images[1])]

        return self.motion_decoder(features[0]).squeeze(0)


class DetectionNet(Encoder, nn.Module):
    def __init__(self, args, input_images):
        super().__init__(args, input_images)
        self.args = args
        self.detection_decoder = YoloDecoderDeployment(self.encoder.num_ch_enc, args).to(args.device)

        if args.init_weights:
            loaded_detection_dict = torch.load(os.path.join(args.model_path, "detection.pth"), map_location=args.device)
            self.detection_decoder.load_state_dict(loaded_detection_dict, strict=True)
            self.detection_decoder.eval()
            print("=> Successfully loaded Detection decoder")

        if args.model_summary:
            img_shape = torch.tensor([args.input_width, args.input_height])
            inputs = [self.encoder(input_images[1]), img_shape]
            print(summary(self.detection_decoder, *inputs, show_input=True,
                          show_hierarchical=True, print_summary=True, max_depth=None, show_parent_layers=False),
                  file=open(os.path.join(args.onnx_export_path, f"{args.onnx_model}_params.txt"), "w"))

    def forward(self, input_images):
        img_shape = torch.tensor([self.args.input_width, self.args.input_height])
        return self.detection_decoder(self.encoder(input_images[1]), img_shape)


class FisheyePerception(Encoder, nn.Module):
    def __init__(self, args, input_images):
        super().__init__(args, input_images)
        self.outputs = []
        self.args = args

        assert args.siamese_net is True  # Assertion is for woodscape requirements
        assert args.semantic_num_classes == 10

        self.norm_decoder = NormDecoder(self.encoder.num_ch_enc, num_output_channels=1).to(args.device)

        self.semantic_decoder = SemanticDecoder(self.encoder.num_ch_enc,
                                                n_classes=args.semantic_num_classes).to(args.device)

        self.motion_decoder = MotionDecoder(self.encoder.num_ch_enc,
                                            n_classes=2,
                                            siamese_net=args.siamese_net).to(args.device)

        self.detection_decoder = YoloDecoderDeployment(self.encoder.num_ch_enc, args).to(args.device)

        if args.init_weights:
            loaded_norm_dict = torch.load(os.path.join(args.model_path, "norm.pth"), map_location=args.device)
            self.norm_decoder.load_state_dict(loaded_norm_dict)
            self.norm_decoder.eval()
            print("=> Successfully loaded Distance decoder")

            loaded_semantic_dict = torch.load(os.path.join(args.model_path, "semantic.pth"), map_location=args.device)
            self.semantic_decoder.load_state_dict(loaded_semantic_dict)
            self.semantic_decoder.eval()
            print("=> Successfully loaded Semantic decoder")

            loaded_motion_dict = torch.load(os.path.join(args.model_path, "motion.pth"), map_location=args.device)
            self.motion_decoder.load_state_dict(loaded_motion_dict)
            self.motion_decoder.eval()
            print("=> Successfully loaded Motion decoder")

            loaded_detection_dict = torch.load(os.path.join(args.model_path, "detection.pth"), map_location=args.device)
            self.detection_decoder.load_state_dict(loaded_detection_dict, strict=True)
            self.detection_decoder.eval()
            print("=> Successfully loaded Detection decoder")

    def scale_norm(self, norm):
        """Convert network's sigmoid output into norm prediction"""
        return self.args.min_distance + self.args.max_distance * norm

    def forward(self, input_images):
        # Obtain features from encoder
        if self.args.siamese_net:
            features_0 = self.encoder(input_images[0])  # t-1
        features_1 = self.encoder(input_images[1])  # t
        self.outputs.append(self.scale_norm(self.norm_decoder(features_1)).squeeze(0))  # Estimates from Norm Decoder
        self.outputs.append(self.semantic_decoder(features_1).squeeze(0))  # Estimates from Semantic Decoder
        if self.args.siamese_net:
            features = [[torch.cat([i, j], dim=1) for i, j in zip(features_0, features_1)]]
        else:
            features = [self.encoder(input_images)]
        self.outputs.append(self.motion_decoder(features[0]).squeeze(0))  # Estimates from Motion Decoder
        img_shape = torch.tensor([self.args.input_width, self.args.input_height])
        self.outputs.append(self.detection_decoder(features_1, img_shape))
        return self.outputs


if __name__ == "__main__":
    args = collect_tupperware()

    if args.siamese_net:  # Due to motion decoder t-1 and t
        inputs = [torch.randn(1, 3, args.input_height, args.input_width).to(args.device),
                  torch.randn(1, 3, args.input_height, args.input_width).to(args.device)]
    else:
        inputs = [torch.cat([torch.randn(1, 3, args.input_height, args.input_width).to(args.device),
                             torch.randn(1, 3, args.input_height, args.input_width).to(args.device)], 1)]

    if args.onnx_model == "normnet":
        model = NormNet(args, inputs)
    elif args.onnx_model == "posenet":
        assert args.siamese_net is False  # Since it is not implemented for PoseNet
        model = PoseNet(args, inputs)
    elif args.onnx_model == "semantic":
        model = SemanticNet(args, inputs)
    elif args.onnx_model == "motion":
        model = MotionNet(args, inputs)
    elif args.onnx_model == "detection":
        model = DetectionNet(args, inputs)
    elif "omnidet" in args.onnx_model:
        model = FisheyePerception(args, inputs)
    else:
        raise NotImplementedError

    onnx_export_dir = os.path.join(args.onnx_export_path, args.model_name)
    if not os.path.isdir(onnx_export_dir):
        os.makedirs(onnx_export_dir)
    model_path = os.path.join(onnx_export_dir, f"{args.onnx_model}_float32_opset{args.opset_version}.onnx")

    torch.onnx.export(model, inputs, model_path,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                      export_params=True, verbose=False, opset_version=args.opset_version)

    # Conversion of float32 bit to float16 model
    onnx_model = onnxmltools.utils.load_model(model_path)
    onnx_model = onnxmltools.utils.float16_converter.convert_float_to_float16(onnx_model)
    float16_model_path = os.path.join(onnx_export_dir, f"{args.onnx_model}_float16_opset{args.opset_version}.onnx")
    onnxmltools.utils.save_model(onnx_model, float16_model_path)

    # onnx_model = onnx.load(model_path)
    # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    # optimized_model = optimizer.optimize(onnx_model, passes)
    # onnx.save(optimized_model, model_path)

    print(f"=> ONNX Export of {args.onnx_model} successful")
    print(f"=> ONNX Float32 bit model exported to {model_path} \n"
          f"=> ONNX Float16 bit model exported to {float16_model_path}")
