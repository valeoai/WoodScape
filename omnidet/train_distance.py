"""
Distance Estimation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

import torch
from colorama import Fore, Style
from torch.utils.data import DataLoader

from data_loader.woodscape_loader import WoodScapeRawDataset
from losses.inverse_warp_fisheye import PhotometricFisheye
from losses.distance_loss import PhotometricReconstructionLoss
from models.normnet_decoder import NormDecoder
from models.posenet import PoseDecoder
from models.resnet import ResnetEncoder
from train_utils.distance_utils import tensor2array
from train_utils.pose_utils import pose_vec2mat
from utils import TrainUtils


class DistanceModelBase(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        # --- INIT MODELS ---
        self.models["encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
        self.models["norm"] = NormDecoder(self.models["encoder"].num_ch_enc).to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["norm"].parameters())

        # --- Init Pose model ---
        if args.pose_model_type == "separate":
            # uses the same encoder design as normNet but does not share it
            self.models["pose_encoder"] = ResnetEncoder(num_layers=self.args.pose_network_layers,
                                                        pretrained=True,
                                                        num_input_images=2).to(self.device)

            self.models["pose"] = PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                              num_input_features=1,
                                              num_frames_to_predict_for=2).to(self.device)

            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        elif args.pose_model_type == "shared":
            self.models["pose"] = PoseDecoder(self.models["encoder"].num_ch_enc,
                                              num_input_features=2).to(self.device)

        self.parameters_to_train += list(self.models["pose"].parameters())

        print(f"{Fore.BLUE}=> Training on the {args.dataset.upper()} projection model \n"
              f"=> Training model named: {args.model_name} \n"
              f"=> Models and tensorboard events files are saved to: {args.output_directory} \n"
              f"=> Loading {args.dataset} training and validation dataset{Style.RESET_ALL}")

        # --- Load Data ---
        self.train_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                                 path_file=args.train_file,
                                                 is_train=True,
                                                 config=args)

        collate_train = self.train_dataset.collate_fn if "detection" in self.args.train else None
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=True,
                                       collate_fn=collate_train)

        print(f"{Fore.RED}=> Total number of training examples: {len(self.train_dataset)}{Style.RESET_ALL}")

        val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                          path_file=args.val_file,
                                          is_train=False,
                                          config=args)

        collate_val = val_dataset.collate_fn if "detection" in self.args.train else None
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True,
                                     collate_fn=collate_val)

        self.val_iter = iter(self.val_loader)

        print(f"{Fore.YELLOW}=> Total number of validation examples: {len(val_dataset)}{Style.RESET_ALL}")

        self.num_total_steps = len(self.train_dataset) // args.batch_size * args.epochs

        # --- Parallelize model to multiple GPUs ---
        inverse_warp = PhotometricFisheye(args)

        if args.use_multiple_gpu:
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            self.models["norm"] = torch.nn.DataParallel(self.models["norm"])
            self.models["pose"] = torch.nn.DataParallel(self.models["pose"])
            if self.args.pose_model_type == "separate":
                self.models["pose_encoder"] = torch.nn.DataParallel(self.models["pose_encoder"])
            self.photometric_losses = torch.nn.DataParallel(PhotometricReconstructionLoss(inverse_warp, args))
            self.encoder_channels = self.models["encoder"].module.num_ch_enc
        else:
            self.photometric_losses = PhotometricReconstructionLoss(inverse_warp, args)
            self.encoder_channels = self.models["encoder"].num_ch_enc

    def pre_init(self):
        if self.args.pretrained_weights:
            self.load_model()

        self.save_args()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def distance_train(self):
        """Trainer function for distance and depth prediction on fisheye images"""

        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()
                # -- PUSH INPUTS DICT TO DEVICE --
                self.inputs_to_device(inputs)

                # -- DISTANCE ESTIMATION --
                outputs, features = self.predict_distances(inputs)

                # -- POSE ESTIMATION --
                outputs.update(self.predict_poses(inputs, features))

                # -- PHOTOMETRIC LOSSES --
                losses, outputs = self.photometric_losses(inputs, outputs)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["distance_loss"].mean().backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["distance_loss"].mean().cpu().data,
                                  data_loading_time, gpu_time)
                    self.distance_statistics("train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0:
                self.save_model()

        print("Training complete!")

    def predict_distances(self, inputs, features=None):
        """Predict distances for target frame or for all monocular sequences."""
        outputs = dict()
        if self.args.pose_model_type == "shared":
            # If we are using a shared encoder for both norm and pose,
            # then all images are fed separately through the norm encoder.
            images = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.args.frame_idxs])
            all_features = self.models["encoder"](images)
            all_features = [torch.split(f, self.args.batch_size) for f in all_features]
            features = dict()
            for i, frame_id in enumerate(self.args.frame_idxs):
                features[frame_id] = [f[i] for f in all_features]
            outputs.update(self.models["norm"](features[0]))
        else:
            # Otherwise, we only feed the target image through the norm encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0]) if features is None else features
            outputs.update(self.models["norm"](features))

        return outputs, features

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences."""
        outputs = dict()
        # Compute the pose to each source frame via a separate forward pass through the pose network.
        # select what features the pose network takes as input
        if self.args.pose_model_type == "shared":
            pose_feats = {frame_id: features[frame_id] for frame_id in self.args.frame_idxs}
        else:
            pose_feats = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.args.frame_idxs}

        for frame_id in self.args.frame_idxs[1:]:
            # To maintain ordering we always pass frames in temporal order
            if frame_id == -1:
                pose_inputs = [pose_feats[frame_id], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[frame_id]]

            if self.args.pose_model_type == "separate":
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = self.models["pose"](pose_inputs)

            # Normalize the translation vec and multiply by the displacement magnitude obtained from speed
            # of the vehicle to scale it to the real world translation
            translation_magnitude = translation[:, 0].squeeze(1).norm(p="fro",
                                                                      dim=1).unsqueeze(1).unsqueeze(2)
            translation_norm = translation[:, 0] / translation_magnitude
            translation_norm *= inputs[("displacement_magnitude", frame_id)].unsqueeze(1).unsqueeze(2)
            translation = translation_norm

            outputs[("axisangle", 0, frame_id)] = axisangle
            outputs[("translation", 0, frame_id)] = translation
            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, frame_id)] = pose_vec2mat(axisangle[:, 0],
                                                               translation,
                                                               invert=(frame_id < 0),
                                                               rotation_mode=self.args.rotation_mode)
        return outputs

    def distance_statistics(self, mode, inputs, outputs, losses) -> None:
        """Print the weights and images"""
        writer = self.writers[mode]
        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)
        writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], self.step)

        for j in range(min(4, self.args.batch_size)):  # write maxmimum of four images
            for s in range(self.args.num_scales):
                for frame_id in self.args.frame_idxs:
                    writer.add_image(f"color_{frame_id}/{j}",
                                     inputs[("color", frame_id, 0)][j].data, self.step)
                    if s == 0 and frame_id == 0:
                        writer.add_image(f"inv_norm_{frame_id}_{s}/{j}",
                                         tensor2array(1 / (outputs[("norm", s)][j, 0]), colormap='magma'), self.step)
                if s == 0:
                    writer.add_image(f"color_pred_-1_{s}/{j}",
                                     outputs[("color", -1, s)][j].data, self.step)


class DistanceModel(DistanceModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.configure_optimizers()
        self.pre_init()
