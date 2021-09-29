"""
Distance estimation, Semantic segmentation and Motion segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

import torch
from colorama import Fore, Style

from losses.mtl_losses import UncertaintyLoss
from models.motion_decoder import MotionDecoder
from train_distance_semantic import DistanceSemanticModelBase
from train_motion import MotionModel, MotionInit
from train_semantic import SemanticModel


class DistanceSemanticMotionModelBase(DistanceSemanticModelBase, MotionInit):
    def __init__(self, args):
        super().__init__(args)

        self.models["motion"] = MotionDecoder(self.encoder_channels,
                                              n_classes=2,
                                              siamese_net=self.args.siamese_net).to(self.device)

        self.parameters_to_train += list(self.models["motion"].parameters())

        if args.use_multiple_gpu:
            self.models["motion"] = torch.nn.DataParallel(self.models["motion"])

    def distance_semantic_motion_train(self):
        """Trainer function for distance, semantic and motion prediction"""

        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()
                self.inputs_to_device(inputs)

                # -- DISTANCE, SEMANTIC AND MOTION SEGMENTATION MODEL PREDICTIONS AND LOSS CALCULATIONS --
                _, outputs, losses = self.distance_semantic_motion_loss_predictions(inputs)

                # -- MTL LOSS --
                losses["mtl_loss"] = self.mtl_loss(losses)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["mtl_loss"].mean().backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mtl_loss"].mean().cpu().data,
                                  data_loading_time, gpu_time)
                    self.distance_statistics("train", inputs, outputs, losses)
                    SemanticModel.semantic_statistics(self, "train", inputs, outputs, losses)
                    MotionModel.motion_statistics(self, "train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0:
                    # -- SAVE SEMANTIC MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_semantic_weights()
                    # -- SAVE MOTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_motion_weights()

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0:
                self.save_model()

        print("Training complete!")

    def distance_semantic_motion_loss_predictions(self, inputs):
        losses = dict()
        # -- SEMANTIC SEGMENTATION --
        outputs, features = self.predict_semantic_seg(inputs)

        # -- MOTION SEGMENTATION --
        motion_predictions = self.predict_motion_seg(inputs, features=features)
        outputs.update(motion_predictions)

        # -- DISTANCE ESTIMATION --
        distance_outputs, features = self.predict_distances(inputs, features=features)
        outputs.update(distance_outputs)
        # -- POSE ESTIMATION --
        outputs.update(self.predict_poses(inputs, features))
        # -- PHOTOMETRIC LOSSES --
        distance_losses, distance_outputs = self.photometric_losses(inputs, outputs)
        losses.update(distance_losses)
        outputs.update(distance_outputs)

        # -- SEMANTIC SEGMENTATION LOSS --
        losses["semantic_loss"] = self.semantic_criterion(outputs["semantic", 0], inputs["semantic_labels", 0, 0])
        if self.args.use_multiple_gpu:
            losses["semantic_loss"] = losses["semantic_loss"].unsqueeze(0)

        # -- MOTION SEGMENTATION LOSS --
        losses["motion_loss"] = self.motion_criterion(outputs["motion", 0], inputs["motion_labels", 0, 0])
        if self.args.use_multiple_gpu:
            losses["motion_loss"] = losses["motion_loss"].unsqueeze(0)

        return features, outputs, losses

    def predict_motion_seg(self, inputs, features=None, mode='val'):
        outputs = dict()
        if self.args.siamese_net:
            previous_frames = self.models["encoder"](inputs["color_aug", -1, 0])
            current_frames = features if mode != 'val' else self.models["encoder"](inputs["color_aug", 0, 0])
            features = [torch.cat([i, j], dim=1) for i, j in zip(previous_frames, current_frames)]
            outputs.update(self.models["motion"](features))
        else:
            features = self.models["encoder"](torch.cat([inputs["color_aug", -1, 0], inputs["color_aug", 0, 0]], 1))
            outputs.update(self.models["motion"](features))
        return outputs

    @torch.no_grad()
    def motion_val(self):
        """Validate the motion model"""
        self.set_eval()

        losses = dict()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)
            outputs = self.predict_motion_seg(inputs, features=None, mode='val')
            losses["motion_loss"] = self.motion_criterion(outputs["motion", 0], inputs["motion_labels", 0, 0])
            if self.args.use_multiple_gpu:
                losses["motion_loss"] = losses["motion_loss"].unsqueeze(0)
            _, predictions = torch.max(outputs["motion", 0].data, 1)
            self.motion_metric.add(predictions, inputs["motion_labels", 0, 0])

        outputs["class_iou"], outputs["mean_iou"] = self.motion_metric.value()

        # Compute stats for the tensorboard
        MotionModel.motion_statistics(self, "val", inputs, outputs, losses)
        self.motion_metric.reset()
        del inputs, losses
        self.set_train()

        return outputs

    def save_best_motion_weights(self):
        # Motion Seg. validation on each step and save model on improvements.
        motion_val_metrics = self.motion_val()
        print(
            f"{Fore.MAGENTA}epoch {self.epoch:>3} | Motion IoU: {motion_val_metrics['mean_iou']:.3f}{Style.RESET_ALL}")
        if motion_val_metrics["mean_iou"] >= self.best_motion_iou:
            print(f"{Fore.MAGENTA}=> Saving motion model weights with mean_iou of {motion_val_metrics['mean_iou']:.3f} "
                  f"at step {self.step} on {self.epoch} epoch.{Style.RESET_ALL}")
            self.best_motion_iou = motion_val_metrics["mean_iou"]
            if self.epoch > 50:  # Weights are quite large! Sometimes, life is a compromise.
                self.save_model()


class DistanceSemanticMotionModel(DistanceSemanticMotionModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.mtl_loss = UncertaintyLoss(tasks=self.args.train).to(self.device)
        self.parameters_to_train += list(self.mtl_loss.parameters())
        if args.use_multiple_gpu:
            self.mtl_loss = torch.nn.DataParallel(self.mtl_loss)
        self.configure_optimizers()
        self.pre_init()
