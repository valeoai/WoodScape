"""
Distance estimation, Semantic segmentation, 2D detection and Motion segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

from colorama import Fore, Style

from losses.mtl_losses import UncertaintyLoss
from train_distance_semantic_detection import DistanceSemanticDetectionModelBase
from train_distance_semantic_motion import DistanceSemanticMotionModelBase
from train_semantic import SemanticModel
from train_motion import MotionModel
from train_detection import DetectionModel


class DistanceSemanticDetectionMotionModel(DistanceSemanticDetectionModelBase, DistanceSemanticMotionModelBase):
    def __init__(self, args):
        super().__init__(args)

        self.mtl_loss = UncertaintyLoss(tasks=self.args.train).to(self.device)
        self.parameters_to_train += list(self.mtl_loss.parameters())
        self.configure_optimizers()
        self.pre_init()

    def distance_semantic_detection_motion_train(self):
        """Trainer function for distance, semantic, detection and motion prediction"""

        print(f"{Fore.BLUE}=> Initial mAP for detection task: 0{Style.RESET_ALL}")

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

                # -- DISTANCE, SEMANTIC, OBJECT DETECTION AND MOTION SEG MODEL PREDICTIONS AND LOSS CALCULATIONS --
                outputs, losses = self.distance_semantic_detection_motion_loss_predictions(inputs)

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
                    DetectionModel.detection_statistics(self, "train")
                    MotionModel.motion_statistics(self, "train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    # -- SAVE SEMANTIC MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_semantic_weights()
                    # -- SAVE DETECTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                    self.save_best_detection_weights()
                    DetectionModel.detection_statistics(self, "val")
                    # -- SAVE MOTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_motion_weights()

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0:
                self.save_model()

        print("Training complete!")

    def distance_semantic_detection_motion_loss_predictions(self, inputs):
        features, outputs, losses = self.distance_semantic_motion_loss_predictions(inputs)
        # Note: We are taking features passed through encoder when the dataset split for all the tasks is same
        detection_outputs, detection_losses = self.predict_detection(inputs, outputs, features=features)
        outputs.update(detection_outputs)
        losses.update(detection_losses)
        return outputs, losses
