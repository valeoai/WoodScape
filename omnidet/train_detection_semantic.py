"""
2D detection and Semantic segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

from colorama import Fore, Style

from losses.mtl_losses import UncertaintyLoss
from models.semantic_decoder import SemanticDecoder
from train_detection import DetectionModelBase
from train_semantic import SemanticModel, SemanticInit
from train_utils.detection_utils import *
from train_utils.detection_utils import log_metrics


class DetectionSemanticModel(DetectionModelBase, SemanticInit):
    def __init__(self, args):
        super().__init__(args)

        self.models["semantic"] = SemanticDecoder(self.encoder_channels,
                                                  n_classes=args.semantic_num_classes).to(self.device)

        self.parameters_to_train += list(self.models["semantic"].parameters())

        if args.use_multiple_gpu:
            self.models["semantic"] = torch.nn.DataParallel(self.models["semantic"])

        self.mtl_loss = UncertaintyLoss(tasks=self.args.train).to(self.device)
        self.parameters_to_train += list(self.mtl_loss.parameters())
        self.configure_optimizers()
        self.pre_init()

    def detection_semantic_train(self):
        """Trainer function for detection and semantic prediction"""

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

                # -- DETECTION AND SEMANTIC SEGMENTATION MODEL PREDICTIONS AND LOSS CALCULATIONS --
                features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["semantic"](features)

                # Detection decoder return the output of the three YOLO heads
                outputs.update(self.models["detection"](features,
                                                        [self.args.input_width, self.args.input_height],
                                                        inputs[("detection_labels", 0)]))
                # -- DETECTION LOSSES --
                losses = dict()
                detection_losses = self.criterion(outputs["yolo_output_dicts"],
                                                  outputs["yolo_target_dicts"])

                losses.update(dict(detection_loss=detection_losses['detection_loss']))

                # -- SEMANTIC LOSSES --
                losses["semantic_loss"] = self.semantic_criterion(outputs["semantic", 0],
                                                                  inputs["semantic_labels", 0, 0])

                # -- DETECTION LOGS --
                self.logs.update(log_metrics(outputs["yolo_output_dicts"],
                                             outputs["yolo_target_dicts"], detection_losses))

                losses["mtl_loss"] = self.mtl_loss(losses)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["mtl_loss"].mean().backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mtl_loss"].mean().cpu().data, data_loading_time,
                                  gpu_time)
                    SemanticModel.semantic_statistics(self, "train", inputs, outputs, losses)
                    self.detection_statistics("train")
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    # -- SAVE SEMANTIC MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_semantic_weights()
                    # -- SAVE DETECTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                    self.save_best_detection_weights()

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0 and False:
                self.save_model()

        print("Training complete!")

    @torch.no_grad()
    def semantic_val(self):
        """Validate the semantic model"""
        self.set_eval()

        losses = dict()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["semantic"](features)
            losses["semantic_loss"] = self.semantic_criterion(outputs["semantic", 0], inputs["semantic_labels", 0, 0])
            _, predictions = torch.max(outputs["semantic", 0].data, 1)
            self.metric.add(predictions, inputs["semantic_labels", 0, 0])
        outputs["class_iou"], outputs["mean_iou"] = self.metric.value()

        # Compute stats for the tensorboard
        SemanticModel.semantic_statistics(self, "val", inputs, outputs, losses)

        self.metric.reset()
        del inputs, losses
        self.set_train()
        return outputs

    def save_best_semantic_weights(self):
        val_metrics = self.semantic_val()
        print(f"{Fore.RED}epoch {self.epoch:>3} | Semantic IoU: {val_metrics['mean_iou']:.3f}{Style.RESET_ALL}")
        if val_metrics["mean_iou"] >= self.best_semantic_iou:
            print(f"{Fore.RED}=> Saving semantic segmentation model weights with mean_iou of"
                  f" {val_metrics['mean_iou']:.3f} at step {self.step} on {self.epoch} epoch.{Style.RESET_ALL}")
            self.best_semantic_iou = val_metrics["mean_iou"]
            if self.epoch > 50:  # Weights are quite large! Sometimes, life is a compromise.
                self.save_model()
