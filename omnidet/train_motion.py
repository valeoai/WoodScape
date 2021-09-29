"""
Motion segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader.woodscape_loader import WoodScapeRawDataset
from losses.semantic_loss import CrossEntropyLoss2d, FocalLoss
from models.motion_decoder import MotionDecoder
from models.resnet import ResnetEncoder
from utils import TrainUtils, IoU


class MotionInit(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        motion_class_weights = dict(motion_enet=[1.45, 23.36], motion_mfb=[0.51, 21.07])

        print(f"=> Setting Class weights based on: {args.motion_class_weighting} \n"
              f"=> {motion_class_weights[args.motion_class_weighting]}")

        motion_class_weights = torch.tensor(motion_class_weights[args.motion_class_weighting]).to(self.device)

        # Setup Metrics
        self.best_motion_iou = 0.0
        self.motion_metric = IoU(2, 'motion', ignore_index=None)

        if args.motion_loss == "cross_entropy":
            self.motion_criterion = CrossEntropyLoss2d(weight=motion_class_weights)
        elif args.motion_loss == "focal_loss":
            self.motion_criterion = FocalLoss(weight=motion_class_weights, gamma=2, size_average=True)

        motion_classes = dict(static=(0, 0, 0), motion=(255, 0, 0))
        self.motion_color_encoding = np.zeros((2, 3), dtype=np.uint8)
        for i, (k, v) in enumerate(motion_classes.items()):
            self.motion_color_encoding[i] = v


class MotionModel(MotionInit):
    def __init__(self, args):
        super().__init__(args)

        # --- Init model ---
        self.models["encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
        self.models["motion"] = MotionDecoder(self.models["encoder"].num_ch_enc,
                                              n_classes=2,
                                              siamese_net=self.args.siamese_net).to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["motion"].parameters())

        if args.use_multiple_gpu:
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            self.models["motion"] = torch.nn.DataParallel(self.models["motion"])

        print(f"=> Training on the {args.dataset.upper()} dataset \n"
              f"=> Training model named: {args.model_name} \n"
              f"=> Models and tensorboard events files are saved to: {args.output_directory} \n"
              f"=> Training is using the cuda device id: {args.cuda_visible_devices}  \n"
              f"=> Loading {args.dataset} training and validation dataset")

        # --- Load Data ---
        train_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                            path_file=args.train_file,
                                            is_train=True,
                                            config=args)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=True)

        val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                          path_file=args.val_file,
                                          is_train=False,
                                          config=args)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True)

        print(f"=> Total number of training examples: {len(train_dataset)} \n"
              f"=> Total number of validation examples: {len(val_dataset)}")

        self.num_total_steps = len(train_dataset) // args.batch_size * args.epochs
        self.configure_optimizers()

        if args.pretrained_weights:
            self.load_model()

        self.save_args()
        self.alpha = 0.5

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def motion_train(self):
        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                current_time = time.time()
                data_loading_time += (current_time - before_op_time)
                before_op_time = current_time
                self.inputs_to_device(inputs)

                outputs = self.predict_motion_seg(inputs)
                losses = dict()
                losses["motion_loss"] = self.motion_criterion(outputs["motion", 0], inputs["motion_labels", 0, 0])

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["motion_loss"].backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["motion_loss"].cpu().data, data_loading_time, gpu_time)
                    self.motion_statistics("train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                self.step += 1
                before_op_time = time.time()

            # Validate on each step, save model on improvements
            val_metrics = self.motion_val()
            print(self.epoch, "IoU:", val_metrics["mean_iou"])
            if val_metrics["mean_iou"] >= self.best_motion_iou:
                print(f"=> Saving model weights with mean_iou of {val_metrics['mean_iou']:.3f} "
                      f"at step {self.step} on {self.epoch} epoch.")
                self.best_motion_iou = val_metrics["mean_iou"]
                self.save_model()

            self.lr_scheduler.step(val_metrics["mean_iou"])

        print("Training complete!")

    def predict_motion_seg(self, inputs):
        outputs = dict()
        if self.args.siamese_net:
            previous_frames = self.models["encoder"](inputs["color_aug", -1, 0])
            current_frames = self.models["encoder"](inputs["color_aug", 0, 0])
            features = [torch.cat([i, j], dim=1) for i, j in zip(previous_frames, current_frames)]
            outputs.update(self.models["motion"](features))
        else:
            motion_inputs = torch.cat([inputs["color_aug", -1, 0], inputs["color_aug", 0, 0]], 1)
            features = self.models["encoder"](motion_inputs)
            outputs.update(self.models["motion"](features))
        return outputs

    @torch.no_grad()
    def motion_val(self):
        """Validate the motion model"""
        self.set_eval()
        losses = dict()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)
            outputs = self.predict_motion_seg(inputs)
            losses["motion_loss"] = self.motion_criterion(outputs["motion", 0], inputs["motion_labels", 0, 0])
            _, predictions = torch.max(outputs["motion", 0].data, 1)
            self.motion_metric.add(predictions, inputs["motion_labels", 0, 0])
        outputs["class_iou"], outputs["mean_iou"] = self.motion_metric.value()
        # Compute stats for the tensorboard
        self.motion_statistics("val", inputs, outputs, losses)
        self.motion_metric.reset()
        del inputs, losses
        self.set_train()
        return outputs

    def motion_statistics(self, mode, inputs, outputs, losses) -> None:
        writer = self.writers[mode]
        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)

        if mode == "val":
            writer.add_scalar(f"mean_iou", outputs["mean_iou"], self.step)
            for k, v in outputs["class_iou"].items():
                writer.add_scalar(f"class_iou/{k}", v, self.step)

        for j in range(min(4, self.args.batch_size)):  # write maximum of two image pairs
            for i in self.args.motion_frame_idxs:
                if self.args.train == "motion":
                    writer.add_image(f"color_motion_{i}/{j}", inputs[("color", i, 0)][j].data, self.step)

            labels = inputs["motion_labels", 0, 0][j].data
            labels_gray = labels.byte().squeeze().cpu().detach().numpy()
            labels_rgb = np.array(self.trans_pil(inputs[("color", 0, 0)].cpu()[j].data))
            not_background = labels_gray != 0
            labels_rgb[not_background, ...] = (labels_rgb[not_background, ...] * (1 - self.alpha) +
                                               self.motion_color_encoding[labels_gray[not_background]] * self.alpha)
            writer.add_image(f"motion_labels_0/{j}", labels_rgb.transpose(2, 0, 1), self.step)

            _, predictions = torch.max(outputs["motion", 0][j].data, 0)
            predictions_gray = predictions.byte().squeeze().cpu().detach().numpy()
            color_motion = np.array(self.trans_pil(inputs[("color", 0, 0)].cpu()[j].data))
            not_background = predictions_gray != 0
            color_motion[not_background, ...] = (color_motion[not_background, ...] * (1 - self.alpha) +
                                                 self.motion_color_encoding[
                                                     predictions_gray[not_background]] * self.alpha)
            writer.add_image(f"motion_pred_0/{j}", color_motion.transpose(2, 0, 1), self.step)
