"""
Semantic segmentation training for OmniDet.

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
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import TrainUtils, semantic_color_encoding, IoU


class SemanticInit(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        semantic_class_weights = dict(
            woodscape_enet=([3.25, 2.33, 20.42, 30.59, 38.4, 45.73, 10.76, 34.16, 44.3, 49.19]),
            woodscape_mfb=(0.04, 0.03, 0.43, 0.99, 2.02, 4.97, 0.17, 1.01, 3.32, 20.35))

        print(f"=> Setting Class weights based on: {args.semantic_class_weighting} \n"
              f"=> {semantic_class_weights[args.semantic_class_weighting]}")

        semantic_class_weights = torch.tensor(semantic_class_weights[args.semantic_class_weighting]).to(args.device)

        # Setup Metrics
        self.metric = IoU(args.semantic_num_classes, args.dataset, ignore_index=None)

        if args.semantic_loss == "cross_entropy":
            self.semantic_criterion = CrossEntropyLoss2d(weight=semantic_class_weights)
        elif args.semantic_loss == "focal_loss":
            self.semantic_criterion = FocalLoss(weight=semantic_class_weights, gamma=2, size_average=True)

        self.best_semantic_iou = 0.0
        self.alpha = 0.5  # to blend semantic predictions with color image
        self.color_encoding = semantic_color_encoding(args)


class SemanticModel(SemanticInit):
    def __init__(self, args):
        super().__init__(args)

        # --- Init model ---
        self.models["encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
        self.models["semantic"] = SemanticDecoder(self.models["encoder"].num_ch_enc,
                                                  n_classes=args.semantic_num_classes).to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["semantic"].parameters())

        if args.use_multiple_gpu:
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            self.models["semantic"] = torch.nn.DataParallel(self.models["semantic"])

        print(f"=> Training on the {self.args.dataset.upper()} dataset \n"
              f"=> Training model named: {self.args.model_name} \n"
              f"=> Models and tensorboard events files are saved to: {self.args.output_directory} \n"
              f"=> Training is using the cuda device id: {self.args.cuda_visible_devices} \n"
              f"=> Loading {self.args.dataset} training and validation dataset")

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
                                       drop_last=False)

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

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def semantic_train(self):
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
                # -- PUSH INPUTS DICT TO DEVICE --
                self.inputs_to_device(inputs)

                features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["semantic"](features)

                losses = dict()
                losses["semantic_loss"] = self.semantic_criterion(outputs["semantic", 0],
                                                                  inputs["semantic_labels", 0, 0])

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["semantic_loss"].backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["semantic_loss"].cpu().data, data_loading_time, gpu_time)
                    self.semantic_statistics("train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                self.step += 1
                before_op_time = time.time()

            # Validate on each step, save model on improvements
            val_metrics = self.semantic_val()
            print(self.epoch, "IoU:", val_metrics["mean_iou"])
            if val_metrics["mean_iou"] >= self.best_semantic_iou:
                print(f"=> Saving model weights with mean_iou of {val_metrics['mean_iou']:.3f} "
                      f"at step {self.step} on {self.epoch} epoch.")
                self.best_semantic_iou = val_metrics["mean_iou"]
                self.save_model()

            self.lr_scheduler.step(val_metrics["mean_iou"])

        print("Training complete!")

    @torch.no_grad()
    def semantic_val(self):
        """Validate the semantic model"""
        self.set_eval()
        losses = dict()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)
            features = self.models["encoder"](inputs["color", 0, 0])
            outputs = self.models["semantic"](features)
            losses["semantic_loss"] = self.semantic_criterion(outputs["semantic", 0], inputs["semantic_labels", 0, 0])
            _, predictions = torch.max(outputs["semantic", 0].data, 1)
            self.metric.add(predictions, inputs["semantic_labels", 0, 0])
        outputs["class_iou"], outputs["mean_iou"] = self.metric.value()

        # Compute stats for the tensorboard
        self.semantic_statistics("val", inputs, outputs, losses)
        self.metric.reset()
        del inputs, losses
        self.set_train()

        return outputs

    def semantic_statistics(self, mode, inputs, outputs, losses) -> None:
        writer = self.writers[mode]
        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)

        if mode == "val":
            writer.add_scalar(f"mean_iou", outputs["mean_iou"], self.step)
            for k, v in outputs["class_iou"].items():
                writer.add_scalar(f"class_iou/{k}", v, self.step)

        writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], self.step)

        for j in range(min(4, self.args.batch_size)):  # write maximum of four images
            if self.args.train == "semantic":
                writer.add_image(f"color/{j}", inputs[("color", 0, 0)][j], self.step)

            # Predictions is one-hot encoded with "num_classes" channels.
            # Convert it to a single int using the indices where the maximum (1) occurs
            _, predictions = torch.max(outputs["semantic", 0][j].data, 0)
            predictions_gray = predictions.byte().squeeze().cpu().detach().numpy()
            color_semantic = np.array(self.trans_pil(inputs[("color", 0, 0)].cpu()[j].data))
            not_background = predictions_gray != 0
            color_semantic[not_background, ...] = (color_semantic[not_background, ...] * (1 - self.alpha) +
                                                   self.color_encoding[predictions_gray[not_background]] * self.alpha)
            writer.add_image(f"semantic_pred_0/{j}", color_semantic.transpose(2, 0, 1), self.step)

            labels = inputs["semantic_labels", 0, 0][j].data
            labels_gray = labels.byte().squeeze().cpu().detach().numpy()
            labels_rgb = np.array(self.trans_pil(inputs[("color", 0, 0)].cpu()[j].data))
            not_background = labels_gray != 0
            labels_rgb[not_background, ...] = (labels_rgb[not_background, ...] * (1 - self.alpha) +
                                               self.color_encoding[labels_gray[not_background]] * self.alpha)
            writer.add_image(f"semantic_labels_0/{j}", labels_rgb.transpose(2, 0, 1), self.step)
