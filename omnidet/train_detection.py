"""
2D detection training for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

import cv2
from colorama import Fore, Style
from torch.utils.data import DataLoader

from data_loader.woodscape_loader import WoodScapeRawDataset
from losses.detection_loss import ObjectDetectionLoss
from models.detection_decoder import YoloDecoder
from models.resnet import ResnetEncoder
from train_utils.detection_utils import *
from train_utils.detection_utils import log_metrics
from utils import TrainUtils


class DetectionModelBase(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        self.logs = dict()
        # --- Init Detection model ---
        self.models["encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
        self.models["detection"] = YoloDecoder(self.models["encoder"].num_ch_enc, self.args).to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["detection"].parameters())

        if args.use_multiple_gpu:
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            self.encoder_channels = self.models["encoder"].module.num_ch_enc
        else:
            self.encoder_channels = self.models["encoder"].num_ch_enc

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
                                       drop_last=True,
                                       collate_fn=train_dataset.collate_fn)

        val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                          path_file=args.val_file,
                                          is_train=False,
                                          config=args)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True,
                                     collate_fn=val_dataset.collate_fn)

        print(f"=> Total number of training examples: {len(train_dataset)} \n"
              f"=> Total number of validation examples: {len(val_dataset)}")

        self.num_total_steps = len(train_dataset) // args.batch_size * args.epochs

        self.criterion = ObjectDetectionLoss(config=args)
        self.best_mAP = 0

    def pre_init(self):
        if self.args.pretrained_weights:
            self.load_model()

        self.save_args()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def detection_train(self):
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

                features = self.models["encoder"](inputs["color_aug", 0, 0])

                # Detection decoder return the output of the three YOLO heads
                detection_output = self.models["detection"](features,
                                                            [self.args.input_width, self.args.input_height],
                                                            inputs[("detection_labels", 0)])
                # -- DETECTION LOSSES --
                losses = self.criterion(detection_output["yolo_output_dicts"],
                                        detection_output["yolo_target_dicts"])

                # -- DETECTION LOGS --
                self.logs.update(log_metrics(detection_output["yolo_output_dicts"],
                                             detection_output["yolo_target_dicts"], losses))

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["detection_loss"].backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["detection_loss"].cpu().data, data_loading_time, gpu_time)
                    self.detection_statistics("train")
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    # -- SAVE DETECTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                    self.save_best_detection_weights()
                    self.detection_statistics("val")

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0:
                self.save_model()

        print("Training complete!")

    @torch.no_grad()
    def detection_val(self, iou_thres, conf_thres, nms_thres, img_size):
        self.set_eval()

        labels, sample_metrics = (list(), list())  # List of tuples (TP, confs, pred)
        for batch_idx, inputs in enumerate(self.val_loader):
            self.inputs_to_device(inputs)

            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["detection"](features, img_dim=[img_size[0], img_size[1]])["yolo_outputs"]

            # Extract labels
            targets = inputs[("detection_labels", 0)].cpu()
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:6] = xywh2xyxy(targets[:, 2:6])
            targets[:, 2] *= img_size[0]
            targets[:, 3] *= img_size[1]
            targets[:, 4] *= img_size[0]
            targets[:, 5] *= img_size[1]

            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres, args=self.args)
            if batch_idx == 0:
                self.logs["input"] = inputs["color_aug", 0, 0]
                self.logs["outputs"] = outputs

        # Handle the case of empty sample_metrics:
        if len(sample_metrics) == 0:
            # Compute stats for the tensorboard
            self.logs["precision"] = 0
            self.logs["recall"] = 0
            self.logs["mAP"] = 0
            self.logs["AP"] = np.zeros(self.args.num_classes_detection)
            self.logs["f1"] = 0
            del inputs, outputs
            self.set_train()
            return np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        self.logs["precision"] = precision.mean()
        self.logs["recall"] = recall.mean()
        self.logs["mAP"] = AP.mean()
        self.logs["AP"] = AP
        self.logs["f1"] = f1.mean()
        del inputs, outputs
        self.set_train()
        return precision, recall, AP, f1, ap_class

    def save_best_detection_weights(self):
        # 2D Detection validation on each step and save model on improvements.
        precision, recall, AP, f1, ap_class = self.detection_val(iou_thres=0.5,
                                                                 conf_thres=0.8,
                                                                 nms_thres=0.2,
                                                                 img_size=[self.args.input_width,
                                                                           self.args.input_height])
        if AP.mean() > self.best_mAP:
            print(f"{Fore.BLUE}=> Saving detection model weights with mean_AP of {AP.mean():.3f} "
                  f"at step {self.step} on {self.epoch} epoch.{Style.RESET_ALL}")
            rounded_AP = [round(num, 3) for num in AP]
            print(f"{Fore.BLUE}=> meanAP per class in order: {rounded_AP}{Style.RESET_ALL}")
            self.best_mAP = AP.mean()
            self.save_model()
        print(f"{Fore.BLUE}=> Detection val mAP {AP.mean():.3f}{Style.RESET_ALL}")

    def detection_statistics(self, mode) -> None:
        writer = self.writers[mode]
        if mode == "val":
            for i, ap in enumerate(self.logs["AP"]):
                writer.add_scalar(f"detection/AP/{self.args.classes_names[i]}", ap, self.step)

            writer.add_scalar(f"detection/precision", self.logs["precision"], self.step)
            writer.add_scalar(f"detection/recall", self.logs["recall"], self.step)
            writer.add_scalar(f"detection/mAP", self.logs["mAP"], self.step)
            writer.add_scalar(f"detection/f1", self.logs["f1"], self.step)

            """  Logging the images with bounding boxes """
            for j in range(min(4, self.args.batch_size)):  # write maximum of four images
                img = self.logs["input"][j].data
                img_cpu = img.cpu().detach().numpy()
                img_cpu = np.transpose(img_cpu, (1, 2, 0))
                img = np.zeros(img_cpu.shape, img_cpu.dtype)
                img[:, :, :] = img_cpu[:, :, :]
                outputs = self.logs["outputs"][j]
                if outputs is not None:
                    for output in outputs:
                        # (x1, y1, x2, y2)
                        box = [output[0].item(), output[1].item(), output[2].item(), output[3].item()]
                        yaw = 0
                        # rotate the boxes:
                        box = get_contour(box, yaw).exterior.coords
                        boxs = np.int0(box)[0:4]
                        box = []
                        for b in boxs:
                            box.append([b[0], b[1]])
                        box = np.int0(box)
                        cv2.drawContours(img, [box], 0, (0, 1 * np.max(img), 0), 1)

                writer.add_image(f"detection_pred_0/{j}", np.swapaxes(np.swapaxes(img, 1, 2), 0, 1), self.step)
        else:
            writer.add_scalar(f"detection_losses/Total_losses", self.logs["detection_loss"], self.step)
            writer.add_scalar(f"detection_losses/X", self.logs["x"], self.step)
            writer.add_scalar(f"detection_losses/Y", self.logs["y"], self.step)
            writer.add_scalar(f"detection_losses/W", self.logs["w"], self.step)
            writer.add_scalar(f"detection_losses/H", self.logs["h"], self.step)
            writer.add_scalar(f"detection_losses/Confidence", self.logs["conf"], self.step)
            writer.add_scalar(f"detection_losses/Class", self.logs["cls"], self.step)
            writer.add_scalar(f"detection_losses/Precision", self.logs["precision"], self.step)
            writer.add_scalar(f"detection_losses/Recall 50", self.logs["recall50"], self.step)
            writer.add_scalar(f"detection_losses/Recall 75", self.logs["recall75"], self.step)


class DetectionModel(DetectionModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.configure_optimizers()
        self.pre_init()
