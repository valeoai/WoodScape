"""
Utilities for for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/nianticlabs/monodepth2
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from ruamel import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class TrainUtils:
    def __init__(self, args):
        """Train Utils class providing training utilities for distance, semantic and motion estimation
        :param args: input params from config file
        """
        self.args = args
        self.device = args.device
        self.log_path = os.path.join(args.output_directory, args.model_name)
        assert args.input_height % 32 == 0, "'height' must be a multiple of 32"
        assert args.input_width % 32 == 0, "'width' must be a multiple of 32"
        self.models = dict()
        self.parameters_to_train = []
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.trans_pil = transforms.ToPILImage()
        self.optimizer = None
        self.lr_scheduler = None

        self.writers = dict()
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

    def inputs_to_device(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

    def set_train(self):
        """Convert all models to training mode"""
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    def log_time(self, batch_idx, duration, loss, data_time, gpu_time):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.args.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print(f"{Fore.GREEN}epoch {self.epoch:>3}{Style.RESET_ALL} "
              f"| batch {batch_idx:>6} "
              f"| current lr {self.optimizer.param_groups[0]['lr']:.4f} "
              f"| examples/s: {samples_per_sec:5.1f} "
              f"| {Fore.RED}loss: {loss:.5f}{Style.RESET_ALL} "
              f"| {Fore.BLUE}time elapsed: {self.sec_to_hm_str(time_sofar)}{Style.RESET_ALL} "
              f"| {Fore.CYAN}time left: {self.sec_to_hm_str(training_time_left)}{Style.RESET_ALL} "
              f"| CPU/GPU time: {data_time:0.1f}s/{gpu_time:0.1f}s")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters_to_train, self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.args.scheduler_step_size)

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}", str(self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.args.input_height
                to_save['width'] = self.args.input_width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "adam.pth")
        if self.epoch > 50:  # Optimizer file is quite large! Sometimes, life is a compromise.
            torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk"""
        self.args.pretrained_weights = os.path.expanduser(self.args.pretrained_weights)

        assert os.path.isdir(self.args.pretrained_weights), f"Cannot find folder {self.args.pretrained_weights}"
        print(f"=> Loading model from folder {self.args.pretrained_weights}")

        for n in self.args.models_to_load:
            print(f"Loading {n} weights...")
            path = os.path.join(self.args.pretrained_weights, f"{n}.pth")
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.args.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading optimizer state
        if not self.args.freeze_encoder:
            optimizer_load_path = os.path.join(self.args.pretrained_weights, f"{self.args.optimizer}.pth")
            if os.path.isfile(optimizer_load_path):
                print(f"Loading {self.args.optimizer} weights")
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.args.device)
                self.optimizer.load_state_dict(optimizer_dict)
            else:
                print(f"Cannot find {self.args.optimizer} weights so {self.args.optimizer} is randomly initialized")

    def save_args(self):
        """Save arguments to disk so we know what we ran this experiment with"""

        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.args.copy()

        with open(os.path.join(models_dir, 'params.yaml'), 'w') as f:
            yaml.dump(to_save, f)

    def sec_to_hm(self, t):
        """Convert time in seconds to time in hours, minutes and seconds
        e.g. 10239 -> (2, 50, 39)
        """
        t = int(t)
        s = t % 60
        t //= 60
        m = t % 60
        t //= 60
        return t, m, s

    def sec_to_hm_str(self, t):
        """Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
        """
        h, m, s = self.sec_to_hm(t)
        return f"{h:02d}h{m:02d}m{s:02d}s"


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    @staticmethod
    def accuracy(preds, label):
        valid = (label >= 0)
        acc_sum = (valid * (preds == label)).sum()
        valid_sum = valid.sum()
        acc = float(acc_sum) / (valid_sum + 1e-10)
        return acc, valid_sum


class Tupperware(dict):
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Tupperware):
            value = Tupperware(value)
        super(Tupperware, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, Tupperware.MARKER)
        if found is Tupperware.MARKER:
            found = Tupperware()
            super(Tupperware, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


class IoU:
    """Computes the intersection over union (IoU) per class and corresponding mean (mIoU).
    The predictions are first accumulated in a confusion matrix and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    :param num_classes (int): number of classes in the classification problem
    :param dataset (string): woodscape_raw
    :param ignore_index (int or iterable, optional): Index of the classes to ignore when computing the IoU.
    """

    def __init__(self, num_classes, dataset, ignore_index=None):
        super().__init__()

        self.conf_metric = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.num_classes = num_classes
        self.dataset = dataset

        self.classes = dict(woodscape_raw=["void", "road", "lanemarks", "curb", "person",
                                           "rider", "vehicles", "bicycle", "motorcycle""traffic_sign"],
                            motion=['static', 'motion'], )

        self.reset()

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.fill(0)

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric."""

        predicted = predicted.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        # hack for bin counting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes ** 2)
        assert bincount_2d.size == self.num_classes ** 2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        self.conf_metric += conf

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns: Tuple: (class_iou, mIoU). The first output is the per class IoU, for K classes it's numpy.ndarray with
        K elements. The second output, is the mean IoU.
        """
        if self.ignore_index is not None:
            for index in self.ignore_index:
                self.conf_metric[:, self.ignore_index] = 0
                self.conf_metric[self.ignore_index, :] = 0
        true_positive = np.diag(self.conf_metric)
        false_positive = np.sum(self.conf_metric, 0) - true_positive
        false_negative = np.sum(self.conf_metric, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        class_dict = self.classes[self.dataset]
        class_iou = dict(zip(class_dict, iou))

        return class_iou, np.nanmean(iou)


def semantic_color_encoding(args):
    semantic_classes = dict(void=(0, 0, 0),
                            road=(149, 213, 0),
                            lanemarks=(216, 45, 128),
                            curb=(0, 140, 88),
                            person=(255, 0, 0),
                            rider=(255, 255, 255),
                            vehicles=(0, 0, 255),
                            bicycle=(0, 255, 255),
                            motorcycle=(30, 170, 250),
                            traffic_sign=(0, 128, 128))

    color_encoding = np.zeros((args.semantic_num_classes, 3), dtype=np.uint8)
    for i, (k, v) in enumerate(semantic_classes.items()):
        color_encoding[i] = v
    return color_encoding
