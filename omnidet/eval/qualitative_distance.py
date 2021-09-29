"""
Qualitative test script of distance estimation for OmniDet.

# usage: ./qualitative_distance.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from matplotlib.colors import ListedColormap
from torchvision import transforms

from eval.qualitative_semantic import pre_image_op
from main import collect_args
from models.normnet_decoder import NormDecoder
from models.resnet import ResnetEncoder
from utils import Tupperware

FRAME_RATE = 1


def scale_norm(norm, min_distance, max_distance):
    """Convert network's sigmoid output into distance prediction"""
    return min_distance + max_distance * norm


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higher resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    depth_decoder_path = os.path.join(args.pretrained_weights, "norm.pth")

    print("=> Loading pretrained encoder")
    encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
    loaded_dict_enc = torch.load(encoder_path, map_location=args.device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()

    print("=> Loading pretrained decoder")
    decoder = NormDecoder(encoder.num_ch_enc).to(args.device)
    loaded_dict = torch.load(depth_decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.join(args.output_directory, f"{args.video_name}.mp4")
    video = cv2.VideoWriter(video_name, fourcc, FRAME_RATE, (feed_width, feed_height * 2))

    image_paths = [line.rstrip('\n') for line in open(args.val_file)]
    print(f"=> Predicting on {len(image_paths)} validation images")

    distances = list()
    for idx, image_path in enumerate(image_paths):
        if image_path.endswith(f"_norm.png"):
            continue
        frame_index, cam_side = image_path.split('.')[0].split('_')
        input_image = pre_image_op(args, 0, frame_index, cam_side)
        input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(args.device)
        features = encoder(input_image)
        outputs = decoder(features)
        norm = outputs[("norm", 0)]
        inv_norm = 1 / norm

        # Saving numpy file
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        scaled_dist = scale_norm(norm, 0.1, 95)
        distances.append(scaled_dist.cpu().numpy())

        # Saving colormapped distance image
        inv_norm_resized_np = inv_norm.squeeze().cpu().numpy()
        vmax = np.percentile(inv_norm_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=inv_norm_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap=high_res_colormap(cm.get_cmap('magma')))
        colormapped_im = (mapper.to_rgba(inv_norm_resized_np)[:, :, :3] * 255).astype(np.uint8)
        color_predictions_pil = Image.fromarray(colormapped_im)

        name_dest_im = os.path.join(args.output_directory, f"{output_name}_norm.png")

        pil_input_image = F.to_pil_image(input_image.cpu().squeeze(0))
        rgb_color_pred_concat = Image.new('RGB', (feed_width, feed_height * 2))
        rgb_color_pred_concat.paste(pil_input_image, (0, 0))
        rgb_color_pred_concat.paste(color_predictions_pil, (0, pil_input_image.height))
        rgb_color_pred_concat.save(name_dest_im)

        rgb_cv2 = np.array(pil_input_image)
        frame = np.concatenate((rgb_cv2, colormapped_im), axis=0)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"=> Processed {idx + 1} of {len(image_paths)} images - saved prediction to {name_dest_im}")
    np.save(os.path.join(args.output_directory, "distances.npy"), np.concatenate(distances))
    video.release()

    print(f"=> LoL! beautiful video created and dumped to disk. \n"
          f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)
