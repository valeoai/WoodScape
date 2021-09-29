"""
Qualitative test script of semantic segmentation for OmniDet.

# usage: ./qualitative_semantic.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from main import collect_args
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import Tupperware
from utils import semantic_color_encoding

FRAME_RATE = 1


def pre_image_op(args, index, frame_index, cam_side):
    total_car1_images = 6054
    cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                    MVL=(343, 5, 1088, 411),
                                    MVR=(185, 5, 915, 425),
                                    RV=(186, 203, 1105, 630)),
                          Car2=dict(FV=(160, 272, 1030, 677),
                                    MVL=(327, 7, 1096, 410),
                                    MVR=(175, 4, 935, 404),
                                    RV=(285, 187, 1000, 572)))
    if args.crop:
        if int(frame_index[1:]) < total_car1_images:
            cropped_coords = cropped_coords["Car1"][cam_side]
        else:
            cropped_coords = cropped_coords["Car2"][cam_side]
    else:
        cropped_coords = None

    cropped_image = get_image(args, index, cropped_coords, frame_index, cam_side)
    return cropped_image


def get_image(args, index, cropped_coords, frame_index, cam_side):
    recording_folder = "rgb_images" if index == 0 else "previous_images"
    file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
    path = os.path.join(args.dataset_dir, recording_folder, file)
    image = Image.open(path).convert('RGB')
    if args.crop:
        return image.crop(cropped_coords)
    return image


@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    semantic_decoder_path = os.path.join(args.pretrained_weights, "semantic.pth")

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
    decoder = SemanticDecoder(encoder.num_ch_enc, n_classes=args.semantic_num_classes).to(args.device)
    loaded_dict = torch.load(semantic_decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.join(args.output_directory, f"{args.video_name}.mp4")
    video = cv2.VideoWriter(video_name, fourcc, FRAME_RATE, (feed_width, feed_height * 2))

    semantic_color_coding = semantic_color_encoding(args)
    image_paths = [line.rstrip('\n') for line in open(args.val_file)]

    semantics = list()
    for idx, image_path in enumerate(image_paths):
        if image_path.endswith(f"_semantic.png"):
            continue
        frame_index, cam_side = image_path.split('.')[0].split('_')
        input_image = pre_image_op(args, 0, frame_index, cam_side)
        input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(args.device)
        features = encoder(input_image)
        outputs = decoder(features)
        semantic_pred = outputs[("semantic", 0)]

        # Saving numpy file
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_npy = os.path.join(args.output_directory, f"{output_name}_semantic.npy")

        # Saving colormapped semantic image
        semantic_pred = semantic_pred[0].data
        _, predictions = torch.max(semantic_pred.data, 0)
        predictions = predictions.byte().squeeze().cpu().detach().numpy()
        # Save prediction labels
        np.save(name_dest_npy, predictions)
        semantics.append(predictions)

        alpha = 0.5
        color_semantic = np.array(transforms.ToPILImage()(input_image.cpu().squeeze(0)))
        not_background = predictions != 0
        color_semantic[not_background, ...] = (color_semantic[not_background, ...] * (1 - alpha) +
                                               semantic_color_coding[
                                                   predictions[not_background]] * alpha)
        semantic_color_mapped_pil = Image.fromarray(color_semantic)

        name_dest_im = os.path.join(args.output_directory, f"{output_name}_semantic.png")

        pil_input_image = F.to_pil_image(input_image.cpu().squeeze(0))
        rgb_color_pred_concat = Image.new('RGB', (feed_width, feed_height + feed_height))
        rgb_color_pred_concat.paste(pil_input_image, (0, 0))
        rgb_color_pred_concat.paste(semantic_color_mapped_pil, (0, pil_input_image.height))
        rgb_color_pred_concat.save(name_dest_im)

        rgb_cv2 = np.array(pil_input_image)
        frame = np.concatenate((rgb_cv2, np.array(semantic_color_mapped_pil)), axis=0)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"=> Processed {idx + 1} of {len(image_paths)} images - saved prediction to {name_dest_im}")
    np.save(os.path.join(args.output_directory, "semantics.npy"), np.concatenate(semantics))

    if args.create_video:
        video.release()

    print(f"=> LoL! Beautiful video created and dumped to disk. \n"
          f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)
