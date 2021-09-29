"""
Qualitative test script of 2D detection for OmniDet.

# usage: ./qualitative_detection.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os

import cv2
import yaml
from PIL import Image
from torchvision import transforms

from eval.qualitative_semantic import pre_image_op
from main import collect_args
from models.detection_decoder import YoloDecoder
from models.resnet import ResnetEncoder
from train_utils.detection_utils import *
from utils import Tupperware

FRAME_RATE = 1


def color_encoding_woodscape_detection():
    detection_classes = dict(vehicles=(43, 125, 255), rider=(255, 0, 0), person=(216, 45, 128),
                             traffic_sign=(255, 175, 58), traffic_light=(43, 255, 255))
    detection_color_encoding = np.zeros((5, 3), dtype=np.uint8)
    for i, (k, v) in enumerate(detection_classes.items()):
        detection_color_encoding[i] = v
    return detection_color_encoding

@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    detection_color_encoding = color_encoding_woodscape_detection()
    encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    depth_decoder_path = os.path.join(args.pretrained_weights, "detection.pth")

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
    decoder = YoloDecoder(encoder.num_ch_enc, args).to(args.device)
    loaded_dict = torch.load(depth_decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.join(args.output_directory, f"{args.video_name}.mp4")
    video = cv2.VideoWriter(video_name, fourcc, FRAME_RATE, (feed_width, feed_height))

    image_paths = [line.rstrip('\n') for line in open(args.val_file)]
    print(f"=> Predicting on {len(image_paths)} validation images")

    for idx, image_path in enumerate(image_paths):
        if image_path.endswith(f"_detection.png"):
            continue
        frame_index, cam_side = image_path.split('.')[0].split('_')
        input_image = pre_image_op(args, 0, frame_index, cam_side)
        input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_im = os.path.join(args.output_directory, f"{output_name}_detection.png")

        # PREDICTION
        input_image = input_image.to(args.device)
        features = encoder(input_image)
        outputs = decoder(features, img_dim=[feed_width, feed_height])
        outputs = non_max_suppression(outputs["yolo_outputs"],
                                      conf_thres=args.detection_conf_thres,
                                      nms_thres=args.detection_nms_thres)

        img_d = input_image[0].cpu().detach().numpy()
        img_d = np.transpose(img_d, (1, 2, 0))
        img_cpu = np.zeros(img_d.shape, img_d.dtype)
        img_cpu[:, :, :] = img_d[:, :, :] * 255

        if not outputs[0] is None:
            outputs = torch.cat(outputs, dim=0)
            for box in outputs:
                # Get class name and color
                cls_pred = int(box[6])
                class_color = (detection_color_encoding[cls_pred]).tolist()
                x1, y1, conf = box[0], box[1], box[4]
                box = get_contour([box[0], box[1], box[2], box[3]], box[5]).exterior.coords
                boxes = np.int0(box)[0:4]
                box = np.int0([[b[0], b[1]] for b in boxes])
                cv2.drawContours(img_cpu, [box], 0, class_color, thickness=2)
                cv2.putText(img_cpu, str(f"{conf:.2f}"), (x1 - 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5e-3 * img_cpu.shape[0], (0, 255, 0), 1)

            video.write(cv2.cvtColor(np.uint8(img_cpu), cv2.COLOR_RGB2BGR))
            cv2.imwrite(name_dest_im, cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR))

        print(f"=> Processed {idx + 1} of {len(image_paths)} images - saved prediction to {name_dest_im}")

    video.release()
    print(f"=> LoL! beautiful video created and dumped to disk. \n"
          f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)
