import argparse
import os
import torch
from tqdm import tqdm
import cv2
import numpy as np
from codes.data.util import read_img
import codes.utils.util as util
from codes.models.modules.UNet_arch import HDRUNet
import scripts.metrics as m
from typing import List


def pyr_crop(img: np.array, num_layers=3) -> np.array:
    """
    Prevents errors for pyramid style networks by center cropping such that dimensions are divisible by num_layers power of 2.

    :param img: input image to be cropped
    :param num_layers: Crop input image to be able to fit a network with pyramid with num_layers layers.
    :returns: cropped image.
    """
    h, w, _ = img.shape
    des_h, des_w = np.floor(np.array([h, w]) / pow(2, num_layers - 1)).astype(np.int32) * pow(2, num_layers - 1)
    w_start, h_start = (w - des_w) // 2, (h - des_h) // 2
    return img[h_start: h_start + des_h, w_start: w_start + des_w, ::]


def restore_pyr_crop(img: np.array, original_shape: tuple) -> np.array:
    """
    Zero pads an image that has been cropped to its original size

    :param img: Cropped image (using pyr_crop for example)
    :param original_shape: Original shape of the image before it was cropped in the format (h, w, c)
    :returns: Zero padded img with size original_shape
    """
    oh, ow, _ = original_shape
    ih, iw, _ = img.shape
    if oh == ih or ow == iw:
        return img

    # Zero pad and place cropped image in center
    restored = np.zeros(original_shape, dtype=np.float32)
    start_h, start_w = int((oh - ih) // 2), int((ow - iw) // 2)
    restored[start_h: start_h + ih, start_w: start_w + iw, ::] = img

    return restored


def get_model_input(image: np.array, device: str) -> List[torch.tensor]:
    """
    :param image: cv2 image to be transformed
    :param device: device string used by torch
    :returns: list of two torch tensors (1, 3, h, w) as required by the model
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.
    cond = image.copy()

    # To Tensor
    image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
    cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond, (2, 0, 1)))).float()

    image, cond = image.to(device), cond.to(device)
    return [torch.stack([image]), torch.stack([cond])]


def visualize_model_output(output: torch.tensor) -> np.array:
    """
    :param output: output from model
    :returns: transformed output, np.uint8
    """
    output = output.detach()[0].float().cpu()
    output = util.tensor2numpy(output)
    output = output ** 2.24
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weights_path", type=str, default="pretrained_models/pretrained_model.pth")
    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    assert os.path.exists(opt.weights_path), "Unable to find pretrained weights"

    model = HDRUNet()
    model.load_state_dict(torch.load(opt.weights_path))
    model.eval()
    model.to(opt.device)

    image_paths = [os.path.join(opt.input_dir, img_path) for img_path in os.listdir(opt.input_dir) if img_path[0] != "."]

    with torch.no_grad():
        for img_path in tqdm(image_paths, total=len(image_paths), desc="Running HDRUNet..."):

            # Load and transform image
            in_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            cropped_img = pyr_crop(in_image)
            model_input = get_model_input(cropped_img, opt.device)

            # Model inference
            model_output = model(model_input)

            # Transform model output and save
            output = visualize_model_output(model_output)
            padded_img = restore_pyr_crop(output, in_image.shape)

            new_name = os.path.splitext(os.path.basename(img_path))[0] + ".hdr"
            output_path = os.path.join(opt.output_dir, new_name)
            cv2.imwrite(output_path, padded_img)
