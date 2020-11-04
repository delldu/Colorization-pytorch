"""Model predict."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:46:28 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from data import rgb2lab, Lab2rgb, color_sample, multiple_crop
from model import get_model, model_load, model_setenv

import pdb

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="output/ImageColor_G.pth", help="checkpint file")
    parser.add_argument('--input', type=str,
                        default="dataset/test/*.png", help="input image")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    model = get_model(trainning=False).net_G
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)
        H, W = input_tensor.shape[2:]
        if (H % 8 != 0 or W % 8 != 0):
            input_tensor = multiple_crop(input_tensor)

        data = {}
        data_lab = rgb2lab(input_tensor)
        data['A'] = data_lab[:, [0, ], :, :]
        data['B'] = data_lab[:, 1:, :, :]
        color_sample(data, p=0.01)
        del input_tensor

        # Sample
        output_tensor = Lab2rgb(
            data['A'], data['hint']).clamp(0, 1.0).squeeze()
        toimage(output_tensor.cpu()).save(
            "output/sample_{}".format(os.path.basename(filename)))

        with torch.no_grad():
            (fake_class, fake) = model(data['A'], data['hint'], data['mask'])

        output_tensor = Lab2rgb(data['A'], fake).clamp(0, 1.0).squeeze()
        toimage(output_tensor.cpu()).save(
            "output/color_{}".format(os.path.basename(filename)))

        del data, fake_class, fake, output_tensor
