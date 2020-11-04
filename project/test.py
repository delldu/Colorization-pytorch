"""Model test."""
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
import os

import torch

from data import get_data, rgb2lab, lab2rgb
from model import get_model, model_load, model_setenv, valid_epoch

if __name__ == "__main__":
    """Test model."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="output/ImageColor_G.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=32, help="batch size")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    # get model
    model = get_model(trainning=False)
    model_load(model.net_G, args.checkpoint)
    model.net_G.to(device)

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model.net_G = amp.initialize(model.net_G, opt_level="O1")

    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model, device, tag='test')
