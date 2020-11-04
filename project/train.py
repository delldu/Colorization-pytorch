"""Model trainning & validating."""
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
import torch.optim as optim

from data import get_data
from model import (get_model, model_load, model_save, model_setenv,
                   train_epoch, valid_epoch)

if __name__ == "__main__":
    """Trainning model."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str,
                        default="output", help="output directory")
    parser.add_argument('--checkpoint_g', type=str,
                        default="output/ImageColor_G.pth", help="checkpoint G file")
    parser.add_argument('--checkpoint_d', type=str,
                        default="output/ImageColor_D.pth", help="checkpoint D file")
    parser.add_argument('--bs', type=int, default=16, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    # get model
    model = get_model(trainning = True)
    model.set_optimizer(args.lr)

    model_load(model.net_G, args.checkpoint_g)
    if model.use_D:
        model_load(model.net_D, args.checkpoint_d)

    model.to(device)

    lr_scheduler_G = optim.lr_scheduler.StepLR(model.optimizer_G, step_size=100, gamma=0.1)
    if model.use_D:
        lr_scheduler_D = optim.lr_scheduler.StepLR(model.optimizer_D, step_size=100, gamma=0.1)

    # get data loader
    train_dl, valid_dl = get_data(trainning=True, bs=args.bs)

    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {} ...".format(epoch + 1,
            args.epochs, lr_scheduler_G.get_last_lr()))

        train_epoch(train_dl, model, device, tag='train')
        valid_epoch(valid_dl, model, device, tag='valid')

        lr_scheduler_G.step()
        if model.use_D:
            lr_scheduler_D.step()

        if (epoch + 1) % 100 == 0 or (epoch == args.epochs - 1):
            model_save(model.net_G, os.path.join(
                args.outputdir, "ImageColor_G_{}.pth".format(epoch + 1)))
            if (model.use_D):
                model_save(model.net_G, os.path.join(
                    args.outputdir, "ImageColor_D_{}.pth".format(epoch + 1)))
