# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:46:28 CST
# ***
# ************************************************************************************/
#

import functools
import pdb

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data

# https://github.com/richzhang/colorization-pytorch.git


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class Generator(nn.Module):
    """
    # L + ab + mask
    input_nc = 1
    output_nc = 2
    num_in = input_nc + output_nc + 1
    norm_layer = color.get_norm_layer(norm_type="batch")
    """

    def __init__(self, input_nc=1, output_nc=2, norm_layer=get_norm_layer(norm_type="batch"), classes=529):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        num_in = input_nc + output_nc + 1  # L + ab + mask

        model1 = [
            nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]

        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]

        model3short8 = [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model8 = [
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        model9up = [
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
        ]
        model2short9 = [
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        # add the two feature maps above

        model9 = [
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        # Conv10
        model10up = [
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
        ]

        model1short10 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        # add the two feature maps above

        model10 = [
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
        ]

        # classification output
        model_class = [
            nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
        ]

        # regression output
        model_out = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True), nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        # self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='bilinear'),])
        self.softmax = nn.Sequential(
            *[
                nn.Softmax(dim=1),
            ]
        )

    def forward_x(self, rgba):
        # input = torch.cat((input_A, input_B, mask_B), dim=1)
        input = data.rgba2lab(rgba)
        lab_l = input[:, 0:1, :, :]
        # lab_m = input[:, 3:4, :, :]

        conv1_2 = self.model1(input)

        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])

        # https://github.com/prokotg/colorization/blob/master/colorizers/siggraph17.py

        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        out_class = self.model_class(conv8_3.detach())
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        # out_class
        output = data.Lab2rgb(lab_l, out_reg)

        return output.clamp(0.0, 1.0)

    def forward(self, x):
        # Define max GPU/CPU memory -- 4G
        max_h = 1024
        max_W = 1024
        multi_times = 8

        # Need Resize ?
        B, C, H, W = x.size()
        if H > max_h or W > max_W:
            s = min(max_h / H, max_W / W)
            SH, SW = int(s * H), int(s * W)
            resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)
        else:
            resize_x = x

        # Need Zero Pad ?
        ZH, ZW = resize_x.size(2), resize_x.size(3)
        if ZH % multi_times != 0 or ZW % multi_times != 0:
            NH = multi_times * math.ceil(ZH / multi_times)
            NW = multi_times * math.ceil(ZW / multi_times)
            resize_zeropad_x = resize_x.new_zeros(B, C, NH, NW)
            resize_zeropad_x[:, :, 0:ZH, 0:ZW] = resize_x
        else:
            resize_zeropad_x = resize_x

        # MS Begin
        y = self.forward_x(resize_zeropad_x)
        del resize_zeropad_x, resize_x  # Release memory !!!

        y = y[:, :, 0:ZH, 0:ZW]  # Remove Zero Pads
        if ZH != H or ZW != W:
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        # MS End

        return y
