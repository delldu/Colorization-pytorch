"""Create model."""
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

import math
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

from networks import define_G, define_D, GANLoss, L1Loss
from data import ImagePool, rgb2lab, lab2rgb, color_sample

import pdb

class ImageColorModel(nn.Module):
    """ImageColor Model."""

    def __init__(self, trainning):
        """Init model."""
        super(ImageColorModel, self).__init__()

        self.trainning = trainning

        # load/define networks
        # L + ab + mask
        input_nc = 1
        output_nc = 2
        num_in = input_nc + output_nc + 1
        ngf = 64
        which_model_netG = "siggraph"
        norm = 'batch'
        use_dropout = True
        init_type = 'normal'
        gpu_ids = [0]
        self.netG = define_G(num_in, output_nc, ngf,
                                      which_model_netG, norm, 
                                      use_dropout, init_type,
                                      gpu_ids,
                                      use_tanh=True)

        if self.trainning:
            use_sigmoid = True
            ndf = 64
            which_model_netD = 'basic'
            n_layers_D = 3
            self.netD = define_D(input_nc + output_nc, ndf,
                                              which_model_netD,
                                              n_layers_D, norm, use_sigmoid,
                                              init_type, gpu_ids)

        if self.trainning:
            self.fake_AB_pool = ImagePool(64)
            self.criterionGAN = GANLoss(use_lsgan=False).to(os.environ["DEVICE"])
            self.criterionL1 = L1Loss()

            self.criterionCE = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            lr = 0.0001
            beta = 0.9
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=lr, betas=(beta, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.use_D = True
            if self.use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=lr, betas=(beta, 0.999))
                self.optimizers.append(self.optimizer_D)


    def forward(self, x):
        """Forward."""
        return x


def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)


def model_export():
    """Export model to onnx."""

    import onnx
    from onnx import optimizer

    # xxxx--modify here
    onnx_file = "model.onnx"
    weight_file = "checkpoint/weight.pth"

    # 1. Load model
    print("Loading model ...")
    model = ImageColorModel()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    # xxxx--modify here
    dummy_input = torch.randn(1, 3, 512, 512)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True,
                      export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('model.onnx')"


def get_model(trainning=True):
    """Create model."""
    model = ImageColorModel(trainning)
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            # xxxx--modify here
            loss = nn.L1Loss(predicts, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            if os.environ["ENABLE_APEX"] == "YES":
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images = data
            count = len(images)

            # Transform data to device
            images = images.to(device)

            data = {}
            data_lab = rgb2lab(images)
            data['A'] = data_lab[:, [0, ], :, :]
            data['B'] = data_lab[:, 1:, :, :]
            color_sample(data, p = 0.05)

            # Predict results without calculating gradients
            # self.netG(self.real_A, self.hint, self.mask)
            with torch.no_grad():
                (fake_class, fake) = model(data['A'], data['hint'], data['mask'])

            # xxxx--modify here
            loss_value = 0.0001
            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default environment variables to avoid exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        try:
            from apex import amp
        except:
            os.environ["ENABLE_APEX"] = "NO"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def infer_perform():
    """Model infer performance ..."""
    model_setenv()
    device = os.environ["DEVICE"]

    model = get_model(trainning = False).netG
    model.eval()
    model = model.to(device)

    print(model)

    for i in tqdm(range(100)):
        input = torch.randn(8, 4, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input[:, 0:1, :, :], input[:, 1:3, :, :], input[:, 3:4, :, :])

if __name__ == '__main__':
    """Test model ..."""

    # model_export()
    infer_perform()
