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
from apex import amp
from tqdm import tqdm

from model_helper import define_G, define_D, GANLoss, L1Loss
from data import ImagePool, rgb2lab, lab2rgb, Lab2rgb, color_sample, ab2index

import pdb


def PSNR(img1, img2):
    """PSNR."""
    difference = (1.*img1-img2)**2
    mse = torch.sqrt(torch.mean(difference)) + 0.000001
    return 20*torch.log10(1./mse)


class ImageColorModel(nn.Module):
    """ImageColor Model."""

    def __init__(self, trainning):
        """Init model."""
        super(ImageColorModel, self).__init__()

        self.trainning = trainning
        self.use_D = False
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
        self.net_G = define_G(num_in, output_nc, ngf,
                              which_model_netG, norm,
                              use_dropout, init_type,
                              gpu_ids,
                              use_tanh=True)

        if self.trainning:
            use_sigmoid = True
            ndf = 64
            which_model_netD = 'basic'
            n_layers_D = 3
            self.net_D = define_D(input_nc + output_nc, ndf,
                                  which_model_netD,
                                  n_layers_D, norm, use_sigmoid,
                                  init_type, gpu_ids)

        if self.trainning:
            self.fake_AB_pool = ImagePool(64)
            # xxxx
            self.criterionGAN = GANLoss(
                use_lsgan=True).to(os.environ["DEVICE"])
            self.criterionL1 = L1Loss()

            self.criterionCE = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            # lr = 0.0001
            # beta = 0.9
            # self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
            #                                     lr=lr, betas=(beta, 0.999))
            # if self.use_D:
            #     self.optimizer_D = torch.optim.Adam(self.net_D.parameters(),
            #                                         lr=lr, betas=(beta, 0.999))

    def set_optimizer(self, lr):
        beta = 0.9
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                            lr=lr, betas=(beta, 0.999))
        if self.use_D:
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(),
                                                lr=lr, betas=(beta, 0.999))

    def forward(self, input):
        """Forward."""
        self.real_A = input['A']
        self.real_B = input['B']
        self.hint_B = input['hint']
        self.mask_B = input['mask']
        self.mask_B_nc = self.mask_B + 0.5
        self.real_B_enc = ab2index(self.real_B[:, :, ::4, ::4])
        # (Pdb) pp self.real_B_enc.size()
        # torch.Size([1, 1, 64, 64])

        (self.fake_B_class, self.fake_B) = self.net_G(
            self.real_A, self.hint_B, self.mask_B)

        # if(self.use_D):
        #     # update D
        #     self.set_requires_grad(self.net_D, True)
        #     self.optimizer_D.zero_grad()
        #     self.backward_D()
        #     self.optimizer_D.step()

        #     self.set_requires_grad(self.net_D, False)

        # # update G
        # self.optimizer_G.zero_grad()
        # self.backward_G()
        # self.optimizer_G.step()

    def optimize(self):
        if(self.use_D):
            # update D
            self.set_requires_grad(self.net_D, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.set_requires_grad(self.net_D, False)

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def backward_D(self):
        # Fake
        fake_AB = self.fake_AB_pool.query(
            torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.net_D(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.net_D(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def compute_losses_G(self):
        if self.use_D:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.net_D(fake_AB)
            self.loss_G = self.criterionGAN(pred_fake, True)
        else:
            lambda_A = 1.0
            # classification statistics
            # cross-entropy loss
            self.loss_G_CE = self.criterionCE(
                self.fake_B_class.type(torch.cuda.FloatTensor),
                self.real_B_enc[:, 0, :, :].type(torch.cuda.LongTensor))

            self.loss_G_L1 = 10 * torch.mean(self.criterionL1(
                self.fake_B.type(torch.cuda.FloatTensor),
                self.real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = self.loss_G_CE * lambda_A + self.loss_G_L1
            # pdb.set_trace()

    def backward_G(self):
        self.compute_losses_G()
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

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


def export_onnx_model():
    """Export onnx model."""

    import onnx
    from onnx import optimizer

    onnx_file = "output/image_color.onnx"
    weight_file = "output/ImageColor.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)

    input_names = ["input"]
    output_names = ["noise_level", "output"]
    # variable lenght axes
    dynamic_axes = {'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
                    'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
    torch.onnx.export(model, dummy_input, onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True,
                      export_params=True,
                      dynamic_axes=dynamic_axes)

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
    # python -c "import netron; netron.start('image_clean.onnx')"


def export_torch_model():
    """Export torch model."""

    script_file = "output/image_color.pt"
    weight_file = "output/ImageColor.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(script_file)


def get_model(trainning=True):
    """Create model."""
    model_setenv()
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


def train_epoch(loader, model, device, tag=''):
    """Trainning model ..."""

    total_loss_G = Counter()
    total_loss_D = Counter()

    model.train()

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
            color_sample(data, p=0.01)

            del images

            model(data)
            model.optimize()

            del data

            loss_value = model.loss_G.item()
            if not math.isfinite(loss_value):
                print("Loss G is {}, stopping training".format(loss_value))
                sys.exit(1)
            total_loss_G.update(loss_value, count)

            if model.use_D:
                loss_value = model.loss_D.item()
                if not math.isfinite(loss_value):
                    print("Loss D is {}, stopping training".format(loss_value))
                    sys.exit(1)
                total_loss_D.update(loss_value, count)
                t.set_postfix(loss='G:{:.6f},D:{:.6f}'.format(
                    total_loss_G.avg, total_loss_D.avg))
            else:
                t.set_postfix(loss='G:{:.6f}'.format(total_loss_G.avg))
            t.update(count)

        return total_loss_G.avg, total_loss_D.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.net_G.eval()

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
            color_sample(data, p=0.01)

            del images

            # Predict results without calculating gradients
            # self.net_G(self.real_A, self.hint, self.mask)
            with torch.no_grad():
                (fake_class, fake) = model.net_G(
                    data['A'], data['hint'], data['mask'])

            loss_value = PSNR(
                Lab2rgb(data['A'], data['B']), Lab2rgb(data['A'], fake))
            valid_loss.update(loss_value, count)
            t.set_postfix(PSNR='{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_device():
    """First call model_setenv. """
    return torch.device(os.environ["DEVICE"])


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
        os.environ["ENABLE_APEX"] = "YES"

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


def enable_amp(x):
    """Init Automatic Mixed Precision(AMP)."""
    if os.environ["ENABLE_APEX"] == "YES":
        x = amp.initialize(x, opt_level="O1")


def infer_perform():
    """Model infer performance ..."""

    model = get_model(trainning=False).net_G
    device = model_device()

    model.eval()
    model = model.to(device)
    enable_amp(model)

    print(model)

    progress_bar = tqdm(total=100)
    progress_bar.set_description("Test Inference Performance ...")

    for i in tqdm(range(100)):
        input = torch.randn(8, 4, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input[:, 0:1, :, :],
                           input[:, 1:3, :, :], input[:, 3:4, :, :])

        progress_bar.update(1)


if __name__ == '__main__':
    """Test model ..."""

    model = get_model()
    print(model)

    export_torch_model()
    export_onnx_model()

    infer_perform()
