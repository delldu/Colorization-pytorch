"""Data loader."""
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

import os

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils
from PIL import Image

train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"

# Color space Lab

def rgb2xyz(rgb):  # rgb from [0,1]
    # [0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:, 0, :, :]+.357580*rgb[:, 1, :, :]+.180423*rgb[:, 2, :, :]
    y = .212671*rgb[:, 0, :, :]+.715160*rgb[:, 1, :, :]+.072169*rgb[:, 2, :, :]
    z = .019334*rgb[:, 0, :, :]+.119193*rgb[:, 1, :, :]+.950227*rgb[:, 2, :, :]

    out = torch.cat(
        (x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    return out


def xyz2rgb(xyz):
    # [ 3.24048134, -1.53715152, -0.49853633],
    # [-0.96925495,  1.87599   ,  0.04155593],
    # [ 0.05564664, -0.20404134,  1.05731107]

    r = 3.24048134*xyz[:, 0, :, :]-1.53715152 * \
        xyz[:, 1, :, :]-0.49853633*xyz[:, 2, :, :]
    g = -0.96925495*xyz[:, 0, :, :]+1.87599 * \
        xyz[:, 1, :, :]+.04155593*xyz[:, 2, :, :]
    b = .05564664*xyz[:, 0, :, :]-.20404134 * \
        xyz[:, 1, :, :]+1.05731107*xyz[:, 2, :, :]

    rgb = torch.cat(
        (r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    # Some times reaches a small negative number, which causes NaNs
    rgb = torch.max(rgb, torch.zeros_like(rgb))

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    # sc.size() torch.Size([1, 3, 1, 1])

    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:, 1, :, :]-16.
    a = 500.*(xyz_int[:, 0, :, :]-xyz_int[:, 1, :, :])
    b = 200.*(xyz_int[:, 1, :, :]-xyz_int[:, 2, :, :])
    out = torch.cat(
        (L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :]+16.)/116.
    x_int = (lab[:, 1, :, :]/500.) + y_int
    z_int = y_int - (lab[:, 2, :, :]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat(
        (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out*sc

    return out


def rgb2lab(rgb, opt):
    lab = xyz2lab(rgb2xyz(rgb))

    l_rs = (lab[:, [0], :, :] - 50.0)/100.0
    ab_rs = lab[:, 1:, :, :]/110.0

    out = torch.cat((l_rs, ab_rs), dim=1)
    return out


def lab2rgb(lab_rs, opt):
    l = lab_rs[:, [0], :, :] * 100.0 + 50.0
    ab = lab_rs[:, 1:, :, :] * 110.0
    lab = torch.cat((l, ab), dim=1)

    out = xyz2rgb(lab2xyz(lab))
    return out


def get_transform(train=True):
    """Transform images."""
    ts = []
    # if train:
    #     ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)


class ImageColorDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, transforms=get_transform()):
        """Init dataset."""
        super(ImageColorDataset, self).__init__()

        self.root = root
        self.transforms = transforms

        # load all images, sorting for alignment
        # xxxx--modify here
        self.images = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        """Load images."""
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = ImageColorDataset(
        train_dataset_rootdir, get_transform(train=True))
    print(train_ds)

    # Split train_ds in train and valid set
    # xxxx--modify here
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(
        valid_ds, batch_size=bs * 2, shuffle=False, num_workers=4)

    return train_dl, valid_dl


def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = ImageColorDataset(
        test_dataset_rootdir, get_transform(train=False))
    test_dl = data.DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)


def ImageColorDatasetTest():
    """Test dataset ..."""

    ds = ImageColorDataset(train_dataset_rootdir)
    print(ds)
    # src, tgt = ds[10]
    # grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()


if __name__ == '__main__':
    ImageColorDatasetTest()
