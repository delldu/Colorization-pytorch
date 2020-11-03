
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html

import string
import torch
import torchvision
import torchvision.transforms as transforms

from data.image_folder import ImageFolder

from util import util
import numpy as np

import pdb


if __name__ == '__main__':
    # sample_ps = [1., .125, .03125]
    sample_ps = [1., .050, 0.100]

    to_visualize = ['gray', 'hint', 'hint_ab', 'real', 'fake', 'real_ab', 'fake_ab', ]

    S = len(sample_ps)

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'
    opt.dataroot = './datasets/%s/' % opt.phase

    opt.serial_batches = True
    opt.aspect_ratio = 1.


    # torchvision.datasets.
    # pdb.set_trace();
    # (Pdb) pp opt
    # Namespace(A=23.0, B=23.0, ab_max=110.0, ab_norm=110.0, ab_quant=10.0, aspect_ratio=1.0, 
    #     avg_loss_alpha=0.986, batch_size=1, beta1=0.9, checkpoints_dir='./checkpoints', 
    #     classification=False, dataroot='./datasets/val/', dataset_mode='aligned', 
    #     display_freq=10000, display_id=-1, display_ncols=5, display_port=8097, 
    #     display_server='http://localhost', display_winsize=256, epoch_count=0, 
    #     fineSize=176, gpu_ids=[0], half=False, how_many=200, init_type='normal', 
    #     input_nc=1, isTrain=True, l_cent=50.0, l_norm=100.0, lambda_A=1.0, lambda_B=1.0, 
    #     lambda_GAN=0.0, lambda_identity=0.5, loadSize=256, load_model=True, lr=0.0001, 
    #     lr_decay_iters=50, lr_policy='lambda', mask_cent=0.5, max_dataset_size=inf, 
    #     model='pix2pix', n_layers_D=3, name='siggraph_retrained', 
    #     ndf=64, ngf=64, niter=100, niter_decay=100, no_dropout=False, no_flip=False, 
    #     no_html=False, no_lsgan=False, norm='batch', num_threads=1, output_nc=2, phase='val', 
    #     pool_size=50, print_freq=200, resize_or_crop='resize_and_crop', 
    #     results_dir='./results/', sample_Ps=[1, 2, 3, 4, 5, 6, 7, 8, 9], 
    #     sample_p=1.0, save_epoch_freq=1, save_latest_freq=5000, serial_batches=True, 
    #     suffix='', update_html_freq=10000, verbose=False, which_direction='AtoB', 
    #     which_epoch='latest', which_model_netD='basic', which_model_netG='siggraph')


    dataset = ImageFolder(opt.dataroot, transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)
    model = create_model(opt)

    model.setup(opt)
    model.eval()

    # pdb.set_trace();

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # statistics
    psnrs = np.zeros((opt.how_many, S))
    entrs = np.zeros((opt.how_many, S))

    # pp dataset_loader.dataset.imgs
    # ['./datasets/val/100APPLE/IMG_0429.png',
    # './datasets/val/100APPLE/IMG_0791.png']
    for i, data_raw in enumerate(dataset_loader):
        data_raw = data_raw.cuda()
        data_raw = util.crop_mult(data_raw, mult=8)

        # pdb.set_trace();
        # (Pdb) pp data_raw.size()
        # torch.Size([1, 3, 256, 256])

        # with no points
        for (pp, sample_p) in enumerate(sample_ps):
            xxx = '%08d_%.3f' % (i, sample_p)
            img_path = [xxx.replace('.', 'p')]

            data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_p)

            # (Pdb) pp data.keys()
            # dict_keys(['A', 'B', 'hint_B', 'mask_B'])

            # (Pdb) pp data['hint_B'].size()
            # torch.Size([1, 2, 256, 256])
            # (Pdb) pp data['hint_B'].max()
            # tensor(0., device='cuda:0')
            # (Pdb) pp data['hint_B'].min()
            # tensor(0., device='cuda:0')

            # (Pdb) pp data['mask_B'].size()
            # torch.Size([1, 1, 256, 256])
            # (Pdb) pp data['mask_B'].min()
            # tensor(-0.5000, device='cuda:0')
            # (Pdb) pp data['mask_B'].max()
            # tensor(-0.5000, device='cuda:0')

            model.set_input(data)
            model.test(True)  # True means that losses will be computed
            visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

            psnrs[i, pp] = util.calculate_psnr_np(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake']))

            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        if i == opt.how_many - 1:
            break

    webpage.save()

    # Compute and print some summary statistics
    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    entrs_mean = np.mean(entrs, axis=0)
    entrs_std = np.std(entrs, axis=0) / np.sqrt(opt.how_many)

    for (pp, sample_p) in enumerate(sample_ps):
        print('p=%.3f: %.2f+/-%.2f' % (sample_p, psnrs_mean[pp], psnrs_std[pp]))
