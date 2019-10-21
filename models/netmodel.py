# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# @Time : 2019/9/13 15:36
# @Author : liufang
# @File : model.py
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import functools

nclass=11

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net




class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
class NetG(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(NetG, self).__init__()
        # construct unet structure
        self.conv1 = nn.Conv2d(9, 3, 3,)
        self.BatchNorm1=nn.BatchNorm2d(3)
        self.LeakyReLU1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.BatchNorm2 = nn.BatchNorm2d(3)
        self.LeakyReLU2 = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(3, ngf, input_nc=3, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        x=self.conv1(input)
        # print(x.shape)
        x=self.BatchNorm1(x)
        x = self.LeakyReLU1(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU2(x)
        # print(x.shape)
        x = self.model(x)
        output=self.tanh(x)
        return output

#TODO modify this network to segmentation network
class NetC(nn.Module):
    def __init__(self,input_nc,output_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(NetC, self).__init__()
        self.output_nc=output_nc
        self.net = [
            nn.Conv2d(input_nc, ndf, 15,stride=4),
            # norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),

            nn.Conv2d(ndf, ndf * 2, 5, stride=4),
            # norm_layer(ndf* 2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),

            nn.Conv2d(ndf * 2, ndf * 4, 3),
            # norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 4, 3),
            # norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 4, ndf * 4, 3),
            # norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),
            nn.Conv2d(ndf * 4, ndf * 8, 7),
            # norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            nn.Conv2d(ndf * 8, ndf * 8, 1, stride=2),
            # norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Conv2d(ndf * 8, 11, 1),
            # norm_layer(11),
        ]
        self.net = nn.Sequential(*self.net)
    def forward(self, x):
        x = self.net(x)
        # print(x.shape)
        x = x.view(-1, 11)
        # x = F.softmax(x, dim=-1)

        return x

class NetD(nn.Module):
    def __init__(self, input_nc,output_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(NetD, self).__init__()
        # size: 3 * 36 * 120
        self.conv1 =nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0)
        self.LeakyReLU1 =nn.LeakyReLU(0.2, True)

        self.localNet = [


            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),

            # nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=0),
            # norm_layer(ndf * 8),
            # nn.LeakyReLU(0.2, True),
            ]

        self.localNet = nn.Sequential(*self.localNet)

        self.globalNet = [

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=0),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
        ]

        self.globalNet = nn.Sequential(*self.globalNet)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4 * 4 * 512, 1024)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048, output_nc)
        self.drop3 = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
    def forward(self, input):
        x=self.LeakyReLU1(self.conv1(input))

        localx = self.localNet(x)
        localx = localx.view(-1, 8 * 8 * 512)
        localx = self.drop1(F.relu(self.fc1(localx)))

        globalx = self.globalNet(x)
        globalx = globalx.view(-1, 4 * 4 * 512)
        globalx = self.drop2(F.relu(self.fc2(globalx)))

        combinex=torch.cat([localx,globalx],-1)
        combinex = self.drop3(F.relu(self.fc3(combinex)))
        output=self.tanh(combinex)
        return output
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_Net(input_nc, output_nc, ngf, netType, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netType == 'NetG':
        net = NetG(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netType == 'NetC':
        net = NetC(input_nc, 11, ngf, norm_layer=norm_layer)
    elif netType == 'NetD':
        net = NetD(input_nc, 1, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netType)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss