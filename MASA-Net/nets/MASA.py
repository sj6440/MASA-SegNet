import functools
from typing import Any, Sequence, Tuple

import cv2
import einops
import numpy as np
import scipy
import torch.nn as nn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import imageio
from scipy import misc
from torch import Tensor
from torchvision import transforms
import PIL.Image
import os
import time

def layer_norm_process(feature: torch.Tensor, beta=0., gamma=1., eps=1e-5):
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)

    mean = var_mean[1]

    var = var_mean[0]

    # layer norm process
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch,height, width ,channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x

def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
          x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
          gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x



class GridGatingUnit(nn.Module):
    def                __init__(self,n1,dim,use_bias=True):
        super().__init__()
        self.bias = use_bias
        self.n1 = n1
        #self.layernorm = nn.LayerNorm(dim)
       # self.fc = nn.Linear(n1,n1,bias=self.bias)
        self.fc = nn.Conv2d(n1, n1, 1, 1, bias=True)
    def forward(self, x):
       # print(x.shape)
        c = x.size(1)
        c = c//2
        u, v = torch.split(x, c, dim=1)
        v = layer_norm_process(v)
        #v = v.permute(0, 3, 1, 2)
        v = self.fc(v)
        return u * (v + 1.)

class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self,n1, dim,grid_size,num_channels, use_bias=True,factor=2,dropout_rate=0.):
        super().__init__()
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.layernorm = nn.LayerNorm(dim)
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.gelu = nn.GELU()
        self.gridgatingunit = GridGatingUnit(n1,dim=dim,use_bias=self.bias)
        self.dropout = nn.Dropout(self.drop)
        self.fc1 = nn.Conv2d(num_channels,num_channels*self.factor,kernel_size=1,stride=1,bias=True)
        self.fc2 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, bias=True)
    def forward(self, x):
        n,num_channels,h, w = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = x.permute(0,2,3,1) #[b,h,w,c]
        x = block_images_einops(x, patch_size=(fh, fw))
        y = layer_norm_process(x)
        y = y.permute(0, 3,1,2) #[b,c,h,w]
        y = self.fc1(y)
        y = self.gelu(y)
        y = self.gridgatingunit(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y.permute(0,2,3,1)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class BlockGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self,n2,dim, use_bias=True):
        super().__init__()
        self.bias = use_bias
        self.layernorm = nn.LayerNorm(dim)
        self.n2=n2
       # self.fc = nn.Linear(n2,n2,bias=self.bias)
        self.fc = nn.Conv2d(n2,n2,1,1,bias=True)
    def forward(self, x):
        c = x.size(1)
        c = c//2
        u, v = torch.split(x, c, dim=1)
        v = layer_norm_process(v)
        v = self.fc(v)
        return u * (v + 1.)

class BlockGmlpLayer(nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self,n2,num_channels, block_size,dim, use_bias=True,factor=2,dropout_rate=0.):
        super().__init__()
        self.block_size = block_size
        self.num_channels = num_channels
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.layernorm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        self.dim=dim
        self.blockgatingunit = BlockGatingUnit(n2=n2,dim=self.dim,use_bias=self.bias)
        self.dropout = nn.Dropout(self.drop)
        self.fc1 = nn.Conv2d(num_channels, num_channels * self.factor, kernel_size=1, stride=1,bias=self.bias)
        self.fc2 = nn.Conv2d(num_channels,num_channels,kernel_size=1,stride=1,bias=True)
    def forward(self, x):
        n, num_channels, h, w = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        x = x.permute(0, 2, 3, 1)  # [b,h,w,c]
        x = block_images_einops(x, patch_size=(fh, fw))
        # MLP2: Local (block) mixing part, provides within-block communication.
        y = layer_norm_process(x)
        y = y.permute(0, 3, 1, 2)  # [b,c,h,w]
        y = self.fc1(y)
        y = self.gelu(y)
        y = self.blockgatingunit(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y.permute(0, 2, 3, 1) # [b,h,w,c]
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, n1,n2,block_size, grid_size,dim,num_channels, block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor = 2,use_bias=True,dropout_rate=0.):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.drop = dropout_rate
        self.fc1 = nn.Conv2d(num_channels,num_channels * self.input_proj_factor, kernel_size=1,stride=1,bias=self.bias)
        self.dim = dim
        self.gelu = nn.GELU()
        self.gridgmlplayer = GridGmlpLayer(n1=n1,dim=self.dim,num_channels=num_channels,grid_size=self.grid_size,factor=self.grid_gmlp_factor,use_bias=self.bias, dropout_rate=self.drop)
        self.blockgmlplayer = BlockGmlpLayer(n2=n2,dim=self.dim,num_channels=num_channels,block_size=self.block_size,factor=self.block_gmlp_factor,  use_bias=self.bias,dropout_rate=self.drop)
        self.fc2 = nn.Conv2d(num_channels * self.input_proj_factor,num_channels, kernel_size=1,stride=1,bias=self.bias)
        self.dropout = nn.Dropout()

    def forward(self, x):
        shortcut = x
        x = layer_norm_process(x)
        x = self.fc1(x)
        x = self.gelu(x)
        c = x.size(1)//2
        u, v = torch.split(x, c, dim=1)
        # GridGMLPLayer/
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        x = torch.cat([u, v], dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.fc2(x)
       # x = x.permute(0,3,1,2)
        x = self.dropout(x)
        x = x + shortcut
        return x

class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x

class MAXIM_backbone(nn.Module):
    def __init__(self, ):
        super().__init__()
##################################################################################################################################
        # self.conv_in = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=3,kernel_size=(3,3),bias=True, padding=1,groups=3),
        #                              nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(1,1),padding=0),
        #                              nn.BatchNorm2d(64),
        #                              nn.GELU(),
        #                              nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3,3),padding=1),
        #                              nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),padding=0),
        #                              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        #                              nn.BatchNorm2d(64),
        #                              nn.GELU()
        # )
#################################################################################################################################
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=3,kernel_size=(3,3),bias=True, padding=1,groups=3),
                                   nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(1,1),padding=0))
        self.BatchNorm2d1 = nn.BatchNorm2d(64)
        self.GELU1 = nn.GELU()
        self.pooling1 = nn.MaxPool2d(3, 2, padding=1)
        self.conv_res1 = PatchEmbed(patch_size=2, in_chans=3, embed_dim=64, kernel_size=1)

        self.residualsplitheadmultiaxisgmlpLayer1 = ResidualSplitHeadMultiAxisGmlpLayer(dim=1,n1=64,n2=64,num_channels=64, grid_size=(16,16),block_size=(16,16),
        grid_gmlp_factor=2,block_gmlp_factor=2,input_proj_factor=2,
        use_bias=True,dropout_rate=0.)


        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), bias=True, padding=1, groups=64),
                                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0))
        self.BatchNorm2d2 = nn.BatchNorm2d(128)
        self.GELU2 = nn.GELU()
        self.pooling2 = nn.MaxPool2d(3, 2, padding=1)
        self.conv_res2 = PatchEmbed(patch_size=2, in_chans=64, embed_dim=128, kernel_size=1)

        self.residualsplitheadmultiaxisgmlpLayer2 = ResidualSplitHeadMultiAxisGmlpLayer(dim=1,n1=128,n2=128,num_channels=128, grid_size=(16,16),block_size=(16,16),
        grid_gmlp_factor=2,block_gmlp_factor=2,input_proj_factor=2,
        use_bias=True,dropout_rate=0.)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3,3),bias=True, padding=1,groups=128),
                                    nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(1,1),padding=0))
       # self.conv3 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3,3),bias=True, padding=1)
        self.BatchNorm2d3 = nn.BatchNorm2d(256)
        self.GELU3 = nn.GELU()
        self.pooling3 = nn.MaxPool2d(3, 2, padding=1)
        self.conv_res3 = PatchEmbed(patch_size=2, in_chans=128, embed_dim=256, kernel_size=1)

        self.residualsplitheadmultiaxisgmlpLayer3 = ResidualSplitHeadMultiAxisGmlpLayer(dim=1,n1=256,n2=256,num_channels=256, grid_size=(16,16),block_size=(16,16),
        grid_gmlp_factor=2,block_gmlp_factor=2,input_proj_factor=2,
        use_bias=True,dropout_rate=0.)
        #

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), bias=True, padding=1, groups=256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), padding=0))
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3,3),bias=True, padding=1)
        self.BatchNorm2d4 = nn.BatchNorm2d(512)
        self.GELU4 = nn.GELU()
        self.apooling4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0,dilation=4),)
        self.conv_res4 = PatchEmbed(patch_size=1, in_chans=256, embed_dim=512, kernel_size=1)

        self.residualsplitheadmultiaxisgmlpLayer4 = ResidualSplitHeadMultiAxisGmlpLayer(dim=1, n1=512, n2=512,
                                                                                        num_channels=512,
                                                                                        grid_size=(16, 16),
                                                                                        block_size=(16, 16),
                                                                                        grid_gmlp_factor=2,
                                                                                        block_gmlp_factor=2,
                                                                                        input_proj_factor=2,
                                                                                        use_bias=True, dropout_rate=0.)

    def forward(self, x):
        x1 = self.pooling1(self.GELU1(self.BatchNorm2d1(self.conv1(x))))
        x1 = self.residualsplitheadmultiaxisgmlpLayer1(x1)
        x1 = self.conv_res1(x) + x1

        x2 = self.pooling2(self.GELU2(self.BatchNorm2d2(self.conv2(x1))))
        x2 = self.residualsplitheadmultiaxisgmlpLayer2(x2)
        mid_f = self.conv_res2(x1) + x2

        x3 = self.pooling3(self.GELU3(self.BatchNorm2d3(self.conv3(mid_f))))
        x3= self.residualsplitheadmultiaxisgmlpLayer3(x3)
        x3 = self.conv_res3(mid_f) + x3

        x4 = self.apooling4(self.GELU4(self.BatchNorm2d4(self.conv4(x3))))
        x4= self.residualsplitheadmultiaxisgmlpLayer4(x4)
        out = self.conv_res4(x3) + x4
        return mid_f,out
