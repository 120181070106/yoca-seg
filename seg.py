import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self,x): return self.relu(self.bn(self.conv(x)))
class parsingNet(torch.nn.Module):#size=(640,640),
    def __init__(self,pretrained=True):
        super(parsingNet, self).__init__()
        self.首上卷 = torch.nn.Sequential(
            conv_bn_relu(64,64,kernel_size=3, stride=1, padding=1),)
        self.次上卷 = torch.nn.Sequential(
            conv_bn_relu(128,64,kernel_size=3, stride=1, padding=1),)
        self.末上卷 = torch.nn.Sequential(
            conv_bn_relu(256,64,kernel_size=3, stride=1, padding=1),)
        self.拼接卷 = torch.nn.Sequential(
            conv_bn_relu(192, 64, 3,padding=2,dilation=2),
            torch.nn.Conv2d(64, 2, 1))#o:b,2(是否牙区),640/8,640/8=80
        initialize_weights(self.首上卷,self.次上卷,self.末上卷,self.拼接卷)
    def forward(self, feat1,feat2,feat3):
        feat2 = F.interpolate(self.次上卷(feat2),scale_factor = 2,mode='bilinear')
        x4 = F.interpolate(self.末上卷(feat3),scale_factor = 4,mode='bilinear')
        aux_seg = self.拼接卷(torch.cat([feat1,feat2,x4],dim=1))
        return F.interpolate(aux_seg,size=(640,640),mode='bilinear')
def initialize_weights(*models):
    for model in models: real_init_weights(model)
def real_init_weights(m):
    if isinstance(m, list): 
        for mini_m in m: real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)