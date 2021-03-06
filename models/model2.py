from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

# draw
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print('Space-time Memory Networks: initialized.')


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # self.weight_x = torch.ones(1, 64, 120, 216)
        # self.weight_x[0, 13, :, :] = torch.tensor(2.0)
        # self.weight_x[0, 14, :, :] = torch.tensor(2.0)
        # self.weight_x[0, 20, :, :] = torch.tensor(2.0)
        # self.weight_x[0, 31, :, :] = torch.tensor(2.0)
        # self.weight_x[0, 33, :, :] = torch.tensor(2.0)
        # self.weight_x[0, 35, :, :] = torch.tensor(1.5)
        # self.weight_x[0, 56, :, :] = torch.tensor(1.5)
        # self.weight_x = self.weight_x.cuda()

    def forward(self, in_f, in_m):
        f = (in_f - self.mean) / self.std
        o = torch.zeros_like(in_m).float()
        # in_m.shape [1,1,480,864],[b,c,,h,w]
        in_m = self.conv1_m(in_m)
        x = self.conv1(f) + in_m  # + self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        # in_m = self.maxpool(self.relu(self.bn1(in_m)))
        # x = x + x*in_m
        # x = x*self.weight_x
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f, x


class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f, x


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred2(F.relu(m3))
        p4 = self.pred2(F.relu(m4))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p, p3, p4  # , p2, p3, p4


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, keys_m, values_m, key_q, value_q):
        '''
        :param keys_m: [B,C,T,H,W], c = 128
        :param values_m: [B,C,T,H,W], c = 512
        :param key_q: [B,C,H,W], c = 128
        :param value_q: [B,C,H,W], c = 512
        :return: final_value [B, C, H, W]
        '''
        B, C_key, T, H, W = keys_m.size()
        # print('#####', B, C_key, T, H, W)
        _, C_value, _, _, _ = values_m.size()

        keys_m_temp = keys_m.view(B, C_key, T * H * W)
        keys_m_temp = torch.transpose(keys_m_temp, 1, 2)  # [b,thw,c]

        key_q_temp = key_q.view(B, C_key, H * W)  # [b,c,hw]

        p = torch.bmm(keys_m_temp, key_q_temp)  # [b, thw, hw]
        p = p / math.sqrt(C_key)
        p = F.softmax(p, dim=1)  # b, thw, hw

        mo = values_m.view(B, C_value, T * H * W)  # [b,c,thw]
        mem = torch.bmm(mo, p)  # Weighted-sum B, c, hw
        mem = mem.view(B, C_value, H, W)

        final_value = torch.cat([mem, value_q], dim=1)
        # print('mem:', torch.max(mem), torch.min(mem))
        # print('value_q:', torch.max(value_q), torch.min(value_q))

        return final_value


class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)
        self.RefUnet = RefUnet(4, 64)

    def segment(self, frame, key, value):
        '''
        :param frame: 当前需要分割的image；[B,C,H,W]
        :param key: 当前memory的key；[B,C,T,H,W]
        :param value: 当前memory的value; [B,C,T,H,W]
        :return: logits []
        '''
        # encode
        r4, r3, r2, _, _, x = self.Encoder_Q(frame)
        curKey, curValue = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16

        # memory select
        final_value = self.Memory(key, value, curKey, curValue)
        logits, p_m2, p_m3 = self.Decoder(final_value, r3, r2)  # [b,2,h,w]
        logits = self.get_logit(logits)
        p_m2 = self.get_logit(p_m2)
        p_m3 = self.get_logit(p_m3)

        return logits, p_m2, p_m3

    @staticmethod
    def get_logit(logits):
        ps = F.softmax(logits, dim=1)[:, 1]  # B h w
        B, H, W = ps.shape
        ps_tmp = torch.unsqueeze(ps, dim=1)  # B,1,H,W
        em = torch.zeros(B, 2, H, W).cuda()
        em[:, 0] = torch.prod(1 - ps_tmp, dim=1)
        em[:, 1] = ps
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def memorize(self, curFrame, curMask):
        '''
        将当前帧编码
        :param curFrame: [b,c,h,w]
        :param curMask: [b,c,h,w]
        :return: 编码后的key与value
        '''
        # print('&&&&&&&&&', curMask.shape, curFrame.shape)
        r4, _, _, _, _, x = self.Encoder_M(curFrame, curMask)
        k4, v4 = self.KV_M_r4(r4)  # num_objects, 128 and 512, H/16, W/16
        return k4, v4

    def refine(self, prevmask, curmask):
        input_tensor = torch.cat([prevmask, curmask], dim=1)
        logits = self.RefUnet(input_tensor)
        pred = self.get_logit(logits)

        return pred

    def forward(self, args, mode='s'):
        # args: Fs[:,:,t-1]
        # kwargs: Es[:,:,t-1]
        if mode =='s':  # keys
            return self.segment(args[0], args[1], args[2])
        elif mode == 'm':
            return self.memorize(args[0], args[1])
        elif mode == 'r':
            return self.refine(args[0], args[1])
