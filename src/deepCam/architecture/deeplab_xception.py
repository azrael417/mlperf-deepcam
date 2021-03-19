# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.cuda.amp as amp

# typing used for torch script
from typing import List


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

    
def compute_padding(kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return pad_beg, pad_end


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        # compute padding here
        pad_beg, pad_end = compute_padding(kernel_size, rate=dilation)
        #self.padding = (pad_beg, pad_end, pad_beg, pad_end)
        #self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
        #                       groups=inplanes, bias=bias)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, (pad_beg, pad_beg), dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        #x = F.pad(x, self.padding)
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ZeroPad(nn.Module):
    def __init__(self, planes, pad):
        super(ZeroPad, self).__init__()
        self.planes = planes
        self.pad = pad
        self.start = self.planes - self.pad
        
    def forward(self, x):
        x[:,self.start:,...] = 0.
        return x


class PaddedBlock(nn.Module):
    def __init__(self, planes, planes_pad, reps, dilation=1, start_with_relu=True, grow_first=True, is_last=False,
                 normalizer=nn.BatchNorm2d, process_group=None):
        super(PaddedBlock, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        rep = []

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1, dilation=dilation))
            rep.append(ZeroPad(planes, planes_pad))
            if process_group is not None:
                rep.append(nn.SyncBatchNorm(planes, process_group = process_group))
            else:
                rep.append(normalizer(planes))

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1, dilation=dilation))
            rep.append(ZeroPad(planes, planes_pad))
            if process_group is not None:
                rep.append(nn.SyncBatchNorm(planes, process_group = process_group))
            else:
                rep.append(normalizer(planes))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1, dilation=dilation))
            rep.append(ZeroPad(planes, planes_pad))
            if process_group is not None:
                rep.append(nn.SyncBatchNorm(planes, process_group = process_group))
            else:
                rep.append(normalizer(planes))

        if not start_with_relu:
            rep = rep[1:]

        if is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))
            rep.append(ZeroPad(planes, planes_pad))
            
        self.rep = nn.Sequential(*rep)
        
    
    def forward(self, inp):
        x = self.rep(inp)
        x += inp

        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False,
                 normalizer=nn.BatchNorm2d, process_group=None):
        super(Block, self).__init__()
        
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            if process_group is not None:
                self.skipbn = nn.SyncBatchNorm(planes, process_group = process_group)
            else:
                self.skipbn = normalizer(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        
        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            if process_group is not None:
                rep.append(nn.SyncBatchNorm(planes, process_group = process_group))
            else:
                rep.append(normalizer(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            if process_group is not None:
                rep.append(nn.SyncBatchNorm(filters, process_group = process_group))
            else:
                rep.append(normalizer(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            if process_group is not None:
                rep.append(nn.SyncBatchNorm(planes, process_group = process_group))
            else:
                rep.append(normalizer(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False, normalizer=nn.BatchNorm2d, process_group=None):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        if process_group is not None:
            self.bn1 = nn.SyncBatchNorm(32, process_group=process_group)
        else:
            self.bn1 = normalizer(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        if process_group is not None:
            self.bn2 = nn.SyncBatchNorm(64, process_group=process_group)
        else:
            self.bn2 = normalizer(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False, normalizer=normalizer, process_group=process_group)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True, normalizer=normalizer, process_group=process_group)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer, process_group=process_group)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True, normalizer=normalizer, process_group=process_group)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_rates[1])
        if process_group is not None:
            self.bn3 = nn.SyncBatchNorm(1536, process_group=process_group)
        else:
            self.bn3 = normalizer(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_rates[1])
        if process_group is not None:
            self.bn4 = nn.SyncBatchNorm(1536, process_group=process_group)
        else:
            self.bn4 = normalizer(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_rates[1])
        if process_group is not None:
            self.bn5 = nn.SyncBatchNorm(2048, process_group=process_group)
        else:
            self.bn5 = normalizer(2048)

        # Init weights
        self.__init_weight()

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            print(k)
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, normalizer=nn.BatchNorm2d, process_group=None):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        if process_group is not None:
            self.bn = nn.SyncBatchNorm(planes, process_group=process_group)
        else:
            self.bn = normalizer(planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                
class InterpolationUpsampler(nn.Module):
    def __init__(self, n_output, normalizer=nn.BatchNorm2d):
        super(InterpolationUpsampler, self).__init__()
        
        #last conv layer
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       normalizer(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       normalizer(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_output, kernel_size=1, stride=1))

    def forward(self, x, low_level_features, input_size: List[int]):
        x = F.interpolate(x, size=(int(math.ceil(input_size[-2]/4)),
                                   int(math.ceil(input_size[-1]/4))), mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

class DeconvUpsampler(nn.Module):
    def __init__(self, n_output, normalizer=nn.BatchNorm2d, process_group=None):
        super(DeconvUpsampler, self).__init__()

        # deconvs
        if process_group is not None:
            self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                         normalizer(256),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                         normalizer(256),
                                         nn.ReLU())
            self.conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       normalizer(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       normalizer(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1))
            self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                         normalizer(256),
                                         nn.ReLU())
        else:
            self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                         nn.SyncBatchNorm(256, process_group=process_group),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                         nn.SyncBatchNorm(256, process_group=process_group),
                                         nn.ReLU())
            self.conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.SyncBatchNorm(256, process_group=process_group),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.SyncBatchNorm(256, process_group=process_group),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=1, stride=1))
            self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False),
                                         nn.SyncBatchNorm(256, process_group=process_group),
                                         nn.ReLU())

	    #no bias or BN on the last deconv
        self.last_deconv = nn.Sequential(nn.ConvTranspose2d(256, n_output, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False))

    def forward(self, x, low_level_features, input_size: List[int]):
        x = self.deconv1(x)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.last_deconv(x)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                
class TrainableAffine(nn.Module):
    def __init__(self, num_features):
        super(TrainableAffine, self).__init__()
        self.num_features = num_features

        # weights for affine trans
        self.weights = nn.Parameter(torch.ones((num_features, 1, 1), requires_grad=True))
        self.bias = nn.Parameter(torch.zeros((num_features, 1, 1), requires_grad=True))

    def forward(self, x):
        return self.weights * x + self.bias

                                          
class MultiplexedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(MultiplexedBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.is_training = True

        if self.affine:
            self.affine_trans = TrainableAffine(self.num_features)

        self.num_samples = 0
        self.running_mean = torch.zeros(num_features, 1, 1)
        self.running_var = torch.ones(num_features, 1, 1)

    def forward(self, x):
        self.num_samples += x.shape[0]

        # put on device
        self.running_mean = self.running_mean.to(x.device)
        self.running_var = self.running_var.to(x.device)

        # update mean and variance
        if self.is_training:
            xred = torch.sum(x, dim=0, keepdim=True)
            mean = self.running_mean + (xred - self.running_mean) / self.num_samples
            var = self.running_var + (xred - self.running_mean) * (xred - mean) / self.num_samples
        else:
            mean = self.running_mean
            var = self.running_var

        # normalize:
        x = (x - mean) / torch.sqrt(var + self.eps)

        # update running stats
        if self.is_training:
            self.running_mean = (1.-self.momentum) * mean.detach() + self.momentum * self.running_mean
            self.running_var = (1.-self.momentum) * var.detach() + self.momentum * self.running_var

        # affine
        if self.affine:
            x = self.affine_trans(x)

        return x
                                          

class GlobalAveragePool(nn.Module):
    def __init__(self, in_channels, out_channels, normalizer=nn.BatchNorm2d):
        super(GlobalAveragePool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if normalizer is not None:
            self.global_average_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                     nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                                                     normalizer(out_channels),
                                                     nn.ReLU())
        else:
            self.global_average_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                     nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=True),
                                                     nn.ReLU())

    def forward(self, x):
        return self.global_average_pool(x)
        
                
                
class DeepLabv3_plus(nn.Module):
    def __init__(self, n_input=3, n_classes=21, os=16, pretrained=False, normalizer=nn.BatchNorm2d ,_print=True, rank = 0, process_group = None):
        if _print and (rank == 0):
            print("Constructing DeepLabv3+ model...")
            print("Number of output channels: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(n_input))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception(n_input, os, pretrained, normalizer, process_group = process_group)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0], normalizer=normalizer, process_group = process_group)
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1], normalizer=normalizer, process_group = process_group)
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2], normalizer=normalizer, process_group = process_group)
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3], normalizer=normalizer, process_group = process_group)

        self.relu = nn.ReLU()

        # removed batch normalization in this layer
        #self.global_avg_pool = GlobalAveragePool(2048, 256, None)
        #self.global_avg_pool = GlobalAveragePool(2048, 256, normalizer)
        self.global_avg_pool = GlobalAveragePool(2048, 256, TrainableAffine)
        #self.global_avg_pool = GlobalAveragePool(2048, 256, MultiplexedBatchNorm2d)
        #self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                     nn.Conv2d(2048, 256, 1, stride=1, bias=True),
        #                                     nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        if process_group is not None:
            self.bn1 = nn.SyncBatchNorm(256, process_group = process_group)
        else:
            self.bn1 = normalizer(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        if process_group is not None:
            self.bn2 = nn.SyncBatchNorm(48, process_group = process_group)
        else:
            self.bn2 = normalizer(48)

        # upsampling
        #self.upsample = InterpolationUpsampler(n_classes)
        self.upsample = DeconvUpsampler(n_classes, process_group = process_group)

    def forward(self, input):
        x, low_level_features = self.xception_features(input)

        # ASPP step
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        # this is very expensive in BW
        #x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # this is the same and much cheaper
        tiled = (1, 1, *(x4.size()[2:]))
        x5 = torch.tile(x5, tiled)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # low level feature processing
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        # decoder / upsampling logic
        x = self.upsample(x, low_level_features, input.size())

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
