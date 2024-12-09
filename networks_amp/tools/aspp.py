import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

BatchNorm2d = nn.BatchNorm2d
inplace = True

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=inplace)

#         self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, align_corner, inplanes=2048, outplanes=512):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        elif output_stride == 32:
            dilations = [1, 3, 6, 9]
        else:
            raise NotImplementedError
        self.outplanes = outplanes
        self.align_corner = align_corner

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             # InPlaceABNSync(256),
                                             BatchNorm2d(256),
                                             nn.ReLU(inplace=inplace)
                                            )
        if self.outplanes is not None:
            self.conv1 = nn.Conv2d(1280, self.outplanes, 1, bias=False)
            self.bn1 = BatchNorm2d(self.outplanes)
            self.relu = nn.ReLU(inplace=inplace)
            # self.bn1_relu = InPlaceABNSync(self.out_channel)
        self.dropout = nn.Dropout2d(0.1)
#         self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=self.align_corner)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        if self.outplanes is not None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            # x = self.bn1_relu(x)
        # x = self.dropout(x)
        return x