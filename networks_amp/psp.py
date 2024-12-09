import torch
import torch.nn as nn
import torch.nn.functional as F
from networks_amp.tools.ppm import PPMModule
from networks_amp.backbone import build_backbone
import functools
from torch.cuda.amp import autocast

BatchNorm2d = nn.BatchNorm2d
inplace = True


class Seg_Model(nn.Module):
    def __init__(self, backbone='resnet', backbone_para={}, model_para={}, num_classes=21, align_corner=False,
                 criterion=None, deepsup=False, **kwards):
        super(Seg_Model, self).__init__()
        in_channels = model_para.get('in_channels', [1024, 2048])
        self.ignore_prune_layer = backbone_para.get('no_prune',['backbone.layer4.2.bn3'])

        self.align_corner = align_corner
        backbone_para['out_index'] = [3,4]
        self.backbone = build_backbone(backbone, backbone_para=backbone_para)
        self.ppm = PPMModule(in_channels[1], out_features=512, align_corners=self.align_corner)
        self.last_conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        self.criterion = criterion
        self.deepsup = deepsup
        if self.deepsup:
            self.conv_deepsup =nn.Sequential(nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1, bias=False),
                                   BatchNorm2d(512),
                                   nn.ReLU(inplace=inplace),
                                   nn.Dropout2d(0.1),
                                   nn.Conv2d(512, num_classes, kernel_size=1, stride=1))

    @autocast()
    def forward(self, input, labels=None, deepsup=False):
        x_deepsup, x = self.backbone(input)
        x = self.ppm(x)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=self.align_corner)
        if self.deepsup and deepsup:
            x_deepsup = self.conv_deepsup(x_deepsup)
            x_deepsup = F.interpolate(x_deepsup, size=input.size()[2:], mode='bilinear', align_corners=self.align_corner)
            outs = [x, x_deepsup]
        else:
            outs = [x]
            
        if self.criterion is not None and labels is not None:
            loss = self.criterion(outs, labels)
            return loss
        else:
            return outs
