import torch
import torch.nn as nn
import torch.nn.functional as F
from networks_amp.tools.aspp import ASPP
from networks_amp.backbone import build_backbone
import functools
from torch.cuda.amp import autocast

BatchNorm2d = nn.BatchNorm2d
inplace = True


class Decoder(nn.Module):
    def __init__(self, num_classes, align_corner, high_level_inplanes=512, low_level_inplanes=256):
        super(Decoder, self).__init__()
        self.align_corner = align_corner

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=inplace)
        self.last_conv = nn.Sequential(nn.Conv2d(high_level_inplanes+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(inplace=inplace),
                                       # nn.Dropout2d(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(inplace=inplace),
                                       # nn.Dropout2d(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
#         self._init_weight()

    @autocast()
    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=self.align_corner)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Seg_Model(nn.Module):
    def __init__(self, backbone='resnet', backbone_para={}, model_para={}, num_classes=21, align_corner=False,
                 criterion=None, deepsup=False, **kwards):
        super(Seg_Model, self).__init__()
        output_stride = backbone_para.get('os',8)
        in_channels = model_para.get('in_channels', [256, 1024, 2048])
        
        self.ignore_prune_layer = model_para.get('no_prune',['decoder.bn1','aspp.bn1']) \
                                + backbone_para.get('no_prune',['backbone.layer4.2.bn3'])

        self.align_corner = align_corner
        backbone_para['out_index'] = [1, 3, 4]
        self.backbone = build_backbone(backbone, backbone_para=backbone_para)
        self.aspp = ASPP(output_stride, self.align_corner, inplanes=in_channels[2])
        self.decoder = Decoder(num_classes, self.align_corner, low_level_inplanes=in_channels[0])
        self.criterion = criterion
        self.deepsup = deepsup
        if self.deepsup:
            self.conv_deepsup =nn.Sequential(nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1, bias=False),
                                   BatchNorm2d(512),
                                   nn.ReLU(inplace=inplace),
                                   nn.Dropout2d(0.1),
                                   nn.Conv2d(512, num_classes, kernel_size=1, stride=1))

    @autocast()
    def forward(self, input, labels=None, deepsup=False):
        low_level_feat, x_deepsup, x = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
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
    
    def get_prune_params(self):
        for m in self.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in self.ignore_prune_layer:
                yield m[1].weight
