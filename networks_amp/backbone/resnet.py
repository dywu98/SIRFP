import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import functools
from mypath import Path
from utils.pyt_utils import load_model

BatchNorm2d = nn.BatchNorm2d
inplace = True
affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out+residual
        out = self.relu_inplace(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, inplanes=128, mg_unit=[1,1,1], out_index=[1,3,4]):
        self.inplanes = inplanes
        self.out_index = out_index
        super(ResNet, self).__init__()
        blocks = mg_unit
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        # self.conv1 = conv3x3(3, 64, stride=2)
        # self.bn1 = BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=inplace)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=inplace)
        # self.conv3 = conv3x3(64, self.inplanes)
        # self.bn3 = BatchNorm2d(self.inplanes)
        # self.relu3 = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(64, self.inplanes, 3, 1, 1, bias=False)
        )
        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=inplace)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = affine_par),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = affine_par),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        # x = self.relu1(self.bn1(self.conv1(input)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu1(self.bn1(self.conv1(input)))

        x = self.maxpool(x)

        outs=[]
        for i in range(1,5):
            res_layer = getattr(self, 'layer'+str(i))
            x = res_layer(x)
            if i in self.out_index:
                outs.append(x)
        return tuple(outs)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_resnet(name, para):
    if name.endswith('50'):
        layers = [3, 4, 6, 3]
    elif name.endswith('101'):
        layers = [3, 4, 23, 3]
    elif name.endswith('152'):
        layers = [3, 8, 36, 3]
    output_stride = para.get('os', 8)
    mg_unit = para.get('mg_unit', [1, 2, 4])
    out_index = para.get('out_index', [1, 3, 4])
    inplanes = para.get('inplanes', 128)

    model = ResNet(Bottleneck, layers, output_stride, inplanes=inplanes, mg_unit=mg_unit, out_index=out_index)
    if para.get('pretrained',True):
        load_model(model, Path.pretrained_dir(name))
    return model
