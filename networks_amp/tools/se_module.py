import torch.nn as nn

class Conv1x1(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=True)
    def forward(self,x):
        return self.conv(x)

class SEModule(nn.Module):
    def __init__(self, inplanes, ratio=16):
        super(SEModule, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 =Conv1x1(inplanes, inplanes//ratio)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = Conv1x1(inplanes//ratio, inplanes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out
    