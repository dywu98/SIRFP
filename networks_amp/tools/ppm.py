import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

BatchNorm2d = nn.BatchNorm2d
inplace = True


class PPMModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), align_corners=True):
        super(PPMModule, self).__init__()

        self.align_corners = align_corners
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_features),
            nn.ReLU(inplace=inplace),
#             nn.Dropout2d(0.1),
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = BatchNorm2d(out_features)
        act = nn.ReLU(inplace=inplace)
        return nn.Sequential(prior, conv, bn, act)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=self.align_corners) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle