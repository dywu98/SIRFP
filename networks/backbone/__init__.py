from networks.backbone import resnet,hrnet,mit


def build_backbone(backbone, backbone_para=None):
    if 'resnet' in backbone:
        return resnet.build_resnet(backbone, backbone_para)
    elif 'hrnetv2' in backbone:
        return hrnet.build_hrnet(backbone, backbone_para)
    elif 'mit' in backbone:
        return mit.build_mit(backbone, backbone_para)
    else:
        raise NotImplementedError
