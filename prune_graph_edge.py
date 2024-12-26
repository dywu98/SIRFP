import argparse
import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import networks
from utils.pyt_utils import load_model
from utils.flops_counter import get_model_complexity_info
from pruners.channel_pruner import init_pruned_model
from pruners.graph_pruner import GraphEdgePruner


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DCFP")
    
    parser.add_argument("--save-path", type=str, default='./ckpt')
    parser.add_argument("--model-path", type=str, default='')
    parser.add_argument("--score-path", type=str, default='')
    
    parser.add_argument("--prune-ratio", type=float, default=0.6)
    parser.add_argument("--start_global_percent", type=float, default=0.5)
    parser.add_argument("--step_global_percent", type=float, default=0.02)

    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--backbone", type=str, default='resnet50')
    parser.add_argument("--backbone-para", type=str, default='{}')
    # parser.add_argument("--backbone-para", type=str, default="{'os':8,'mg_unit':[1,2,4],'inplanes':128}")
    parser.add_argument("--model-para", type=str, default='{}')
    parser.add_argument("--align-corner", type=str2bool, default='True')
    parser.add_argument("--dataset", type=str, default='CS')
    
    return parser


def get_num_classes(dataset):
    if dataset.startswith('CS'):
        return 19
    elif dataset.startswith('CTX'):
        return 59
    elif dataset.startswith('ADE'):
        return 150
    elif dataset.startswith('COCO'):
        return 171


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    backbone_para = json.loads(args.backbone_para)
    model_para = json.loads(args.model_para)
    seg_model = eval('networks.' + args.model + '.Seg_Model')(
        backbone=args.backbone,
        backbone_para=backbone_para,
        model_para=model_para,
        num_classes=get_num_classes(args.dataset),
        align_corner=args.align_corner,
        criterion=None,
        deepsup=False)
    flops, params = get_model_complexity_info(seg_model, (3, 512, 512), print_per_layer_stat=False)
    flops = float(flops.split(' GFLOPs')[0])

    seg_model = eval('networks.'+args.model+'.Seg_Model')(
                    backbone=args.backbone,
                    backbone_para=backbone_para,
                    model_para=model_para,
                    num_classes=get_num_classes(args.dataset),
                    align_corner=args.align_corner,
                    criterion=None,
                    deepsup=True)
    load_model(seg_model, args.model_path)
    
    global_percent = args.start_global_percent
    while True:
        seg_model_copy = copy.deepcopy(seg_model)
        pruner = GraphEdgePruner(global_percent=global_percent,layer_keep=0.02, score_file=args.score_path)
        sub_model, channel_cfg = pruner.prune_model(seg_model_copy, except_start_keys=['conv_deepsup'])
        torch.save(sub_model.state_dict(), os.path.join(args.save_path, 'pruned.pth')) 
        torch.save(channel_cfg, os.path.join(args.save_path, 'channel_cfg.pth')) 

        seg_model2 = eval('networks.'+args.model+'.Seg_Model')(
                        backbone=args.backbone,
                        backbone_para=backbone_para,
                        model_para=model_para,
                        num_classes=get_num_classes(args.dataset),
                        align_corner=args.align_corner,
                        criterion=None,
                        deepsup=False)
        channel_cfg = torch.load(os.path.join(args.save_path, 'channel_cfg.pth'))
        init_pruned_model(seg_model2, channel_cfg)
        load_model(seg_model2, os.path.join(args.save_path, 'pruned.pth'))

        flops2, params2 = get_model_complexity_info(seg_model2, (3,512,512),print_per_layer_stat=False)
        flops2 = float(flops2.split(' GFLOPs')[0])
        
        print('global_percent: {}, flops_ratio: {}'.format(global_percent, flops2/flops))
        if flops2/flops<=(1-args.prune_ratio):
            print('Finish!')
            print('flops: {}, params: {}'.format(flops, params))
            print('flops2: {}, params2: {}'.format(flops2, params2))
            break
        else:
            global_percent = global_percent+args.step_global_percent
            if global_percent>=1.0:
                break

if __name__ == '__main__':
    main()
