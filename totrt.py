import argparse
import os
import json
import time
import numpy as np
import torch
import tensorrt as trt
from torch2trt import torch2trt, TRTModule
# from torch2trt_git import torch2trt
# from torch2trt_git.torch2trt import TRTModule
from tqdm import tqdm
# from torchstat import stat

import networks
import pruners
from utils.pyt_utils import load_model

torch.manual_seed(1989)


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
    parser = argparse.ArgumentParser(description="tensorrt")

    # test
    parser.add_argument("--input-size", type=str, default='1024,2048',
                        help="Comma-separated string with height and width of images.")

    # model
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--backbone", type=str, default='None',
                        help="backbone")
    parser.add_argument("--backbone-para", type=str, default='{}')
    parser.add_argument("--model-para", type=str, default='{}')
    parser.add_argument("--align-corner", type=str2bool, default='True',
                        help="choose align corner.")
    parser.add_argument("--dataset", type=str, default='CS')

    # ckpt
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--channel-cfg", type=str, default=None, help="path to channel_cfg.")
    parser.add_argument("--save-dir", type=str, default=None)
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


def benchmark(model, inputs, dtype='fp32', nwarmup=10, nruns=50):
    model.eval()
    if dtype == 'fp16':
        inputs = inputs.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            outputs = model(inputs)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in tqdm(range(1, nruns + 1)):
            start_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
        print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))


if __name__ == '__main__':
    parser = get_parser()
    
    args = parser.parse_args()
    print(os.path.join(args.save_dir, "trt.pth"))
    # 1 get pytorch model
    backbone_para = json.loads(args.backbone_para)
    model_para = json.loads(args.model_para)
    seg_model = eval('networks.' + args.model + '.Seg_Model')(
        backbone=args.backbone,
        backbone_para=backbone_para,
        model_para=model_para,
        num_classes=get_num_classes(args.dataset),
        align_corner=args.align_corner)

    if args.channel_cfg is not None:
        channel_cfg = torch.load(args.channel_cfg)
        pruners.init_pruned_model(seg_model, channel_cfg)
    load_model(seg_model, args.restore_from)
    seg_model = seg_model.eval().to('cuda')

    # # 2 conver to tensorrt model
    h, w = map(int, args.input_size.split(','))
    arr = torch.ones(1, 3, h, w).cuda()
    print('Convert start.')
    model_trt = torch2trt(seg_model,
                          [arr],
                          fp16_mode=True,
                          log_level=trt.Logger.INFO,
                          max_workspace_size=(1 << 32),
                          max_batch_size=1,
                          )
    
    torch.save(model_trt.state_dict(), os.path.join(args.save_dir, "trt.pth"))
    print('Convert over.')

    # 3 check speedup
    h, w = map(int, args.input_size.split(','))
    inputs = torch.randn((1, 3, h, w)).to('cuda')
    # benchmark(seg_model, inputs, dtype='fp32')

    model_trt_test = TRTModule()
    model_trt_test.load_state_dict(torch.load(os.path.join(args.save_dir, "trt.pth")))
    benchmark(model_trt, inputs, dtype='fp32')
