import sys
import json
import os
import os.path as osp
import argparse
from tqdm import tqdm
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data

from engine import Engine
import networks
from datasets import build_dataset
from optimizer import build_optimizer, adjust_learning_rate
from loss.criterion import build_criterions
from utils.pyt_utils import load_model
from utils.logger import get_logger
import pruners


BACK_BONE = 'resnet50'
MODEL = 'deeplabv3'
BATCH_SIZE = 8
IGNORE_LABEL = 255
INPUT_SIZE = '769,769'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
BETAS = '0.9,0.999'
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 12345
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SAVE_STEPS = NUM_STEPS-5*SAVE_PRED_EVERY
SNAPSHOT_DIR = 'ckpt'
WEIGHT_DECAY = 0.0005

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
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="choose the number of workers.")
    parser.add_argument("--ddp", type=str2bool, default='True')

    # snapshot
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS,
                        help="Steps start to save checkpoint.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-log", action="store_true",
                        help="Where to save log file.")

    # data
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--dataset", type=str, default='CS',
                        help="choose dataset.")
    parser.add_argument("--data-dir", type=str, default='train',
                        help="choose data type.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-brightness", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--balance", type=int, default=0,
                        help="Whether to use the balanced dataset.")
    parser.add_argument("--longsize", type=int, default=-1)
    parser.add_argument("--shortsize", type=int, default=-1)
    parser.add_argument("--data-para", type=str, default='{}')

    # model
    parser.add_argument("--model", type=str, default=MODEL,
                        help="choose model.")
    parser.add_argument("--backbone", type=str, default=BACK_BONE,
                        help="backbone")
    parser.add_argument("--backbone-para", type=str, default='{}')
    parser.add_argument("--model-para", type=str, default='{}')
    parser.add_argument("--align-corner", type=str2bool, default='True',
                        help="choose align corner.")
    parser.add_argument("--no-decay", type=str, default=None,
                        help="no weight decay.")

    # optim
    parser.add_argument("--optim", type=str, default='sgd',
                        help="optimizer")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--betas", type=str, default=BETAS)
    parser.add_argument("--warmup", type=int, default=-1)

    # loss
    parser.add_argument("--deepsup", type=str2bool, default='True',
                        help="use deepsup")
    parser.add_argument("--loss-type", type=str, default='ce')
    parser.add_argument("--loss-para", type=str, default='{}')
    
    # prune
    parser.add_argument("--prune-type", type=str, default=None)
    parser.add_argument("--backbone-ratio", type=float, default=1.0)
    parser.add_argument("--channel-cfg", type=str, default=None, help="path to channel_cfg.")
    return parser


def main():
    """Create the model and start the training."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        main_flag = (not engine.distributed) or (engine.distributed and engine.local_rank == 0)

        if not os.path.exists(args.snapshot_dir):
            if main_flag:
                os.makedirs(args.snapshot_dir)
        time.sleep(1)
                
        if args.save_log:
            logger = get_logger(log_file=os.path.join(args.snapshot_dir, 'log.txt'))
        else:
            logger = get_logger()
                
        args.save_steps = min(args.save_steps, args.num_steps)

        if main_flag:
            logger.info('Running with config:\n{}'.format(
                '\n'.join('{}:{}'.format(k,v) for k,v in vars(args).items())))

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.random_seed + engine.local_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # data loader
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)
        data_para = json.loads(args.data_para)
        dataset = build_dataset(args.dataset, split='train', data_dir=args.data_dir, crop_size=input_size, 
                            scale=args.random_scale, mirror=args.random_mirror, brightness=args.random_brightness, 
                            ignore_label=args.ignore_label, balance=args.balance,
                            longsize=args.longsize, shortsize=args.shortsize, data_para=data_para)
        train_loader, train_sampler = engine.get_train_loader(dataset)

        #criterion
        loss_para = json.loads(args.loss_para)
        criterion = build_criterions(args.loss_type, dataset, loss_para)
        
        # model
        backbone_para = json.loads(args.backbone_para)
        model_para = json.loads(args.model_para)
        seg_model = eval('networks.'+args.model+'.Seg_Model')(
                        backbone=args.backbone,
                        backbone_para=backbone_para,
                        model_para=model_para,
                        num_classes=dataset.num_classes,
                        align_corner=args.align_corner,
                        criterion = criterion,
                        deepsup=args.deepsup)
        if args.channel_cfg is not None:
            channel_cfg = torch.load(args.channel_cfg)
            pruners.init_pruned_model(seg_model, channel_cfg)
            if main_flag:
                logger.info('prune from {}'.format(args.channel_cfg))
                torch.save(channel_cfg, osp.join(args.snapshot_dir, 'channel_cfg.pth'))
        if args.resume:
            load_model(seg_model, args.resume)
            if main_flag:
                torch.save(seg_model.state_dict(), osp.join(args.snapshot_dir, 'resume_model.pth'))
                logger.info('resume from {}'.format(args.resume))

        optimizer = build_optimizer(args, seg_model)
        optimizer.zero_grad()

        if args.prune_type=='dcfp':
            train_pruning = pruners.dcfp_pruning(seg_model,0.999)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)

        model = engine.data_parallel(seg_model)
        model.train()
            
        run = True
        global_iteration = args.start_iters
        num_epochs = args.num_steps // len(train_loader)
        while run:
            avgloss = 0
            epoch = global_iteration // len(train_loader)
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(len(train_loader)), file=sys.stdout,
                        bar_format=bar_format)
            
            train_loader.dataset.pre_processing(epoch, num_epochs)
            dataloader = iter(train_loader)
            for idx in pbar:
                global_iteration += 1

                data = dataloader.next()
                images, labels, img_meta = data["img"], data["label"], data["img_meta"]
                if isinstance(images, dict):
                    for key in images:
                        images[key] = images[key].cuda(non_blocking=True)
                else:
                    images = images.cuda(non_blocking=True)
                if isinstance(labels, dict):
                    for key in labels:
                        labels[key] = labels[key].cuda(non_blocking=True)
                else:
                    labels = labels.long().cuda(non_blocking=True)

                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, args.learning_rate, global_iteration-1,
                                          args.num_steps, args.power, args.warmup)

                loss = model(images, labels, deepsup=args.deepsup)
                assert loss['loss'] == loss['loss']

                reduce_loss = engine.all_reduce_tensor(loss['loss'])
                avgloss = avgloss+reduce_loss.item()
                
                loss['loss'].backward()

                if args.prune_type=='dcfp':
                    train_pruning.step(seg_model)
                    
                optimizer.step()

                print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                        + ' Iter{}/{}:'.format(idx + 1, len(train_loader)) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduce_loss.item()
                pbar.set_description(print_str, refresh=False)

                if main_flag:
                    if global_iteration>=args.save_steps:
                        if (args.num_steps-global_iteration) % args.save_pred_every == 0 or global_iteration >= args.num_steps:
                            logger.info('taking snapshot ...')
                            torch.save(seg_model.state_dict(),osp.join(args.snapshot_dir, args.dataset+'_scenes_'+str(global_iteration)+'.pth'))

                if global_iteration >= args.num_steps:
                    run = False
                    if main_flag and args.prune_type=='dcfp':
                        train_pruning.export_eic(osp.join(args.snapshot_dir, 'score.pth'))
                    break
            
            if main_flag:
                avgloss = avgloss*1.0/len(train_loader)
                logger.info('Epoch %d: avgloss=%.2f'%(epoch,avgloss))


if __name__ == '__main__':
    main()
