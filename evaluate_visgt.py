import argparse
import numpy as np
import sys
import json
import time
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import networks
from datasets import build_dataset
import os
from math import ceil
from PIL import Image as PILImage

from utils.pyt_utils import load_model
from utils.edge_utils import mask_to_boundary
from engine import Engine
import pruners
from torch.cuda.amp import GradScaler, autocast

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" #总是报显存不足的问题，是因为碎片没完全释放
# if hasattr(torch.cuda, 'empty_cache'):
#    torch.cuda.empty_cache()


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
    parser.add_argument("--dataset", type=str, default='CS',
                        help="choose dataset.")
    parser.add_argument("--data-dir", type=str, default='val',
                        help="choose data type.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ddp", type=str2bool, default='True')

    # test
    parser.add_argument("--input-size", type=str, default='769,769',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--longsize", type=int, default=-1)
    parser.add_argument("--shortsize", type=int, default=-1)
    parser.add_argument("--whole", type=str2bool, default='False',
                        help="use whole input size.")
    parser.add_argument("--flip", type=str2bool, default='False',
                        help="flip test.")
    parser.add_argument("--ms", type=str, default='1',
                        help="multi scale")
    parser.add_argument("--iou-type", type=str, default='segm')
    parser.add_argument("--dilation-ratio", type=float, default=0.02)

    # model
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--backbone", type=str, default='renet50',
                        help="backbone")
    parser.add_argument("--backbone-para", type=str, default='{}')
    parser.add_argument("--model-para", type=str, default='{}')
    parser.add_argument("--align-corner", type=str2bool, default='True',
                        help="choose align corner.")

    # ckpt
    parser.add_argument("--restore-from", type=str, default='xxx.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--save-predict", type=str2bool, default='True',
                        help="save predict images")
    parser.add_argument("--channel-cfg", type=str, default=None, help="path to channel_cfg.")
    parser.add_argument("--use-trt", type=str2bool, default='False')
    return parser

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def pad(image, target_size):
    rows_missing = target_size[0] - image.shape[2]
    cols_missing = target_size[1] - image.shape[3]
    padded_img = F.pad(image, (0, cols_missing, 0, rows_missing), mode='constant', value=0.)
    return padded_img.contiguous()

def pad_inf(image, label=None):
    h, w = image.size()[-2:] 
    stride = 8
    pad_h = (stride + 1 - h % stride) % stride
    pad_w = (stride + 1 - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', 
                          value=255)
            return image, label
    return image

def generate_size_image(image, size, mode):
    h, w = image.shape[2:]
    if mode=='long':
        f_scale = size*1.0/max(h,w)
    elif mode=='short':
        f_scale = size*1.0/min(h,w)
    else:
        raise NotImplementedError(mode)
    new_h = np.int(h * f_scale + 0.5)
    new_w = np.int(w * f_scale + 0.5)
    image = F.interpolate(image, size=(new_h,new_w), mode='bilinear', align_corners=False)
    return image

def predict_sliding(net, image, tile_size, classes):
#     interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = torch.zeros((image_size[0],classes, image_size[2], image_size[3]))
    count_predictions = torch.zeros((1, classes, image_size[2], image_size[3]))

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            padded_prediction = net(padded_img.cuda(non_blocking=True))
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            elif isinstance(padded_prediction, dict):
                padded_prediction = padded_prediction['pred']
            prediction = padded_prediction.cpu()[:, :, 0:img.shape[2], 0:img.shape[3]]
            count_predictions[0, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs

def predict_whole(net, image):
    N_, C_, H_, W_ = image.shape
#     interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    with torch.no_grad():
        prediction = net(image)
    if isinstance(prediction, list):
        prediction = prediction[0]
    elif isinstance(prediction, dict):
        prediction = prediction['pred']
#     prediction = interp(prediction)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, align_corner, whole):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    N_, C_, H_, W_ = image.shape
    full_probs = torch.zeros((N_, classes, H_, W_ ))
    for scale in scales:
        scale = float(scale)
        hs = int(H_*scale)
        ws = int(W_*scale)
        scale_image = F.interpolate(image, size=[hs, ws], mode='bilinear', align_corners=align_corner)
        with torch.no_grad():
            if whole:
                scaled_probs = predict_whole(net, scale_image)
            else:
                scaled_probs = predict_sliding(net, scale_image, tile_size, classes)
            if flip_evaluation == True:
                flip_image = torch.flip(scale_image, [3])
                if whole:
                    flip_scaled_probs = predict_whole(net, flip_image)
                else:
                    flip_scaled_probs = predict_sliding(net, flip_image, tile_size, classes)
                scaled_probs = 0.5 * (scaled_probs + torch.flip(flip_scaled_probs, [3]))
            scaled_probs = F.interpolate(scaled_probs, size=[H_, W_], mode='bilinear',align_corners=align_corner)
        full_probs += scaled_probs.cpu()
    full_probs /= len(scales)
    # full_probs = full_probs.numpy().transpose(0,2,3,1)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        cudnn.benchmark = True

        h, w= map(int, args.input_size.split(','))
        input_size = (h,w)
         
        args.ms = [float(s) for s in args.ms.split(',')]
        
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            print("Running with config:")
            for k,v in vars(args).items():
                print('{}: {}'.format(k,v))

        dataset = build_dataset(args.dataset, split='val', data_dir=args.data_dir)
        test_loader, test_sampler = engine.get_test_loader(dataset)
        if engine.distributed:
            test_sampler.set_epoch(0)

        if args.use_trt:
            from torch2trt import TRTModule
            seg_model = TRTModule()
            seg_model.load_state_dict(torch.load(args.restore_from))
        else:
            backbone_para = json.loads(args.backbone_para)
            model_para = json.loads(args.model_para)
            seg_model = eval('networks.'+args.model+'.Seg_Model')(
                            backbone=args.backbone,
                            backbone_para=backbone_para,
                            model_para=model_para,
                            num_classes=dataset.num_classes,
                            align_corner=args.align_corner)

            if args.channel_cfg is not None:
                channel_cfg = torch.load(args.channel_cfg)
                pruners.init_pruned_model(seg_model, channel_cfg)
            load_model(seg_model, args.restore_from)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)

        model = engine.data_parallel(seg_model)
        model.eval()

        data_list = []
        confusion_matrix = np.zeros((dataset.num_classes,dataset.num_classes))
        # palette = get_palette(256)
        palette = list(dataset.cmap_labels.reshape(-1))

        if args.save_predict:
            save_path = os.path.join(os.path.dirname(args.restore_from), 'gt_outputs')
            if not os.path.exists(save_path):
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    os.makedirs(save_path)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(test_loader)
        
        fps_warmup = 5
        pure_inf_time = 0
        
        with torch.no_grad():
        
            for idx in pbar:
                
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" #总是报显存不足的问题，是因为碎片没完全释放
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                data = dataloader.next()
                image, label, img_meta = data["img"], data["label"], data["img_meta"]
                # if args.longsize>0:
                #     image = generate_size_image(image,args.longsize,'long')
                # elif args.shortsize>0:
                #     image = generate_size_image(image,args.shortsize,'short')
                # size_scale = image.shape[2:]
                # if args.whole and args.align_corner:
                #     image = pad_inf(image)
                # image = image.cuda()
                # # with autocast():
                # with torch.no_grad():
                #     torch.cuda.synchronize()
                #     start_time = time.perf_counter()

                #     with autocast():
                #         if args.whole and args.ms == [1.0]:
                #             output = predict_whole(model, image)
                #         else:
                #             output = predict_multiscale(model, image, input_size, args.ms, dataset.num_classes, args.flip, args.align_corner, args.whole)
                        
                #     torch.cuda.synchronize()
                #     elapsed = time.perf_counter() - start_time
                    
                #     # output = output.cpu()
                #     output = output[:,:,:size_scale[0],:size_scale[1]]
                #     if args.longsize>0 or args.shortsize>0:
                #         output = F.interpolate(output, size=(img_meta[0]["size"][0], img_meta[0]["size"][1]), mode='bilinear', align_corners=False)
                # output = output.cpu()
                # output = output.numpy().transpose(0,2,3,1)
                # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
                seg_gt = np.asarray(label.numpy(), dtype=np.int)

                if args.save_predict:
                    for i in range(image.size(0)): 
                        # output_im = PILImage.fromarray(seg_pred[i])
                        # output_im.putpalette(palette)
                        # output_im.save(os.path.join(save_path, img_meta[i]["name"]+'.png'))
                        
                        seg_gt_test = np.asarray(label.numpy(), dtype=np.uint8)
                        output_im = PILImage.fromarray(seg_gt_test[i])
                        output_im.putpalette(palette)
                        output_im.save(os.path.join(save_path, img_meta[i]["name"]+'_GT'+'.png'))

                # if args.iou_type == 'boundary':
                #     for i in range(seg_pred.shape[0]): 
                #         seg_pred[i] = mask_to_boundary(seg_pred[i], dataset.num_classes, 
                #                             dilation_ratio=args.dilation_ratio, background=dataset.ignore_label)
                #         seg_gt[i] = mask_to_boundary(seg_gt[i], dataset.num_classes, 
                #                             dilation_ratio=args.dilation_ratio, background=dataset.ignore_label)
            
                # ignore_index = seg_gt != 255
                # seg_gt = seg_gt[ignore_index]
                # seg_pred = seg_pred[ignore_index]
                # confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, dataset.num_classes)

                # print_str = ' Iter{}/{}'.format(idx + 1, len(test_loader))
                # if idx >= fps_warmup:
                #     pure_inf_time += elapsed
                #     fps = (idx + 1 - fps_warmup) / pure_inf_time
                #     print_str = print_str + f' FPS: {fps:.2f} img / s'
                # pbar.set_description(print_str, refresh=False)
        # if engine.distributed:
        #     confusion_matrix = torch.from_numpy(confusion_matrix).contiguous().cuda()
        #     confusion_matrix = engine.all_reduce_tensor(confusion_matrix, norm=False).cpu().numpy()
        #     # confusion_matrix = torch.from_numpy(confusion_matrix)
        #     # confusion_matrix = engine.all_reduce_tensor(confusion_matrix, norm=False).cpu().numpy()
        # pos = confusion_matrix.sum(1)
        # res = confusion_matrix.sum(0)
        # tp = np.diag(confusion_matrix)
        # p = (tp/(res+1e-5)).mean()
        # r = (tp/(pos+1e-5)).mean() 

        # IU_array = (tp / np.maximum(1.0, pos + res - tp))
        # mean_IU = IU_array.mean()
        
        # getConfusionMatrixPlot(confusion_matrix)
        # if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
        #     print({'meanIU':mean_IU, 'IU_array':IU_array})
        #     model_path = os.path.dirname(args.restore_from)
        #     with open(os.path.join(model_path, 'result.txt'), 'a') as f:
        #         f.write('test with {}\n'.format(args.restore_from))
        #         f.write(json.dumps({'meanIU':mean_IU, 'IU_array':IU_array.tolist()}))
        #         f.write('\n')
        #         f.write(json.dumps({'meanP':p, 'p': (tp/(res+1e-5)).tolist()}))
        #         f.write('\n')
        #         f.write(json.dumps({'meanR': r, 'r': (tp / (pos + 1e-5)).tolist()}))
        #         f.write(f' FPS: {fps:.2f} img / s\n')
        #         f.write('--------\n')

if __name__ == '__main__':
    main()
