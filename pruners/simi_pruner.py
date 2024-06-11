import numpy as np
import torch
import torch.nn as nn
from .channel_pruner import ChannelPruner


class simi_pruning():
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0
                
    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach())+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]]*self.r + grad_tmp*(1-self.r)
            
    def get_eic(self):
        return self.state_dict
    
    def export_eic(self, path):
        torch.save(self.state_dict, path)
                

class DCFPPruner(ChannelPruner):
    def __init__(self, global_percent=0.8, layer_keep=0.01, except_start_keys=['head.fc'], score_file='', **kwards):
        super(DCFPPruner, self).__init__(except_start_keys=except_start_keys)
        self.layer_keep = layer_keep
        self.global_percent = global_percent
        self.eic = torch.load(score_file,map_location='cpu')['eic']
        
    def get_bn_group(self, bn_layer):
        return 0 if bn_layer.startswith('backbone') else 1 
    
    def get_para_score(self, bn_layer):
        score = self.eic[bn_layer]
        return score
        
    def get_thresh(self):
        bn_size = [0,0]
        for bn_layer in self.norm_conv_links:
            if bn_layer not in self.except_layers:
                group = self.get_bn_group(bn_layer)
                bn_size[group] += self.name2module[bn_layer].weight.data.shape[0]
        
        index = [0,0]
        bn_weights = [torch.zeros(i) for i in bn_size]
        for bn_layer in self.norm_conv_links:
            if bn_layer not in self.except_layers:
                group = self.get_bn_group(bn_layer)     
                size = self.name2module[bn_layer].weight.data.shape[0]
                bn_weights[group][index[group]:(index[group] + size)] = self.get_para_score(bn_layer)
                index[group] += size
        
        thresh = [0,0]
        for i in range(len(thresh)):
            if bn_weights[i].numel()>0:
                sorted_bn, sorted_index = torch.sort(bn_weights[i])
                thresh_index = int(bn_size[i] * self.global_percent)
                thresh[i] = sorted_bn[thresh_index]
        # print('Threshold: {}.'.format(thresh))
        return thresh

    def gen_channel_mask(self):
        thresh = self.get_thresh()
        pruned = 0
        total = 0
        for bn_layer, conv_layer in self.norm_conv_links.items():
            channels = self.name2module[bn_layer].weight.data.shape[0]
            if conv_layer not in self.except_layers:
                weight_copy = self.get_para_score(bn_layer) #
                group = self.get_bn_group(bn_layer)
                mask = weight_copy.gt(thresh[group]).float()
                
                min_channel_num = int(channels * self.layer_keep) if int(channels * self.layer_keep) > 0 else 1
                if int(torch.sum(mask)) < min_channel_num: 
                    _, sorted_index_weights = torch.sort(weight_copy,descending=True)
                    mask[sorted_index_weights[:min_channel_num]]=1. 

                self.name2module[conv_layer].out_mask = mask.reshape(self.name2module[conv_layer].out_mask.shape)

                remain = int(mask.sum())
            else:
                remain = channels
            pruned = pruned + channels - remain
            # print('layer {} \t total channel: {} \t remaining channel: {}'.format(conv_layer, channels, remain))
            
            total += channels

        # prune_ratio = pruned / total
        # print('Prune channels: {}\t Prune ratio: {}'.format(pruned, prune_ratio))