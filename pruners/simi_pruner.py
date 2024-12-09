import numpy as np
import torch
import torch.nn as nn
from .channel_pruner import ChannelPruner
import os
from tqdm import tqdm
import torch.nn.functional as F

def get_feature_hook(self, input, output):

    conv_output = output.detach()
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    conv_reshape_norm = torch.norm(conv_reshape, p=1, dim=-1)
    conv_reshape_normed = conv_reshape/(conv_reshape_norm.unsqueeze(-1)+1e-5)
    
    dice = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1], conv_reshape.shape[1]]) #先产生了一个全零的矩阵 准备计算单个算的时候的矩阵奇异值
    for i in range(conv_reshape.shape[0]):
        dice[i] = conv_reshape_normed[i].matmul(conv_reshape_normed[i].T)
    ci = torch.mean(-dice, dim=-1)
    self.ci = ci.mean(dim=0)

class simi_pruning():
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0
                m[1].register_forward_hook(get_feature_hook)

    def register_hook(self, model):
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0
                m[1].register_forward_hook(get_feature_hook)
                
    # def step(self, model):
    #     for m in model.named_modules():
    #         if m[0] in self.state_dict['eic']:
    #             flag = (m[1].weight.grad.data * m[1].weight.data > 0)
    #             grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach())+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]
    #             self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]]*self.r + grad_tmp*(1-self.r)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                # flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                # grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach())+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]]*self.r + m[1].ci*(1-self.r)

    def get_eic(self):
        return self.state_dict
    
    def export_eic(self, path):
        torch.save(self.state_dict, path)
                

class SIMIPruner(ChannelPruner):
    def __init__(self, global_percent=0.8, layer_keep=0.01, except_start_keys=['head.fc'], score_file='', **kwards):
        super(SIMIPruner, self).__init__(except_start_keys=except_start_keys)
        self.layer_keep = layer_keep
        self.global_percent = global_percent
        self.eic = torch.load(score_file,map_location='cpu')['eic']
        
    def get_bn_group(self, bn_layer):
        return 0 if bn_layer.startswith('backbone') else 1 
    
    def get_para_score(self, bn_layer):
        # score = self.eic[bn_layer].mean(dim=0)
        score = self.eic[bn_layer]
        return score
        
    def get_thresh(self):
        bn_size = [0,0] # 这里是在计算backbone部分和decoder部分的bn的output channel个数
        for bn_layer in self.norm_conv_links:
            if bn_layer not in self.except_layers:
                group = self.get_bn_group(bn_layer) # 这个东西是个flag，指示到底是backbone的还是decoder
                bn_size[group] += self.name2module[bn_layer].weight.data.shape[0]
        
        index = [0,0] # 这个部分是把bn层的importance score 付给bn_weights
        bn_weights = [torch.zeros(i) for i in bn_size]
        for bn_layer in self.norm_conv_links:
            if bn_layer not in self.except_layers:
                group = self.get_bn_group(bn_layer)     
                size = self.name2module[bn_layer].weight.data.shape[0] # 获得这个bn层的output channel size
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
                mask = weight_copy.gt(thresh[group]).float() # gt 是greater than
                
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

class simi2_pruning():
    def __init__(self, model, r=0.99, p=1, **kwards):
        self.r = r
        self.p = p
        self.state_dict = {'eic':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0
                m[1].register_forward_hook(get_feature_hook)
            if (isinstance(m[1], nn.Conv2d)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0

    def register_hook(self, model):
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0
                m[1].register_forward_hook(get_feature_hook)
                
    # def step(self, model):
    #     for m in model.named_modules():
    #         if m[0] in self.state_dict['eic']:
    #             flag = (m[1].weight.grad.data * m[1].weight.data > 0)
    #             grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach())+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]
    #             self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]]*self.r + grad_tmp*(1-self.r)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                # flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                # grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach())+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]
                if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)):
                    self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]]*self.r + m[1].ci*(1-self.r)
                # elif (isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear)) and not m[0] in model.ignore_prune_layer:
                #     self.state_dict['eic'][m[0]]

    def update_for_conv(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                # flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                # grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach())+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]
                if (isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear)) and not m[0] in model.ignore_prune_layer:
                    self.state_dict['eic'][m[0]] = m[1].weight.data.flatten(1).abs().pow(self.p).mean(1)
                    
    def get_eic(self):
        return self.state_dict
    
    def export_eic(self, path):
        torch.save(self.state_dict, path)

class SIMI2Pruner(ChannelPruner):
    def __init__(self, global_percent=0.8, layer_keep=0.01, except_start_keys=['head.fc'], score_file='', **kwards):
        super(SIMI2Pruner, self).__init__(except_start_keys=except_start_keys)
        self.layer_keep = layer_keep
        self.global_percent = global_percent
        self.eic = torch.load(score_file,map_location='cpu')['eic']

    def get_bn_group(self, bn_layer):
        return 0 if bn_layer.startswith('backbone') else 1 

    def get_para_score(self, bn_layer):
        # score = self.eic[bn_layer].mean(dim=0)
        bn_score = self.eic[bn_layer]
        conv_layer = self.norm_conv_links[bn_layer]
        conv_score = self.eic[conv_layer]
        result_score = bn_score * conv_score
        return result_score
    
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
                mask = weight_copy.gt(thresh[group]).float() # gt 是greater than
                
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


class simi3_pruning(simi_pruning):
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = torch.tensor(0.).to(m[1].weight.device)
                m[1].register_forward_hook(get_feature_hook)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                # print('flag:',flag.device)
                # print('another',torch.abs(m[1].weight.grad.data.detach()))
                # print('logical:', torch.logical_not(flag).device)
                # print('another',self.state_dict['eic'][m[0]])
                
                # if self.state_dict['eic'].get(m[0], None) is None:
                #     grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach()).to(device=flag.device)
                # else:
                #     grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach()).to(device=flag.device)+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]

                grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach()).to(device=flag.device)+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]        
                # self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * self.r + (m[1].ci.to(device=flag.device) * grad_tmp.to(device=flag.device)) * (1-self.r)        
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * self.r + (m[1].ci.to(device=flag.device) + 0.4*grad_tmp.to(device=flag.device)) * (1-self.r)

    def get_eic(self):
        return self.state_dict
    
    def export_eic(self, path):
        torch.save(self.state_dict, path)

class simi4_pruning(simi_pruning):

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)            
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * self.r + m[1].ci.to(device=flag.device) * (1-self.r)
    
    def update_last(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:         
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * m[1].weight.grad.data.abs().to(device=self.state_dict['eic'][m[0]].device)
                
class simi5_pruning(simi_pruning):

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)            
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * self.r + m[1].ci.to(device=flag.device) * (1-self.r)
    
    def conv_fitler_simi(self, module):
        flatten = module.weight.data.flatten(start_dim=1) # output_num, -1
        flatten_norm = torch.norm(flatten, p=1, dim=-1)
        flatten_normed = flatten/(flatten_norm.unsqueeze(-1)+1e-5)
        filter_simi = flatten_normed.matmul(flatten_normed.T).mean(dim=-1, keepdim=False)
        filter_simi = filter_simi.mean(dim=0, keepdim=False)
        
        return filter_simi
    
    def update_last(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                if isinstance(m[1], nn.Conv2d):         
                    filter_simi = self.conv_filter_simi(m[1])
                    self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * torch.exp(-filter_simi.abs()).to(device=self.state_dict['eic'][m[0]].device)
                else:
                    pass


class simi5add_pruning(simi5_pruning):
    def set_gamma(self, gamma):
        self.gamma = gamma
    def update_last(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                if isinstance(m[1], nn.Conv2d):         
                    filter_simi = self.conv_filter_simi(m[1])
                    res = torch.exp(-filter_simi.abs()).to(device=self.state_dict['eic'][m[0]].device)
                    self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] + self.gamma*res
                else:
                    pass

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

class SI(nn.Module):
    def __init__(self, inp, k_sobel, device):
        super(SI, self).__init__()

        self.inp = inp
        
        sobel_2D = get_sobel_kernel(k_sobel) #第一个方向的
        sobel_2D_trans = sobel_2D.T #第二个方向的
        sobel_2D = torch.from_numpy(sobel_2D).float().to(device)
        sobel_2D_trans = torch.from_numpy(sobel_2D_trans).float().to(device)
        # sobel_2D = torch.from_numpy(sobel_2D).to(device)
        # sobel_2D_trans = torch.from_numpy(sobel_2D_trans).to(device)
        sobel_2D = sobel_2D.unsqueeze(0).repeat(inp,1,1,1) # 在channel上重复
        sobel_2D_trans = sobel_2D_trans.unsqueeze(0).repeat(inp,1,1,1) # 在channel上重复
        
        self.vars = nn.ParameterList()
        self.vars.append(nn.Parameter(sobel_2D, requires_grad=False))
        self.vars.append(nn.Parameter(sobel_2D_trans, requires_grad=False))
    
    @torch.no_grad()
    def forward(self, x):
        grad_x = F.conv2d(x, self.vars[0], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        grad_y = F.conv2d(x, self.vars[1], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        value = torch.sqrt(grad_x**2 + grad_y**2)
        # value = 1/1.4142 * (torch.abs(grad_x) + torch.abs(grad_y))
        denom = value.shape[2]*value.shape[3]
        out = torch.sum(value**2, dim=(2,3))/denom - (torch.sum(value, dim=(2,3))/denom)**2
        return out ** 0.5 # 这个东西记录的是细节多少 这个很贼

@torch.no_grad()
def get_canny_hook(self, input, output):
    
    # conv_output = output.detach()
    # conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    # conv_reshape_norm = torch.norm(conv_reshape, p=1, dim=-1)
    # conv_reshape_normed = conv_reshape/(conv_reshape_norm.unsqueeze(-1)+1e-5)
    
    # dice = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1], conv_reshape.shape[1]]) #先产生了一个全零的矩阵 准备计算单个算的时候的矩阵奇异值
    # for i in range(conv_reshape.shape[0]):
    #     dice[i] = conv_reshape_normed[i].matmul(conv_reshape_normed[i].T)
    # ci = torch.mean(-dice, dim=-1)
    # self.ci = ci.mean(dim=0)
    
    if 'si' in dir(self):
        pass
    else:
        self.si = SI(output.shape[1], 3, self.weight.device)
    edge = self.si(output) # b,c 的 shape
    edge = edge.mean(dim=0)
    self.ci = edge
    
    

class simi6canny_pruning(simi_pruning):

    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = torch.tensor(0.).to(m[1].weight.device)
                m[1].register_forward_hook(get_canny_hook)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)            
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * self.r + m[1].ci.to(device=flag.device) * (1-self.r)
    
    # def conv_fitler_simi(self, module):
    #     flatten = module.weight.data.flatten(start_dim=1) # output_num, -1
    #     flatten_norm = torch.norm(flatten, p=1, dim=-1)
    #     flatten_normed = flatten/(flatten_norm.unsqueeze(-1)+1e-5)
    #     filter_simi = flatten_normed.matmul(flatten_normed.T).mean(dim=-1, keepdim=False)
    #     filter_simi = filter_simi.mean(dim=0, keepdim=False)
        
    #     return filter_simi
    
    # def update_last(self, model):
    #     for m in model.named_modules():
    #         if m[0] in self.state_dict['eic']:
    #             if isinstance(m[1], nn.Conv2d):         
    #                 filter_simi = self.conv_filter_simi(m[1])
    #                 self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * torch.exp(-filter_simi.abs()).to(device=self.state_dict['eic'][m[0]].device)
    #             else:
    #                 pass

@torch.no_grad()
def get_feature_canny_hook(self, input, output):
    
    conv_output = output.detach()
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    conv_reshape_norm = torch.norm(conv_reshape, p=1, dim=-1)
    conv_reshape_normed = conv_reshape/(conv_reshape_norm.unsqueeze(-1)+1e-5)
    
    dice = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1], conv_reshape.shape[1]]) #先产生了一个全零的矩阵 准备计算单个算的时候的矩阵奇异值
    for i in range(conv_reshape.shape[0]):
        dice[i] = conv_reshape_normed[i].matmul(conv_reshape_normed[i].T)
    ci = torch.mean(-dice, dim=-1)
    self.ci = ci.mean(dim=0)
    
    if 'si' in dir(self):
        pass
    else:
        self.si = SI(output.shape[1], 3, self.weight.device)
    edge = self.si(output) # b,c 的 shape
    edge = edge.mean(dim=0)
    self.ci = edge + self.ci.to(edge.device)

class simi7_pruning(simi_pruning):

    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = torch.tensor(0.).to(m[1].weight.device)
                m[1].register_forward_hook(get_feature_canny_hook)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)            
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * self.r + m[1].ci.to(device=flag.device) * (1-self.r)

def get_feature_canny_2_hook(self, input, output):
    
    conv_output = output.detach()
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    conv_reshape_norm = torch.norm(conv_reshape, p=1, dim=-1)
    conv_reshape_normed = conv_reshape/(conv_reshape_norm.unsqueeze(-1)+1e-5)
    
    dice = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1], conv_reshape.shape[1]]) #先产生了一个全零的矩阵 准备计算单个算的时候的矩阵奇异值
    for i in range(conv_reshape.shape[0]):
        dice[i] = conv_reshape_normed[i].matmul(conv_reshape_normed[i].T)
    ci = torch.mean(-dice, dim=-1)
    self.ci = ci.mean(dim=0)
    
    if 'si' in dir(self):
        pass
    else:
        self.si = SI(output.shape[1], 3, self.weight.device)
    edge = self.si(output) # b,c 的 shape
    edge = edge.mean(dim=0)
    self.ci = edge + 0.5*self.ci.to(edge.device)

class simi8_pruning(simi7_pruning):
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = torch.tensor(0.).to(m[1].weight.device)
                m[1].register_forward_hook(get_feature_canny_2_hook)
                
                
def get_feature_canny_01_hook(self, input, output):
    
    conv_output = output.detach()
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    conv_reshape_norm = torch.norm(conv_reshape, p=1, dim=-1)
    conv_reshape_normed = conv_reshape/(conv_reshape_norm.unsqueeze(-1)+1e-5)
    
    dice = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1], conv_reshape.shape[1]]) #先产生了一个全零的矩阵 准备计算单个算的时候的矩阵奇异值
    for i in range(conv_reshape.shape[0]):
        dice[i] = conv_reshape_normed[i].matmul(conv_reshape_normed[i].T)
    ci = torch.mean(-dice, dim=-1)
    self.ci = ci.mean(dim=0)
    
    if 'si' in dir(self):
        pass
    else:
        self.si = SI(output.shape[1], 3, self.weight.device)
    edge = self.si(output) # b,c 的 shape
    edge = edge.mean(dim=0)
    self.ci = edge + 0.1*self.ci.to(edge.device)

class simi801_pruning(simi7_pruning):
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        print("simi8 but edge is multiplied with 0.1")
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = torch.tensor(0.).to(m[1].weight.device)
                m[1].register_forward_hook(get_feature_canny_01_hook)

def get_feature_canny_025_hook(self, input, output):
    
    conv_output = output.detach()
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    conv_reshape_norm = torch.norm(conv_reshape, p=1, dim=-1)
    conv_reshape_normed = conv_reshape/(conv_reshape_norm.unsqueeze(-1)+1e-5)
    
    dice = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1], conv_reshape.shape[1]]) #先产生了一个全零的矩阵 准备计算单个算的时候的矩阵奇异值
    for i in range(conv_reshape.shape[0]):
        dice[i] = conv_reshape_normed[i].matmul(conv_reshape_normed[i].T)
    ci = torch.mean(-dice, dim=-1)
    self.ci = ci.mean(dim=0)
    
    if 'si' in dir(self):
        pass
    else:
        self.si = SI(output.shape[1], 3, self.weight.device)
    edge = self.si(output) # b,c 的 shape
    edge = edge.mean(dim=0)
    self.ci = 0.25*edge + self.ci.to(edge.device)

class simi8025_pruning(simi7_pruning):
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}}
        print("simi8 but edge is multiplied with 0.1")
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = torch.tensor(0.).to(m[1].weight.device)
                m[1].register_forward_hook(get_feature_canny_025_hook)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                # print('flag:',flag.device)
                # print('another',torch.abs(m[1].weight.grad.data.detach()))
                # print('logical:', torch.logical_not(flag).device)
                # print('another',self.state_dict['eic'][m[0]])
                
                # if self.state_dict['eic'].get(m[0], None) is None:
                #     grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach()).to(device=flag.device)
                # else:
                #     grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach()).to(device=flag.device)+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]

                grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach()).to(device=flag.device)+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]                
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]] * self.r + m[1].ci.to(device=flag.device) * grad_tmp.to(device=flag.device) * (1-self.r)