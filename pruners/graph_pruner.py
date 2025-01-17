import numpy as np
import torch
import torch.nn as nn
from .channel_pruner import ChannelPruner
import os
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy

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

# this edge module is used in DCFP
class SI(nn.Module):
    def __init__(self, inp, k_sobel, device):
        super(SI, self).__init__()

        self.inp = inp
        
        sobel_2D = get_sobel_kernel(k_sobel) 
        sobel_2D_trans = sobel_2D.T 
        sobel_2D = torch.from_numpy(sobel_2D).float().to(device)
        sobel_2D_trans = torch.from_numpy(sobel_2D_trans).float().to(device)
        sobel_2D = sobel_2D.unsqueeze(0).repeat(inp,1,1,1) 
        sobel_2D_trans = sobel_2D_trans.unsqueeze(0).repeat(inp,1,1,1) 
        
        self.vars = nn.ParameterList()
        self.vars.append(nn.Parameter(sobel_2D, requires_grad=False))
        self.vars.append(nn.Parameter(sobel_2D_trans, requires_grad=False))
    
    @torch.no_grad()
    def forward(self, x):
        grad_x = F.conv2d(x, self.vars[0], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        grad_y = F.conv2d(x, self.vars[1], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        value = torch.sqrt(grad_x**2 + grad_y**2)
        denom = value.shape[2]*value.shape[3]
        out = torch.sum(value**2, dim=(2,3))/denom - (torch.sum(value, dim=(2,3))/denom)**2
        return out ** 0.5


def get_graph_edge_hook(self, input, output):

    conv_output = output.detach()
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    conv_reshape_norm = torch.norm(conv_reshape, p=1, dim=-1)
    conv_reshape_normed = conv_reshape/(conv_reshape_norm.unsqueeze(-1)+1e-5)
    
    dice = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1], conv_reshape.shape[1]]) 
    for i in range(conv_reshape.shape[0]):
        dice[i] = conv_reshape_normed[i].matmul(conv_reshape_normed[i].T)
    self.ci = torch.mean(dice, dim=0) 

    if 'si' in dir(self):
        pass
    else:
        self.si = SI(output.shape[1], 3, self.weight.device)
    edge = self.si(output) # b,c 的 shape
    edge = edge.mean(dim=0)
    self.edge = edge.to(self.ci.device)

class graph_edge_pruning():
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}, 'edge':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0
                self.state_dict['edge'][m[0]] = 0
                m[1].register_forward_hook(get_graph_edge_hook)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]]*self.r + m[1].ci*(1-self.r)
                self.state_dict['edge'][m[0]] = self.state_dict['edge'][m[0]]*self.r + m[1].edge*(1-self.r)

    def get_eic(self):
        return self.state_dict
    
    def export_eic(self, path):
        torch.save(self.state_dict, path)


class GraphEdgePruner(ChannelPruner):
    def __init__(self, global_percent=0.8, layer_keep=0.01, except_start_keys=['head.fc'], score_file='', **kwards):
        super(GraphEdgePruner, self).__init__(except_start_keys=except_start_keys)
        self.layer_keep = layer_keep
        self.global_percent = global_percent
        self.eic = torch.load(score_file,map_location='cpu')['eic']
        self.edge = torch.load(score_file,map_location='cpu')['edge']
        
    def get_bn_group(self, bn_layer):
        return 0 if bn_layer.startswith('backbone') else 1 
    
    def get_para_score(self, bn_layer):
        # score = self.eic[bn_layer].mean(dim=0)
        simi_matrix = deepcopy(self.eic[bn_layer]) # 这个应该是CxC形状的
        channel_num = simi_matrix.shape[0]
        if 'backbone' in bn_layer:
            score = 0.4 * self.edge[bn_layer] # this score can be adjusted for different datasets with large input images like Cityscapes (0.2 or 0.4)
            # print(bn_layer)
        else:
            score = 0. * self.edge[bn_layer]# for ade
        
        
        left_index_simi_matrix = set(range(channel_num))
        for i in range(channel_num):
            current_score = torch.sum(simi_matrix.abs(),dim=1)
            value, index = torch.sort(current_score, descending=True)
            p = 0
            while int(index[p]) not in left_index_simi_matrix:
                p += 1
            index_delete = int(index[p])
            score[index_delete] -= value[p]
            simi_matrix[index_delete] *=  0.0
            simi_matrix[:,index_delete] *= 0.0
            left_index_simi_matrix -= {index_delete}
            
        return score

    def get_thresh(self):
        bn_size = [0,0] 
        for bn_layer in self.norm_conv_links:
            if bn_layer not in self.except_layers:
                group = self.get_bn_group(bn_layer) # flag      backbone or decoder?
                bn_size[group] += self.name2module[bn_layer].weight.data.shape[0]
        
        index = [0,0] # assign importance score
        bn_weights = [torch.zeros(i) for i in bn_size]
        for bn_layer in self.norm_conv_links:
            if bn_layer not in self.except_layers:
                group = self.get_bn_group(bn_layer)     
                size = self.name2module[bn_layer].weight.data.shape[0] # output channel size of BN
                bn_weights[group][index[group]:(index[group] + size)] = self.get_para_score(bn_layer)
                index[group] += size
        
        thresh = [0,0] # for backbone and decoder
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
                weight_copy = self.get_para_score(bn_layer)
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


class graph_edgev2_pruning(graph_edge_pruning):
    def __init__(self, model, r=0.99, **kwards):
        self.r = r
        self.state_dict = {'eic':{}, 'edge':{}, 'grad':{}}
        for m in model.named_modules():
            if (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm) or isinstance(m[1], nn.LayerNorm)) and not m[0] in model.ignore_prune_layer:
                self.state_dict['eic'][m[0]] = 0
                self.state_dict['edge'][m[0]] = 0
                self.state_dict['grad'][m[0]] = 0.
                m[1].register_forward_hook(get_graph_edge_hook)

    def step(self, model):
        for m in model.named_modules():
            if m[0] in self.state_dict['eic']:
                # flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                # grad_tmp = flag*torch.abs(m[1].weight.grad.data.detach())+ torch.logical_not(flag)*self.state_dict['eic'][m[0]]
                self.state_dict['eic'][m[0]] = self.state_dict['eic'][m[0]]*self.r + m[1].ci*(1-self.r)
                self.state_dict['edge'][m[0]] = self.state_dict['edge'][m[0]]*self.r + m[1].edge*(1-self.r)
                flag = (m[1].weight.grad.data * m[1].weight.data > 0)
                grad_tmp = flag * (torch.abs(m[1].weight.grad.data.detach()).to(device=flag.device))
                grad_tmp += torch.logical_not(flag) * self.state_dict['grad'][m[0]]
                self.state_dict['grad'][m[0]] = self.state_dict['grad'][m[0]]*self.r + grad_tmp.to(device=flag.device)*(1-self.r)
              

class GraphEdgeV2Pruner(GraphEdgePruner):
    def __init__(self, global_percent=0.8, layer_keep=0.01, except_start_keys=['head.fc'], score_file='', **kwards):
        super(GraphEdgePruner, self).__init__(except_start_keys=except_start_keys)
        self.layer_keep = layer_keep
        self.global_percent = global_percent
        self.eic = torch.load(score_file,map_location='cpu')['eic']
        self.edge = torch.load(score_file,map_location='cpu')['edge']
        self.grad = torch.load(score_file,map_location='cpu')['grad']
        
    def get_para_score(self, bn_layer):
        # score = self.eic[bn_layer].mean(dim=0)
        simi_matrix = deepcopy(self.eic[bn_layer]) # 这个应该是CxC形状的
        channel_num = simi_matrix.shape[0]
        if 'backbone' in bn_layer:
            score = 0.4 * self.grad[bn_layer]
            # print("larger than 35")
            # print(bn_layer)
        else:
            score = 0.4 * self.grad[bn_layer]# for ade
        # if 'backbone' in bn_layer:
        #     score = 0.4 * self.edge[bn_layer]
        #     # print("larger than 35")
        #     # print(bn_layer)
        # else:
        #     # score = 0.4 * self.grad[bn_layer]# for ade 0726 46.8
        #     score = self.grad[bn_layer]# for ade
            
        
        
        left_index_simi_matrix = set(range(channel_num))
        for i in range(channel_num):
            current_score = torch.sum(-(simi_matrix.abs()),dim=1)
            # 这里就已经转化成了redundancy取反，就是不相似度 越像越负
            value, index = torch.sort(current_score, descending=False) # 升序排序，就是跟别人最相似的排在前面
            p = 0
            # value_delete = value[0]
            # index_delete = index[0]
            while int(index[p]) not in left_index_simi_matrix:
                p += 1
            index_delete = int(index[p])
            score[index_delete] += value[p]
            # 这个就已经被删掉了 下一次的simi_matrix里面就不应该有这个了
            # 所以这些地方就都置为0
            simi_matrix[index_delete] *=  0.0
            simi_matrix[:,index_delete] *= 0.0
            left_index_simi_matrix -= {index_delete}
            
        return score