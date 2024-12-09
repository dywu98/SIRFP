import random
import copy


CLIQUE = set([])
W = 0
W_STAR = 0
CLIQUE_STAR = set([])


def main_mewc(simi_matrix, b):
    global CLIQUE, W, W_STAR, CLIQUE_STAR
    n = simi_matrix.shape[0]

    ###################
    #### init
    u = [i for i in range(n)] # 因为每个点度都一样，所以直接按顺序排了
    l = [i for i in range(n)]
    ###################
    #### b and b 分支定界开始
    
    def prune(simi_matrix, up, lp):
        global CLIQUE, W, W_STAR, CLIQUE_STAR
        q = False
        w_quo = 0
        w_line = lp[-1]
        gamma = [0]*len(up)
        for i in range(len(up)):
            v = up[i]
            for node_element in CLIQUE:
                u_element = node_element
                gamma[i] += simi_matrix[u_element][v]
            delta_i = [None] * len(up)
            j = 0
            for j in range(len(up)):
                u_element = up[j]
                if u_element != v:
                    delta_i[j] = 1/2*(simi_matrix[u_element][v])
                else:
                    delta_i[j] = 0
            delta_i = sorted(delta_i,reverse=True)
            for k in range(w_line-1):
                gamma[i] += delta_i[k]
        gamma = sorted(gamma, reverse=True)
        k = 0
        for k in range(w_line):
            w_quo += gamma[k]
        if w_quo <= (W_STAR-W):
            q = True
        return q
        
        
    def branch(simi_matrix, b, u, l):
        global CLIQUE, W, W_STAR, CLIQUE_STAR
        u = copy.deepcopy(u)
        l = copy.deepcopy(l)
        while u and l :
            q = prune(simi_matrix, u, l)
            if q:
                return 
            else:
                v = l[-1]
                w_v = 0
                for node in CLIQUE:
                    w_v += simi_matrix[node][v]
                CLIQUE = CLIQUE | set([v])
                if len(CLIQUE)>b:
                    CLIQUE = CLIQUE - set([v])
                    if W>W_STAR :
                        CLIQUE_STAR = copy.deepcopy(CLIQUE)
                        W_STAR = copy.deepcopy(W)
                        # print('The best is:')
                        # print(CLIQUE_STAR,W_STAR)
                    return 
                W += w_v
                u_v = []
                for i in u: # line 11
                    if i!=v:
                        u_v.append(i)
                if u_v != []:
                    l_v = subcolor(simi_matrix, u_v)
                    branch(simi_matrix, b, u_v, l_v)
                elif W>W_STAR :
                    CLIQUE_STAR = copy.deepcopy(CLIQUE)
                    W_STAR = W
                CLIQUE = CLIQUE - set([v])
                W -= w_v
            l = l.remove(v)
            u = u.remove(v)
        return
    
    def subcolor(simi_matrix, u_v):
        global CLIQUE, W, W_STAR, CLIQUE_STAR
        l_v = [None] * len(u_v)
        K = 0
        I = [[]] * len(u_v)
        for i in range(len(u_v)):
            u_element = u_v[i]
            k = 0
            while len(I[k]) != 0:
                k = k+1
            if k > K :
                K = k
                I[k] = []
            I[k] = I[k] + [u_element]
        i = 0
        for k in range(K+1):
            for j in range(len(I[k])):
                l_v[i] = I[k][j]
                # color(l_v[i]) = k
                i = i +1
        return l_v

    branch(simi_matrix, b, u, l)
    return (CLIQUE_STAR, W_STAR)

# import torch
# simi_matrix = torch.Tensor([
#     [0, 5, 1],
#     [5, 0, 3],
#     [1, 3, 0]
# ])

# b = 2

# main_mewc(simi_matrix, b)
# print('The best is last:')
# print(CLIQUE_STAR,W_STAR)
        
    