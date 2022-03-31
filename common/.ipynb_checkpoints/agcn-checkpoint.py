import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import sys, os

num_node = 17
self_link = [(i, i) for i in range(num_node)]
parent_nodes = {
    0:[],
    1:[0],
    2:[1],
    3:[2],
    4:[0],
    5:[4],
    6:[5],
    7:[0],
    8:[7],
    9:[8],
    10:[9],
    11:[8],
    12:[11],
    13:[12],
    14:[8],
    15:[14],
    16:[15],
}

parent_link = []
for k,v in parent_nodes.items():
    i = k
    for j in v:
        parent_link.append([i, j])

child_link = [(j, i) for (i, j) in parent_link]
neighbor_link = parent_link + child_link

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, parent_link, child_link):
    I = edge2mat(self_link, num_node)
    Par = normalize_digraph(edge2mat(parent_link, num_node))
    Child = normalize_digraph(edge2mat(child_link, num_node))
    A = np.stack((I, Par, Child))
    A = torch.from_numpy(A).float()
    return A


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.parent_link = parent_link
        self.child_link = child_link
        self.neighbor_link = neighbor_link
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        A = get_spatial_graph(num_node, self_link, self.parent_link, self.child_link)
        return A

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),stride=(stride, 1))
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(kernel_size, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        # subset: num of label types
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        # B
        self.PA = nn.Parameter(A.float())
        nn.init.constant_(self.PA, 1e-6)
        # A
        self.A = Variable(A.float(), requires_grad=False)
        self.num_subset = num_subset

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        B, C, T, N = x.size()
        A = self.A.to(self.PA.device)
     
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1=A[i]
            # mul
            A2 = x.view(B, C * T, N)
            z = self.conv_d[i](torch.matmul(A2, A1).view(B, C, T, N))
            y = z + y if y is not None else z

        # Res
        y = self.bn(y)
        
        y += self.down(x)
        return self.relu(y)

class TCN_unit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        conv_layers = []
        bn_layers = []
        self.relu = ReLU(inplace = True)
        conv_layers.append(nn.Conv2d(channels, channels, kernel_size = (3, 1), stride = (3, 1)))
        bn_layers.append(nn.BatchNorm2d(channels))
        conv_layers.append(nn.Conv2d(channels, channels, kernel_size = (1, 1), stride = (1, 1)))
        bn_layers.append(nn.BatchNorm2d(channels))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
    def forward(self, x):
        #(B, C, T, V)
        res = x[:,:,1::3]
        x = self.relu(self.bn_layers[0](self.conv_layers[0](x)))
        x = self.relu(self.bn_layers[1](self.conv_layers[1](x))) + res
        return x
        
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels):
            self.residual = lambda x: x[:,:,(stride // 2)::stride]
    
    def forward(self, x):        
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class AGCNModel(nn.Module):
    def __init__(self,num_point=17, in_channels=9, channels = 128):
        super().__init__()
       
        self.graph=Graph()
        A = self.graph.A
        self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1))
        self.expand_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace = True)
        self.l1 = TCN_GCN_unit(channels, channels, A, stride = 3)
        #self.l2 = TCN_GCN_unit(channels, channels, A)
        self.l3 = TCN_GCN_unit(channels, channels, A, stride = 3)
        #self.l4 = TCN_GCN_unit(channels, channels, A)
        #self.l5 = TCN_GCN_unit(channels, channels, A, stride = 3)
        #self.l6 = TCN_GCN_unit(channels, channels, A)
        #self.l7 = TCN_GCN_unit(channels, channels, A, stride = 3)
        #self.l8 = TCN_GCN_unit(channels, channels, A)
        #self.l9 = TCN_GCN_unit(channels, channels, A, stride = 3)
        #self.l10 = TCN_GCN_unit(channels, channels, A)
        self.shrink = nn.Conv2d(channels, 3, (1, 1))
        bn_init(self.expand_bn, 1)
    def forward(self, x):
        out = []
        B, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()#(B, C, T, V)
     
        x = self.relu(self.expand_bn(self.expand_conv(x)))
        out.append(self.shrink(x).permute(0, 2, 3, 1).contiguous())#(B, T, V, C)
        
        x = self.l1(x)
        out.append(self.shrink(x).permute(0, 2, 3, 1).contiguous())
        #x = self.l2(x)
        x = self.l3(x)
        out.append(self.shrink(x).permute(0, 2, 3, 1).contiguous())
        #x = self.l4(x)
        #x = self.l5(x)
        #x = self.l6(x)
        #x = self.l7(x)
        #x = self.l8(x)
        #x = self.l9(x)
        #x = self.l10(x)
        
        return out

                
if __name__ == '__main__':
    model = Model()
    inp = torch.rand(1024,243, 17, 9).float()
    oup= model(inp)












