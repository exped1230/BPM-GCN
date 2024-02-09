import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


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
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
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
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_person=1, graph=None, graph_args=dict(), in_channels_p=3,in_channels_m=8):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn_p = nn.BatchNorm1d(num_person * in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(num_person * in_channels_m * num_point)

        self.l1_p = TCN_GCN_unit(in_channels_p, 64, A, residual=False)
        self.l1_m = TCN_GCN_unit(in_channels_m, 64, A, residual=False)

        self.l2_p = TCN_GCN_unit(64, 64, A)
        self.l3_p = TCN_GCN_unit(64, 64, A)
        self.l4_p = TCN_GCN_unit(64, 64, A)
        self.l5_p = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_p = TCN_GCN_unit(128, 128, A)
        self.l7_p = TCN_GCN_unit(128, 128, A)
        self.l8_p = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_p = TCN_GCN_unit(256, 256, A)
        self.l10_p = TCN_GCN_unit(256, 50*48, A)

        self.l2_m = TCN_GCN_unit(64, 64, A)
        self.l3_m = TCN_GCN_unit(64, 64, A)
        self.l4_m = TCN_GCN_unit(64, 64, A)
        self.l5_m = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_m = TCN_GCN_unit(128, 128, A)
        self.l7_m = TCN_GCN_unit(128, 128, A)
        self.l8_m = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_m = TCN_GCN_unit(256, 256, A)
        self.l10_m = TCN_GCN_unit(256, 256, A)

        self.fc1_classifier_p = nn.Linear(50*48, num_class)
        self.fc1_classifier_m = nn.Linear(256, num_class)
        # self.fc2_aff = nn.Linear(256, 31*48)

        nn.init.normal_(self.fc1_classifier_m.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / (50*48)))
        # nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (31*48)))
        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)

    def forward(self, x_p,x_m):
        N, C_p, T, V, M = x_p.size()
        N, C_m, T, V, M = x_m.size()

        x_p = x_p.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_p, T)
        x_m = x_m.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_m, T)


        x_p = self.data_bn_p(x_p)
        x_m = self.data_bn_m(x_m)

        x_p = x_p.view(N, M, V, C_p, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_p, T, V)
        x_m = x_m.view(N, M, V, C_m, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_m, T, V)

        x_p = self.l1_p(x_p)
        x_m = self.l1_m(x_m)

        x_p = self.l2_p(x_p)
        x_p = self.l3_p(x_p)
        x_p = self.l4_p(x_p)
        x_p = self.l5_p(x_p)
        x_p = self.l6_p(x_p)
        x_p = self.l7_p(x_p)
        x_p = self.l8_p(x_p)
        x_p = self.l9_p(x_p)
        x_p = self.l10_p(x_p)

        x_m = self.l2_m(x_m)
        x_m = self.l3_m(x_m)
        x_m = self.l4_m(x_m)
        x_m = self.l5_m(x_m)
        x_m = self.l6_m(x_m)
        x_m = self.l7_m(x_m)
        x_m = self.l8_m(x_m)
        x_m = self.l9_m(x_m)
        x_m = self.l10_m(x_m)

        # N*M,C,T,V
        c_new_m = x_m.size(1)
        x_m = x_m.view(N, M, c_new_m, -1)
        x_m = x_m.mean(3).mean(1)

        c_new_p = x_p.size(1)
        x_p = x_p.view(N, M, c_new_p, -1)
        x_p = x_p.mean(3).mean(1)

        # x_cat=torch.cat((x_m,x_p),1)

        return self.fc1_classifier_p(x_p),x_p[:,:(31*48)],self.fc1_classifier_m(x_m)
