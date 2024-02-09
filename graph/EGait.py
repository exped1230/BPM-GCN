import numpy as np
import sys

sys.path.extend(['../'])
from graph import tools
import networkx as nx

# 0: root, 1: spine, 2: neck, 3: head
# 4: rshoulder, 5: relbow, 6: rhand
# 7: lshoulder, 8: lelbow, 9: lhand
# 10: rhip, 11: rknee, 12: rfoot
# 13: lhip, 14: lknee, 15: lfoot

# Edge format: (origin, neighbor)
num_node = 16
self_link = [(i, i) for i in range(num_node)]
inward = [(9, 8), (8, 7), (7, 2), (3, 2), (4, 2), (5, 4), (6, 5), (2, 1),
          (1, 0), (13, 0), (10, 0), (15, 14), (14, 13), (12, 11), (11, 10)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')
